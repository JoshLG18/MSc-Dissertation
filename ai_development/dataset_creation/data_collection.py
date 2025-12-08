# ---- Script to collect all data from the various sources ----
# -------------------------------------------------------------

# Loading in all necessary libraries
import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv
from fredapi import Fred
from pytrends.request import TrendReq
import time
from typing import Optional
import requests
import io
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random

# stop warnings
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------

# Define Constants
START = '2004-04-01'
END = '2025-03-31'
TICKER = 'SPY'  

# -------------------------------------------------------------

def collect_OHLVC(ticker): # Collect OHLCV data from Yahoo Finance
    print(f"--- Collecting OHLVC data for {TICKER} ---")
    df = yf.download(TICKER, start=START, end=END) # Download data from Yahoo Finance
    print(f"Collected {len(df)} days of data")
    return df

# -------------------------------------------------------------

def collect_Macro():
    load_dotenv()

    # Verify FRED API key
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if FRED_API_KEY is None:
        raise RuntimeError("FRED_API_KEY not set")

    print("--- Collecting Macro Data from FRED ---")
    fred = Fred(api_key=FRED_API_KEY) # Initialize FRED client

    # Define the data we want to collect
    ids = {
        "FED_FUNDS": "FEDFUNDS",
        "MatRate_10Y": "DGS10",
        "MatRate_2Y": "DGS2",
        "Mat_Rate_CURVE": "T10Y2Y",
        "CPI": "CPIAUCSL",
        "CORE_CPI": "CPILFESL",
        "UNEMPLOYMENT": "UNRATE",
        "JOBLESS_CLAIMS": "ICSA",
        "GDP": "GDP",
        "CONSUMER_SENTIMENT": "UMCSENT",
        "VIX": "VIXCLS"
    }

    # Collect each series with progress bar
    series_dict = {}

    for name, sid in tqdm(ids.items(), desc="Fetching FRED series", unit="series"): # Loop through each series ID
        try: # Try to collect the data
            data = fred.get_series( # Fetch the series data from the API
                sid,
                observation_start=START,
                observation_end=END
            )
            series_dict[name] = data # add the series to the dictionary

        except Exception as e: # if an error occurs, log it and skip the series
            tqdm.write(f"FAILED to collect {name} ({sid}): {e}")
            tqdm.write(f"Skipping {name}...")


    # Create DataFrame with daily frequency
    df = pd.DataFrame(index=pd.date_range(START, END, freq='D'))

    # Add each series and forward-fill
    for name, series in series_dict.items():
        df[name] = series # Add series to DataFrame
        df[name] = df[name].ffill() # Forward-fill missing values

    # Filter to business days only (Mon-Fri) to match market data
    df = df[df.index.dayofweek < 5]

    print(f"Collected {len(df)} days of macro data")
    return df # Return the final DataFrame

# -------------------------------------------------------------

def collect_google_trends(keywords, start_date, end_date, geo="US", sleep=15, backoff=120, max_retries=5):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    all_trends = []
    pytrends = TrendReq(hl="en-US", tz=360, timeout=(30, 60))
    
    # Use 269-day chunks with 30-day overlap
    chunk_size = 269
    overlap = 30
    
    total_days = (end - start).days # work out total days to collect
    total_chunks = (total_days // (chunk_size - overlap)) + 1 # work out total chunks needed
    
    print(f"--- Google Trends Collection: {start.date()} to {end.date()} ---")
    print(f"Chunks: {total_chunks} (269-day windows, 30-day overlap)")
    
    pbar = tqdm(total=total_chunks, desc="Google Trends", unit="chunk") # progress bar
    
    current_start = start.normalize() # set current start date
    previous_chunk = None # to hold previous chunk for normalisation
    
    while current_start <= end: # loop until end date reached
        current_end = min(current_start + pd.Timedelta(days=chunk_size), end) # define current end date
        timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
        
        pbar.set_postfix_str(f"{current_start.date()}") # update progress bar
        
        chunk_trends = None 
        retries = 0
        
        # Fetch chunk with retry logic
        while retries <= max_retries:
            try:
                pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop="")
                chunk_trends = pytrends.interest_over_time() # fetch the data
                
                if chunk_trends is not None and not chunk_trends.empty:
                    if "isPartial" in chunk_trends.columns:
                        chunk_trends = chunk_trends.drop(columns=["isPartial"])
                    break
                    
            except Exception as e: # handle exceptions and rate limiting
                msg = str(e).lower()
                is_rate_limited = (
                    ("429" in msg or "rate limit" in msg or "too many" in msg or 
                     "quota" in msg or "limit exceeded" in msg) or
                    (hasattr(e, "response") and getattr(e.response, "status_code", None) == 429)
                )
                
                if is_rate_limited:
                    retries += 1
                    wait = (backoff * (2 ** retries)) + random.uniform(30, 90)
                    tqdm.write(f"Rate-limited (attempt {retries}/{max_retries}). Waiting {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"ERROR {timeframe}: {e}")
                    break
        
        if retries > max_retries: # exceeded max retries
            tqdm.write(f"Max retries exceeded for {timeframe}. Skipping.")
            chunk_trends = None
        
        # Normalisation: Scale current chunk to previous
        if chunk_trends is not None and not chunk_trends.empty:
            if previous_chunk is not None:
                # Find overlapping period
                overlap_start = current_start
                overlap_end = overlap_start + pd.Timedelta(days=overlap)
                
                # Get overlap data from both chunks
                prev_overlap = previous_chunk[
                    (previous_chunk.index >= overlap_start) & 
                    (previous_chunk.index < overlap_end)
                ]
                curr_overlap = chunk_trends[
                    (chunk_trends.index >= overlap_start) & 
                    (chunk_trends.index < overlap_end)
                ]
                
                # Scale each keyword independently
                if len(prev_overlap) > 0 and len(curr_overlap) > 0:
                    for keyword in keywords:
                        # use median to reduce outlier impact
                        prev_median = prev_overlap[keyword].median()
                        curr_median = curr_overlap[keyword].median()
                        
                        # Apply scaling factor to entire chunk 
                        if curr_median > 0:  # Avoid division by zero
                            scale_factor = prev_median / curr_median
                            chunk_trends[keyword] = chunk_trends[keyword] * scale_factor
            
            # Store scaled chunk
            all_trends.append(chunk_trends)
            previous_chunk = chunk_trends.copy()
        
        # Move to next chunk (with overlap)
        current_start = current_start + pd.Timedelta(days=(chunk_size - overlap))
        pbar.update(1)
        time.sleep(sleep + random.uniform(1, 5))
    
    pbar.close()
    
    if not all_trends:
        return pd.DataFrame()
    
    # Combine chunks
    result = pd.concat(all_trends).sort_index()
    
    # Average duplicate dates from overlaps 
    result = result.groupby(result.index).median()
    
    # Trim to exact date range
    result = result[(result.index >= start) & (result.index <= end)]
    
    # Handle any all-zero rows - interpolate linearly
    all_zero_mask = (result == 0).all(axis=1)
    if all_zero_mask.sum() > 0:
        print(f"Fixing {all_zero_mask.sum()} days with all zeros")
        result[all_zero_mask] = np.nan
        result = result.interpolate(method='linear')
        result = result.bfill().ffill()
    
    # Normalisation to 0-100 scale so it matches google trends scale
    print("\nApplying 0-100 normalisation...") 
    for keyword in result.columns: # scale each keyword column
        col_min = result[keyword].min() 
        col_max = result[keyword].max()
        col_range = col_max - col_min
        
        if col_range > 0:
            result[keyword] = ((result[keyword] - col_min) / col_range) * 100
            print(f"  {keyword}: [{col_min:.2f}, {col_max:.2f}] → [0, 100]")
        else:
            # If no variation, raise error -> this reall shouldn't happen but just in case
            raise ValueError(f"Column {keyword} has no variation (min == max == {col_min}). Cannot normalize.")

    # Verify that all columns are now in 0-100 range
    print("\nVerification:")
    for col in result.columns:
        print(f"  {col}: min={result[col].min():.1f}, max={result[col].max():.1f}, mean={result[col].mean():.1f}")
    
    print(f"\nCollected {len(result)} days of normalised Google Trends data (0-100 scale)")
    return result

# -------------------------------------------------------------
# Functions to collect news articles from NYT API

def rank_by_tfidf(articles, current_date=None):
    """Rank articles by financial relevance and recency"""
    if not articles:
        return []
    
    # define a document for the TF-IDF vectoriser to learn financial terms
    financial_corpus = [
        "stock market trading equity shares investment portfolio dividend yield",
        "recession economic downturn crisis financial collapse depression",
        "inflation prices consumer index monetary policy interest rates fed",
        "unemployment jobs employment labor market workforce layoffs hiring",
        "federal reserve central bank quantitative easing tightening rates",
        "earnings revenue profit corporate quarterly results guidance",
        "volatility risk uncertainty market fluctuation correction crash",
        "debt credit bonds treasury securities yield curve spreads",
        "GDP growth economic expansion productivity output manufacturing",
        "bull market bear market rally correction selloff",
        "S&P 500 Dow Jones NASDAQ index futures options derivatives",
        "commodities oil gold silver copper currency forex dollar euro",
        "banking financial sector lending mortgage credit default",
        "trade tariffs imports exports deficit surplus international",
        "housing real estate property prices mortgage rates construction",
        "retail consumer spending sales earnings holiday shopping",
        "technology sector tech stocks FAANG innovation disruption",
        "healthcare pharmaceutical biotech drug approval FDA",
        "energy sector oil gas renewable solar wind utilities",
        "mergers acquisitions IPO buyout private equity venture capital"
    ]
    
    # Prepare texts for TF-IDF vectorizer - strip and lowercase
    texts = [(a.get('headline', '') + ' ' + a.get('abstract', '')).lower().strip() for a in articles]
    
    # TF-IDF - fit the vectorizer on financial corpus + article texts
    vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(financial_corpus + texts)
    
    # Calculate similarity to financial centroid
    centroid = np.mean(tfidf[:len(financial_corpus)].toarray(), axis=0)
    article_vecs = tfidf[len(financial_corpus):].toarray()
    similarities = np.dot(article_vecs, centroid) / (np.linalg.norm(article_vecs, axis=1) * np.linalg.norm(centroid) + 1e-10)
    
    # Calculate recency scores based on article age
    reference_date = current_date if current_date else pd.Timestamp.now()
    recency = []
    for a in articles:
        try:
            hours_old = (reference_date - pd.to_datetime(a.get('date'))).total_seconds() / 3600
            recency.append(np.exp(-hours_old / 12))
        except:
            recency.append(0.5)
    
    # combine the relevance and recency scores to get a final score
    for i, article in enumerate(articles):
        relevance = float(similarities[i]) * 100
        
        # 20/80 split to prioritize recency more
        article['_score'] = (0.2 * relevance) + (0.8 * recency[i] * 100)
    
    return sorted(articles, key=lambda x: x.get('_score', 0), reverse=True)

def collect_nyt_top_daily_articles(start_date, end_date, articles_per_day=15):
    load_dotenv() # load environment variables

    NYT_API_KEY = os.getenv("NEWS_API_KEY") # get NYT API key

    print(f"--- Collecting Top {articles_per_day} Articles Per Day: {start_date} to {end_date} ---")

    # Date range setup
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    current = pd.to_datetime(start_date).replace(day=1)

    all_articles = []

    # Calculate total months for progress bar
    total_months = ((pd.to_datetime(end_date).year - current.year) * 12 +
                    pd.to_datetime(end_date).month - current.month + 1)

    # Add progress bar
    pbar = tqdm(total=total_months, desc="NYT Articles", unit="month")

    while current <= pd.to_datetime(end_date):
        year, month = current.year, current.month
        url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json" # define the archive URL for the given year and month

        pbar.set_postfix_str(f"{year}-{month:02d}") 

        for attempt in range(3): # Retry up to 3 times
            try:
                resp = requests.get(url, params={'api-key': NYT_API_KEY}, timeout=120) # make the API request
                resp.raise_for_status() # raise error for bad responses

                daily_articles = {} # Store articles by date

                for article in resp.json()['response']['docs']: # Loop through each article in the response
                    pub_date = pd.to_datetime(article['pub_date']).date() # get publication date
                    if not (start <= pub_date <= end): # check if publication date is within range
                        continue

                    headline = article.get('headline', {}).get('main', '') # get headline
                    abstract = article.get('abstract', '') # get abstract

                    # Combine text fields and clean
                    text = (headline + ' ' + abstract + ' ').lower().strip()

                    if len(text.strip()) < 50: # Skip very short articles
                        continue

                    # Apply section-based filtering (only business/finance related)
                    section = article.get('section_name', '').lower()
                    news_desk = article.get('news_desk', '').lower()

                    if section in ['business', 'business day', 'markets', 'economy', 'dealbook'] or \
                       news_desk in ['business', 'business day', 'markets', 'economy', 'dealbook']:
                        
                        if pub_date not in daily_articles:
                            daily_articles[pub_date] = []

                        daily_articles[pub_date].append({
                            'date': pub_date,
                            'headline': headline,
                            'abstract': abstract,
                            'section': article.get('section_name', ''),
                            'news_desk': article.get('news_desk', ''),
                            'word_count': article.get('word_count', 0),
                            'web_url': article.get('web_url', '')
                        })

                # Rank articles for each day using TF-IDF
                for date, articles in daily_articles.items():
                    # Use the article's date as reference for recency calculation
                    ranked_articles = rank_by_tfidf(articles, current_date=pd.Timestamp(date) + pd.Timedelta(hours=23, minutes=59))
                    all_articles.extend(ranked_articles[:articles_per_day])

                break

            except Exception as e:
                if attempt < 2:
                    time.sleep(30 * (attempt + 1))
                else:
                    tqdm.write(f"{year}-{month:02d}: Skipped after 3 attempts - {str(e)}")

        current += pd.DateOffset(months=1)
        pbar.update(1)
        time.sleep(15)

    pbar.close()

    df = pd.DataFrame(all_articles)
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove internal scoring column before saving
        df = df.drop(columns=['_score'], errors='ignore')

    print(f"Complete: {len(df):,} articles")
    return df

# -------------------------------------------------------------

def save_data():
    print("STARTING DATA COLLECTION")
    print("=" * 60)

    # Get the absolute path to the project root (parent of 'code' directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data', 'raw')

    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)

    # Collect all data
    ohlvc_data = collect_OHLVC("SPY")
    print("=" * 60)

    macro_data = collect_Macro()
    print("=" * 60)

    trends_keywords = ["Debt", "Stocks", "Inflation", "Unemployment", 'S&P 500']
    google_trends_data = collect_google_trends(trends_keywords, START, END)
    print("=" * 60)

    nyt_data = collect_nyt_top_daily_articles(START, END, articles_per_day=15)
    print("=" * 60)

    # Save all data with absolute paths
    print("\nSAVING DATA TO CSV FILES")
    print(f"Saving to: {data_dir}")

    try:
        ohlvc_path = os.path.join(data_dir, "US_OHLVC_Data.csv")
        ohlvc_data.to_csv(ohlvc_path)
        print(f"Saved: {ohlvc_path}")
    except Exception as e:
        print(f"Failed to save US_OHLVC_Data.csv: {e}")

    try:
        macro_path = os.path.join(data_dir, "US_Macro_Data.csv")
        macro_data.to_csv(macro_path)
        print(f"Saved: {macro_path}")
    except Exception as e:
        print(f"Failed to save US_Macro_Data.csv: {e}")

    try:
        trends_path = os.path.join(data_dir, "US_Google_Trends_Data.csv")
        google_trends_data.to_csv(trends_path)
        print(f"Saved: {trends_path}")
    except Exception as e:
        print(f"Failed to save US_Google_Trends_Data.csv: {e}")

    try:
        nyt_path = os.path.join(data_dir, "NYT_Top_Daily_Articles.csv")
        nyt_data.to_csv(nyt_path, index=False)
        print(f"Saved: {nyt_path}")
    except Exception as e:
        print(f"Failed to save NYT_Top_Daily_Articles.csv: {e}")

    print("=" * 60)
    print("ALL DATA COLLECTION COMPLETE ✓")
    print(f"Files saved in: {data_dir}")


# -------------------------------------------------------------
# Main execution to collect and save all data
if __name__ == '__main__':
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    print("Current working directory:", os.getcwd())
    print("Script location:", current_dir)
    print("Project root:", project_root)

    # Create data directory structure
    data_dir = os.path.join(project_root, 'data', 'raw')

    try:
        os.makedirs(data_dir, exist_ok=True)
        print(f"Data directory ready: {data_dir}")
    except Exception as e:
        print(f"ERROR creating directory {data_dir}: {e}")
        exit(1)

    save_data()

# ------ References ------

# Normalisation Logic - https://stats.stackexchange.com/questions/193936/google-trends-stitching-90-day-periods-of-daily-data-together
# NYT Article Collection - https://developer.nytimes.com/docs/archive-product/1/overview
# News Article Relevance Ranking Logic - https://www.sciencedirect.com/science/article/abs/pii/0306457388900210?via%3Dihub
# FRED Macro Data - https://fred.stlouisfed.org/docs/api/fred/
# Yahoo Finance OHLCV Data - https://pypi.org/project/yfinance/
# Google Trends Data Collection API - https://pypi.org/project/pytrends/

# -------------------------------------------------------------