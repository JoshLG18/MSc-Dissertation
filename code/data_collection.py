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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------------------------------------

# Define Constants
START = '2004-04-01'
END = '2025-10-31'

# -------------------------------------------------------------

def collect_OHLVC(ticker): # Collect OHLCV data from Yahoo Finance
    print(f"--- Collecting OHLVC data for {ticker} ---")
    df = yf.download(ticker, start=START, end=END) # Download data from Yahoo Finance
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
            data = fred.get_series(
                sid, 
                observation_start=START, 
                observation_end=END
            )
            series_dict[name] = data
            
        except Exception as e: # if an error occurs, log it and skip the series
            tqdm.write(f"FAILED to collect {name} ({sid}): {e}")
            tqdm.write(f"Skipping {name}...")

    # Create DataFrame with daily frequency
    df = pd.DataFrame(index=pd.date_range(START, END, freq='D'))
    
    # Add each series and forward-fill
    # Only loop through the keys that actually exist in our collected dictionary
    for name, series in series_dict.items():
        df[name] = series
        df[name] = df[name].ffill()
    
    # Filter to business days only (Mon-Fri)
    df = df[df.index.dayofweek < 5]
    
    print(f"Collected {len(df)} days of macro data")
    return df # Return the final DataFrame

# -------------------------------------------------------------

def collect_google_trends(keywords, start_date, end_date, geo="US", sleep=15, backoff=120, max_retries=5):    
    import random
    
    start = pd.to_datetime(start_date) # Convert to datetime
    end = pd.to_datetime(end_date) # Convert to datetime

    pytrends = TrendReq(hl="en-US", tz=360, timeout=(30, 60)) # Initialize pytrends client
    all_trends = []
    current_start = start.normalize() # Start from the beginning of the day
    
    # Calculate total chunks for progress bar
    total_days = (end - start).days
    chunk_size = 30
    total_chunks = (total_days // chunk_size) + 1
    
    print(f"--- Starting Google Trends Collection for {start.date()} to {end.date()} ---")
    print(f"Total chunks to fetch: {total_chunks}")
    
    # Add progress bar
    pbar = tqdm(total=total_chunks, desc="Google Trends", unit="chunk")
    
    while current_start <= end: # Loop until we reach the end date
        current_end = current_start + pd.Timedelta(days=30) # Define chunk end date
        if current_end > end: # Adjust if it exceeds the overall end date
            current_end = end
        
        # Define timeframe string
        timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
        pbar.set_postfix_str(f"{timeframe}") # Update progress bar with current timeframe
        
        chunk_trends: Optional[pd.DataFrame] = None
        retries = 0
        
        # try to fetch data with retries
        while retries <= max_retries:
            try:
                pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop="") # Build payload for the given timeframe
                chunk_trends = pytrends.interest_over_time() # Fetch data
                break
            
            except Exception as e: # Handle exceptions
                msg = str(e).lower()
                
                is_429 = False
                try: # Check if the exception is a rate limit error
                    if hasattr(e, "response") and getattr(e.response, "status_code", None) == 429:
                        is_429 = True
                except Exception:
                    pass

                # Check for rate limit errors
                if "429" in msg or "rate limit" in msg or "too many" in msg or is_429:
                    retries += 1
                    wait = (backoff * retries) + random.uniform(0, 30)
                    tqdm.write(f"WARNING: Rate-limited (attempt {retries}/{max_retries}). Waiting {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                else:
                    tqdm.write(f"ERROR: Unexpected error for {timeframe}. Skipping. Error: {e}")
                    chunk_trends = None
                    break
        
        if retries > max_retries: # If max retries exceeded, log and skip
            tqdm.write(f"ERROR: Exceeded max retries for {timeframe}. Skipping.")
            chunk_trends = None
        
        if chunk_trends is not None and not chunk_trends.empty: # If data was fetched successfully
            if "isPartial" in chunk_trends.columns: # Remove 'isPartial' column if it exists
                chunk_trends = chunk_trends.drop(columns=["isPartial"]) # Drop 'isPartial' column
            all_trends.append(chunk_trends) # Append to the list of all trends
        
        current_start = current_end + pd.Timedelta(days=1) # Move to the next chunk
        pbar.update(1) # Update progress bar
        time.sleep(sleep + random.uniform(0, 5)) # Sleep to avoid rate limits
    
    pbar.close() # Close progress bar
    
    if not all_trends: # If no data was collected, return empty DataFrame
        return pd.DataFrame()
    
    result = pd.concat(all_trends).sort_index() # Concatenate all chunks and sort by index
    result = result[~result.index.duplicated(keep="first")] # Remove duplicate indices, keeping the first occurrence
    
    print(f"Collected {len(result)} days of Google Trends data")
    return result

# -------------------------------------------------------------

# Might not need this beacuse I have the NYT API collection, but leaving it for now
def get_daily_news_sentiment(start_date, end_date):
    print("--- Fetching SF Fed Daily News Sentiment Index ---")
    
    base_page_url = "https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/"
    
    try:
        print(f"Scraping main page: {base_page_url}")
        page_response = requests.get(base_page_url)
        page_response.raise_for_status()
        
        soup = BeautifulSoup(page_response.content, 'lxml')
        download_link_tag = soup.find('a', string=lambda text: 'Daily News Sentiment data' in str(text))
        
        if not download_link_tag or not download_link_tag.has_attr('href'):
            raise ValueError("Could not find the download link on the page.")
        
        relative_url = download_link_tag['href']
        file_url = urljoin(base_page_url, relative_url)
        
        print(f"Found download link: {file_url}")

        print("Downloading file content...")
        file_response = requests.get(file_url)
        file_response.raise_for_status()
        
        with io.BytesIO(file_response.content) as file_in_memory:
            df = pd.read_excel(file_in_memory, index_col=0, parse_dates=True, sheet_name='Data')
        
        df.index = pd.to_datetime(df.index, errors='coerce')
        df_filtered = df.loc[start_date:end_date]
        
        print(f"Returned {len(df_filtered)} records.")
        return df_filtered
    
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------

def collect_nyt_top_daily_articles(start_date, end_date, articles_per_day=15):    

    load_dotenv()
    
    NYT_API_KEY = os.getenv("NEWS_API_KEY")
    if NYT_API_KEY is None:
        raise RuntimeError("NEWS_API_KEY not set in .env file")
    
    print(f"--- Collecting Top {articles_per_day} Articles Per Day: {start_date} to {end_date} ---")
    
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    current = pd.to_datetime(start_date).replace(day=1)
    
    # Define reference corpus for financial/market relevance
    financial_corpus = [
        "stock market trading equity shares investment portfolio",
        "recession economic downturn crisis financial collapse",
        "inflation prices consumer index monetary policy interest rates",
        "unemployment jobs employment labor market workforce",
        "federal reserve central bank monetary policy quantitative easing",
        "earnings revenue profit corporate financial results quarterly",
        "volatility risk uncertainty market fluctuation",
        "debt credit bonds treasury securities",
        "GDP growth economic expansion productivity output",
        "bull market bear market correction rally",
        "S&P Dow Jones NASDAQ index futures options",
        "commodities oil gold currency forex exchange"
    ]
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )
    
    # Fit on financial corpus
    corpus_vectors = vectorizer.fit_transform(financial_corpus)
    
    all_articles = []
    
    # Calculate total months for progress bar
    total_months = ((pd.to_datetime(end_date).year - current.year) * 12 + 
                    pd.to_datetime(end_date).month - current.month + 1)
    
    # Add progress bar
    pbar = tqdm(total=total_months, desc="NYT Articles", unit="month")
    
    while current <= pd.to_datetime(end_date):
        year, month = current.year, current.month
        url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
        
        pbar.set_postfix_str(f"{year}-{month:02d}")
        
        for attempt in range(3):
            try:
                resp = requests.get(url, params={'api-key': NYT_API_KEY}, timeout=120)
                resp.raise_for_status()
                
                daily_articles = {}

                for article in resp.json()['response']['docs']:
                    pub_date = pd.to_datetime(article['pub_date']).date()
                    if not (start <= pub_date <= end):
                        continue
                    
                    # Combine text fields for TF-IDF analysis
                    text = ' '.join([
                        article.get('headline', {}).get('main', ''),
                        article.get('abstract', ''),
                        article.get('lead_paragraph', ''),
                        article.get('snippet', '')
                    ])
                    
                    if len(text.strip()) < 50:  # Skip very short articles
                        continue
                    
                    # Calculate TF-IDF similarity score
                    try:
                        article_vector = vectorizer.transform([text])
                        similarity_scores = cosine_similarity(article_vector, corpus_vectors)
                        tfidf_score = float(np.max(similarity_scores))
                    except Exception:
                        tfidf_score = 0.0
                    
                    # Boost score for business/financial sections
                    section_boost = 1.0
                    if article.get('section_name', '') in ['Business Day', 'Markets', 'Economy', 'Business', 'Financial']:
                        section_boost = 1.3
                    elif article.get('news_desk', '') in ['Business', 'Financial', 'Business Day']:
                        section_boost = 1.2
                    
                    final_score = tfidf_score * section_boost
                    
                    if final_score < 0.1:  # Minimum relevance threshold
                        continue
                    
                    if pub_date not in daily_articles:
                        daily_articles[pub_date] = []
                    
                    daily_articles[pub_date].append({
                        'date': pub_date,
                        'headline': article['headline']['main'],
                        'abstract': article.get('abstract', ''),
                        'lead_paragraph': article.get('lead_paragraph', ''),
                        'section': article.get('section_name', ''),
                        'news_desk': article.get('news_desk', ''),
                        'word_count': article.get('word_count', 0),
                        'web_url': article.get('web_url', ''),
                        '_score': final_score
                    })
                
                for _, arts in daily_articles.items():
                    all_articles.extend(sorted(arts, key=lambda x: x['_score'], reverse=True)[:articles_per_day])
                
                break
                
            except Exception as e:
                if attempt < 2:
                    time.sleep(30 * (attempt + 1))
                else:
                    tqdm.write(f"{year}-{month:02d}: Skipped after 3 attempts")
        
        current += pd.DateOffset(months=1)
        pbar.update(1)
        time.sleep(15)
    
    pbar.close()
    
    df = pd.DataFrame(all_articles)
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True).drop(columns=['_score'])
    
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
    print(f"✓ Data directory ready: {data_dir}")
    print("=" * 60)
    
    # Collect all data
    ohlvc_data = collect_OHLVC("SPY")
    print("=" * 60)
    
    macro_data = collect_Macro()
    print("=" * 60)
    
    trends_keywords = ["Debt", "Recession", "Stocks to buy", "Unemployment", "Market crash"]
    google_trends_data = collect_google_trends(trends_keywords, START, END)
    print("=" * 60)
    
    # sentiment_data = get_daily_news_sentiment(START, END)
    
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
        print(f"✓ Saved: {macro_path}")
    except Exception as e:
        print(f"Failed to save US_Macro_Data.csv: {e}")
    
    try:
        trends_path = os.path.join(data_dir, "US_Google_Trends_Data.csv")
        google_trends_data.to_csv(trends_path)
        print(f"✓ Saved: {trends_path}")
    except Exception as e:
        print(f"Failed to save US_Google_Trends_Data.csv: {e}")
    
    # sentiment_data.to_csv(os.path.join(data_dir, "US_Sentiment_Data.csv"))
    # print("✓ Saved: US_Sentiment_Data.csv")
    
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