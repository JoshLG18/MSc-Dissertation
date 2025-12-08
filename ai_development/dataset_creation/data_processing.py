# ---- Script to clean and combine all data from the various sources ----
# -----------------------------------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from data_collection import START, END
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch  # only needed when running on google colab with GPU

# -----------------------------------------------------------------------

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

# -----------------------------------------------------------------------

def clean_stock_data(df):
    # Clean the stock data

    # remove the top 3 rows
    df = df.iloc[2:]

    # add column headers
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    df = df[(df['Date'] <= END) & (df['Date'] >= START)]

    # convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # set all to float except date
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(int)

    return df

# -----------------------------------------------------------------------

def clean_macro_data(df):
    # Clean the macroeconomic data

    # add clean colum headings 
    df.columns = ['Date', 
                  'FED_FUNDS', 
                  'MatRate_10Y', 
                  'MatRate_2Y', 
                  'Mat_Rate_CURVE', 
                  'CPI', 
                  'CORE_CPI', 
                  'UNEMPLOYMENT', 
                  'JOBLESS_CLAIMS',
                  'GDP', 
                  'CONSUMER_SENTIMENT', 
                  'VIX']
    
    df = df[(df['Date'] <= END) & (df['Date'] >= START)]

    # only keep rows that are in the ohclv data date column - load in the stock data dates
    stock_data = load_data('../data/raw/US_OHLVC_Data.csv')
    stock_data = stock_data.iloc[2:]
    stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].isin(stock_data['Date'])]

    # back fill the jobless claims column
    df['JOBLESS_CLAIMS'] = df['JOBLESS_CLAIMS'].fillna(method='bfill')

    return df

# -----------------------------------------------------------------------

def clean_google_trends_data(df):
    # Clean the Google Trends data

    # remove date beyond end date
    df = df[(df['date'] <= END) & (df['date'] >= START)]

    # rename date column
    df = df.rename(columns={'date': 'Date'})

    # set all columns to have trend before it
    for col in df.columns:
        if col != 'Date':
            df = df.rename(columns={col: f"{col}_trend"})

    # only keep rows that are in the ohclv data date column - load in the stock data dates
    stock_data = load_data('../data/raw/US_OHLVC_Data.csv')
    stock_data = stock_data.iloc[2:]
    stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[df['Date'].isin(stock_data['Date'])]

    return df

# -----------------------------------------------------------------------

def clean_news_data(df):
    # Clean the news data and run finbert to classify sentiment 

    # load finbert model

    # clean the news data
    df = df[(df['date'] <= END) & (df['date'] >= START)]
    df['date'] = pd.to_datetime(df['date'])

    # only keep relevant columns
    df = df[['date', 'headline', 'abstract']]
    df['text'] = df['headline'] + ' ' + df['abstract']
    df = df[['date', 'text']]
    
    # handle missing text by filling with empty string
    df['text'] = df['text'].fillna("")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Process in batches for speed (instead of one at a time)
    batch_size = 32  # Adjust based on GPU memory
    sentiments = []
    
    for i in range(0, len(df), batch_size):
        batch_texts = df['text'].iloc[i:i+batch_size].tolist()
        
        # Tokenize entire batch at once
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        # Convert batch results to sentiment scores
        for score in scores:
            sentiment_idx = score.argmax()
            sentiment_map = {0: 1, 1: -1, 2: 0}
            sentiment_score = sentiment_map[sentiment_idx]
            sentiments.append(sentiment_score)
        
        # Progress tracking
        if (i + batch_size) % 1000 == 0:
            print(f"Processed {i + batch_size}/{len(df)} articles...")
    
    # Add sentiment score column
    df['sentiment'] = sentiments

    # # combine sentiment scores by date (average sentiment per day)
    # df = df.groupby('date')['sentiment'].mean().reset_index()
    # df = df.rename(columns={'date': 'Date'})

    # # only keep rows that are in the ohclv data date column - load in the stock data dates
    # stock_data = load_data('../data/raw/US_OHLVC_Data.csv')
    # stock_data = stock_data.iloc[2:]
    # stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    # df['Date'] = pd.to_datetime(df['Date'])
    # df = df[df['Date'].isin(stock_data['Date'])]
    
    # #return the cleaned and sentiment dataframe
    # return df

# -----------------------------------------------------------------------


def clean_all_data():
    # General cleaning function (if needed)
    # Clean the stock data
    stock_data = load_data('data/raw/US_OHLVC_Data.csv')
    cleaned_stock_data = clean_stock_data(stock_data)
    cleaned_stock_data.to_csv('data/processed/cleaned_stock_data.csv', index=False)

    # Clean the macroeconomic data
    macro_data = load_data('data/raw/US_Macro_Data.csv')
    cleaned_macro_data = clean_macro_data(macro_data)
    cleaned_macro_data.to_csv('data/processed/cleaned_macro_data.csv', index=False)

    # Clean the Google Trends data
    google_trends_data = load_data('data/raw/US_Google_Trends_Data.csv')
    cleaned_google_trends_data = clean_google_trends_data(google_trends_data)
    cleaned_google_trends_data.to_csv('data/processed/cleaned_google_trends_data.csv', index=False)

# -----------------------------------------------------------------------

# ---- Main Execution ----

if __name__ == '__main__':
    # clean_all_data()

    # clean just the news for testing
    news_data = load_data('../data/raw/NYT_Top_Daily_Articles.csv')
    cleaned_news_data = clean_news_data(news_data)
    print(cleaned_news_data.head())
    print(cleaned_news_data.isna().sum())

# -----------------------------------------------------------------------