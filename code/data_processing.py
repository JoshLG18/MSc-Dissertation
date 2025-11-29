# ---- Script to clean and combine all data from the various sources ----
# -----------------------------------------------------------------------

import pandas as pd

# -----------------------------------------------------------------------

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

# -----------------------------------------------------------------------

def clean_stock_data(df):
    # Clean the stock data

    # remove the top 3 rows
    df = df.iloc[3:]

    # add column headers
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    df = df[df['Date'] <= '2025-10-31']

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
    
    df = df[df['Date'] <= '2025-10-31']

    return df

# -----------------------------------------------------------------------

def clean_google_trends_data(df):
    # Clean the Google Trends data

    # remove date after 2025-10-31
    df = df[df['date'] <= '2025-10-31']

    # rename date column
    df = df.rename(columns={'date': 'Date'})

    # remove weekends
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.dayofweek < 5]

    return df

# -----------------------------------------------------------------------

# ---- Main Execution ----

if __name__ == '__main__':

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

    