# ---- Script to collect all data from the various sources ----
# -------------------------------------------------------------

# Loading in all necessary libraries
import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv
from fredapi import Fred

# Define Constants
START = '2000-01-01'
END = pd.Timestamp.today()
# -------------------------------------------------------------

# Collecting the OHLVC data from Yahoo Finance - SPY ticker

def collect_OHLVC(ticker):
    df = yf.download(ticker, start = START, end = END)
    return df

# -------------------------------------------------------------

# Collecting macroeconomic indicators from FRED

def collect_Macro():
    load_dotenv()
    
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if FRED_API_KEY is None:
        raise RuntimeError("FRED_API_KEY not set")
    
    fred = Fred(api_key=FRED_API_KEY)
    
    # Define series IDs
    ids = {
        # Interest Rates
        "FED_FUNDS": "FEDFUNDS", # Fed interest rate
        "MatRate_10Y": "DGS10", # 10Y bond maturation rate
        "MatRate_2Y": "DGS2", # 2Y bond maturation rate
        "Mat_Rate_CURVE": "T10Y2Y", # spread of 10Y + 2Y maturation
        
        # Inflation
        "CPI": "CPIAUCSL", # consumer price index - inflation rate
        "CORE_CPI": "CPILFESL", # inflation rate removing food + energy
        
        # Employment 
        "UNEMPLOYMENT": "UNRATE", # Unemployment rates
        "JOBLESS_CLAIMS": "ICSA", # jobless claims
        
        # Growth
        "GDP": "GDP", # Gross domestic product
        "CONSUMER_SENTIMENT": "UMCSENT", # consumer sentiment of the market
    }
    
    # Collect each series separately
    series_dict = {}
    for name, sid in ids.items():
        series_dict[name] = fred.get_series(
            sid, 
            observation_start=START, 
            observation_end=END
        )
    
    # Create DataFrame with daily frequency
    # Start with daily series as base
    df = pd.DataFrame(index=pd.date_range(START, END, freq='D'))
    
    # Add each series and forward-fill monthly data to daily
    for name, series in series_dict.items():
        df[name] = series
        df[name] = df[name].ffill()  # Forward fill monthly values
    
    # Filter to business days only (remove weekends)
    df = df[df.index.dayofweek < 5]
    
    return df
# -------------------------------------------------------------
def collect_NewsSentiment():
    # Placeholder for news sentiment collection logic
    pass

# -------------------------------------------------------------
def collect_SocialSentiment():
    # Placeholder for social sentiment collection logic
    pass


# -------------------------------------------------------------
# Save collected data to CSV files
def save_data():
    ohlvc_data = collect_OHLVC("SPY")
    macro_data = collect_Macro()

    ohlvc_data.to_csv("../data/US_OHLVC_Data.csv")
    macro_data.to_csv("../data/US_Macro_Data.csv")


if __name__ == '__main__':
    save_data()





