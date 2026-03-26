import os
import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_gold_data(ticker="GC=F", years=10, output_dir="data/raw"):
    """
    Fetches historical gold price data from Yahoo Finance.
    
    Args:
        ticker (str): The Yahoo Finance ticker symbol for Gold.
        years (int): Number of years of historical data to fetch.
        output_dir (str): Directory to save the raw CSV file.
    """
    print(f"Fetching {years} years of historical data for {ticker}...")
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years*365)
    
    # Download data
    df_gold = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    df_inr = yf.download("INR=X", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if df_gold.empty or df_inr.empty:
        print("Error: No data fetched. Please check your internet connection or the ticker symbol.")
        return
    
    # Clean column names (if multi-index)
    if isinstance(df_gold.columns, pd.MultiIndex):
        df_gold.columns = df_gold.columns.droplevel(1)
        df_inr.columns = df_inr.columns.droplevel(1)
        
    df_gold.reset_index(inplace=True)
    df_inr.reset_index(inplace=True)
    
    # Fix timezone issues before merging
    df_gold['Date'] = pd.to_datetime(df_gold['Date']).dt.tz_localize(None).dt.date
    df_inr['Date'] = pd.to_datetime(df_inr['Date']).dt.tz_localize(None).dt.date

    df = pd.merge(df_gold, df_inr[['Date', 'Close']], on='Date', how='inner', suffixes=('', '_INR'))
    
    # Convert Gold prices to INR per 10 grams
    # 1 Troy Ounce = 31.1034768 grams -> so 1 Ounce = 3.11034768 * 10g
    # Price per 10g in INR = (Price_USD * USD_INR) / 3.11034768
    conversion_factor = df['Close_INR'] / 3.11034768
    
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        if col in df.columns:
            df[col] = df[col] * conversion_factor
            
    df.drop(columns=['Close_INR'], inplace=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "gold_prices.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Successfully saved raw data to {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Gold Historical Data")
    parser.add_argument("--ticker", type=str, default="GC=F", help="Yahoo Finance Ticker")
    parser.add_argument("--years", type=int, default=10, help="Years of history")
    args = parser.parse_args()
    
    fetch_gold_data(ticker=args.ticker, years=args.years)
