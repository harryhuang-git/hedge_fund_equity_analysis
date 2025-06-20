"""
Simple script to download historical data from Tushare API.
Run this script manually when VPN is off.
"""

import tushare as ts
import pandas as pd
import os
from datetime import datetime
import json
from tqdm import tqdm
import time
import random

def download_stock_data(
    ts_api: ts.pro_api,
    stock_code: str,
    start_date: str,
    end_date: str,
    output_dir: str
) -> bool:
    """
    Download daily data and adjusted factors for a single stock.
    
    Args:
        ts_api: Tushare API instance
        stock_code: Stock code
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        output_dir: Output directory
    
    Returns:
        bool: True if successful, False otherwise
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Download daily data
            df = ts_api.daily(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                print(f"No data found for {stock_code}")
                return False
            
            # Add a small delay to avoid rate limits
            time.sleep(0.1)
            
            # Download adjusted factors
            adj_factors = ts_api.adj_factor(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if adj_factors is not None and not adj_factors.empty:
                # Merge with daily data
                df = pd.merge(
                    df,
                    adj_factors[['trade_date', 'adj_factor']],
                    on='trade_date',
                    how='left'
                )
                
                # Fill missing adj_factor with 1.0
                df['adj_factor'] = df['adj_factor'].fillna(1.0)
                
                # Calculate adjusted prices
                df['adj_open'] = df['open'] * df['adj_factor']
                df['adj_high'] = df['high'] * df['adj_factor']
                df['adj_low'] = df['low'] * df['adj_factor']
                df['adj_close'] = df['close'] * df['adj_factor']
            
            # Save to file
            output_file = f"{output_dir}/{stock_code}.csv"
            df.to_csv(output_file, index=False)
            return True
            
        except Exception as e:
            if "每分钟最多访问该接口500次" in str(e):  # Rate limit error
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (attempt + 1) + random.uniform(0, 1)
                    print(f"\nRate limit hit, waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    continue
            print(f"Error downloading {stock_code}: {str(e)}")
            return False
    
    return False

def get_index_constituents(ts_api: ts.pro_api, index_code: str, start_date: str, end_date: str) -> list:
    """
    Get index constituents with rate limiting.
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            df = ts_api.index_weight(
                index_code=index_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                print("No constituents found")
                return []
            
            return df['con_code'].unique().tolist()
            
        except Exception as e:
            if "每分钟最多访问该接口500次" in str(e):  # Rate limit error
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (attempt + 1) + random.uniform(0, 1)
                    print(f"\nRate limit hit, waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    continue
            print(f"Error getting index constituents: {str(e)}")
            return []

def main():
    # Configuration
    TUSHARE_TOKEN = '2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211'  # Your Tushare token
    OUTPUT_DIR = 'data/tushare'
    START_DATE = '20230101'
    END_DATE = '20231231'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Tushare API
    ts_api = ts.pro_api(TUSHARE_TOKEN)
    
    # Get CSI300 constituents
    index_code = '000300.SH'
    print("Getting CSI300 constituents...")
    stock_codes = get_index_constituents(ts_api, index_code, START_DATE, END_DATE)
    
    if not stock_codes:
        print("Failed to get stock codes")
        return
    
    print(f"Found {len(stock_codes)} stocks in CSI300")
    
    # Download data for each stock
    successful_downloads = 0
    for stock_code in tqdm(stock_codes, desc="Downloading stock data"):
        if download_stock_data(ts_api, stock_code, START_DATE, END_DATE, OUTPUT_DIR):
            successful_downloads += 1
            # Add a small delay between stocks
            time.sleep(0.1)
    
    # Save metadata
    metadata = {
        'index': 'CSI300',
        'start_date': START_DATE,
        'end_date': END_DATE,
        'num_stocks': successful_downloads,
        'stock_codes': stock_codes,
        'downloaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nDownload completed:")
    print(f"Successfully downloaded data for {successful_downloads} stocks")
    print(f"Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 