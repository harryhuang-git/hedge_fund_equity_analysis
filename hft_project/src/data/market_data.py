"""
Market data collection module for the HFT project.
"""

import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from config.config import TUSHARE_TOKEN, HISTORICAL_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataCollector:
    def __init__(self):
        """Initialize the market data collector."""
        self.pro = ts.pro_api(TUSHARE_TOKEN)
        self._ensure_data_directories()
    
    def _ensure_data_directories(self):
        """Ensure that data directories exist."""
        os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
    
    def get_daily_data(self, stock_code, start_date, end_date):
        """
        Get daily market data for a specific stock.
        
        Args:
            stock_code (str): Stock code (e.g., '000001.SZ')
            start_date (str): Start date in format 'YYYYMMDD'
            end_date (str): End date in format 'YYYYMMDD'
            
        Returns:
            pandas.DataFrame: Daily market data
        """
        try:
            df = self.pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
            return df.sort_values('trade_date')
        except Exception as e:
            logger.error(f"Error fetching daily data for {stock_code}: {str(e)}")
            return None
    
    def get_realtime_quotes(self, stock_codes):
        """
        Get real-time quotes for specified stocks.
        
        Args:
            stock_codes (list): List of stock codes
            
        Returns:
            pandas.DataFrame: Real-time quotes
        """
        try:
            df = ts.get_realtime_quotes(stock_codes)
            return df
        except Exception as e:
            logger.error(f"Error fetching real-time quotes: {str(e)}")
            return None
    
    def save_historical_data(self, stock_code, start_date, end_date):
        """
        Save historical data to CSV file.
        
        Args:
            stock_code (str): Stock code
            start_date (str): Start date
            end_date (str): End date
        """
        df = self.get_daily_data(stock_code, start_date, end_date)
        if df is not None:
            filename = f"{HISTORICAL_DATA_DIR}/{stock_code}_{start_date}_{end_date}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved historical data to {filename}")
        else:
            logger.error("Failed to save historical data")

if __name__ == "__main__":
    # Example usage
    collector = MarketDataCollector()
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    
    # Example: Get data for Ping An Bank (000001.SZ)
    data = collector.get_daily_data('000001.SZ', start_date, end_date)
    if data is not None:
        print(data.head()) 