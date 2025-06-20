"""
Class to load and manage shadow data for testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ShadowDataLoader:
    def __init__(self, data_dir: str = 'data/tushare'):
        """
        Initialize shadow data loader.
        
        Args:
            data_dir (str): Directory containing shadow data
        """
        self.data_dir = data_dir
        self.data = {}
        self.metadata = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from JSON file."""
        metadata_file = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all stock data from CSV files.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Load each CSV file
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                stock_code = filename[:-4]  # Remove .csv extension
                file_path = os.path.join(self.data_dir, filename)
                
                try:
                    # Read CSV and set date as index
                    df = pd.read_csv(file_path)
                    
                    # Convert trade_date to datetime and set as index
                    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    df.set_index('trade_date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Select and rename required columns
                    column_mapping = {
                        'adj_open': 'open',
                        'adj_high': 'high',
                        'adj_low': 'low',
                        'adj_close': 'close',
                        'vol': 'volume',
                        'amount': 'amount'
                    }
                    
                    if all(col in df.columns for col in column_mapping.keys()):
                        df = df[list(column_mapping.keys())].copy()
                        df = df.rename(columns=column_mapping)
                        self.data[stock_code] = df
                    else:
                        logger.warning(f"Missing required columns in {filename}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
                    continue
        
        logger.info(f"Loaded data for {len(self.data)} stocks")
        return self.data
    
    def get_data_range(self) -> tuple:
        """
        Get the date range of the loaded data.
        
        Returns:
            tuple: (start_date, end_date) as datetime objects
        """
        if not self.data:
            raise ValueError("No data loaded")
        
        all_dates = []
        for df in self.data.values():
            all_dates.extend(df.index)
        
        return min(all_dates), max(all_dates)
    
    def get_stock_codes(self) -> List[str]:
        """
        Get list of stock codes in the loaded data.
        
        Returns:
            List[str]: List of stock codes
        """
        return list(self.data.keys())
    
    def get_data_for_date(self, date: str) -> Dict[str, pd.DataFrame]:
        """
        Get data for all stocks on a specific date.
        
        Args:
            date (str): Date in YYYYMMDD format
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data for the date
        """
        if not self.data:
            raise ValueError("No data loaded")
        
        date_dt = pd.to_datetime(date)
        current_data = {}
        
        for ticker, df in self.data.items():
            if date_dt in df.index:
                current_data[ticker] = df.loc[:date_dt]
        
        return current_data
    
    def get_metadata(self) -> dict:
        """
        Get metadata for the loaded data.
        
        Returns:
            dict: Metadata dictionary
        """
        return self.metadata

def main():
    """Example usage of ShadowDataLoader."""
    # Initialize loader
    loader = ShadowDataLoader()
    
    # Load data
    stock_data = loader.load_data()
    
    # Get data range
    start_date, end_date = loader.get_data_range()
    print(f"Data range: {start_date} to {end_date}")
    
    # Get stock codes
    stock_codes = loader.get_stock_codes()
    print(f"Number of stocks: {len(stock_codes)}")
    
    # Get metadata
    metadata = loader.get_metadata()
    print("\nMetadata:")
    print(json.dumps(metadata, indent=4))

if __name__ == "__main__":
    main() 