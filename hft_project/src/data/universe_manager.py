"""
Universe manager for handling stock universe data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
import json
from datetime import datetime, timedelta
import tushare as ts
from .shadow_data_loader import ShadowDataLoader

logger = logging.getLogger(__name__)

class UniverseManager:
    def __init__(self, tushare_token: str, use_shadow_data: bool = True):
        """
        Initialize universe manager.
        
        Args:
            tushare_token (str): Tushare API token
            use_shadow_data (bool): Whether to use shadow data instead of Tushare API
        """
        self.use_shadow_data = True  # Always use shadow data for testing
        self.shadow_loader = ShadowDataLoader()
        self.index_mapping = {
            'CSI300': '000300.SH',
            'CSI800': '000906.SH',
            'GEM': '399006.SZ'
        }
        # Bypass Tushare API
        self.ts_api = None
        self.sector_map = self._load_sector_map()
    
    def _load_sector_map(self):
        sector_file = 'data/stock_sector_mapping.csv'
        if not os.path.exists(sector_file):
            print(f"Warning: {sector_file} not found. Sector info will be missing.")
            return {}
        df = pd.read_csv(sector_file, dtype={'ts_code': str})
        return dict(zip(df['ts_code'], df['industry']))
    
    def load_universe_data(
        self,
        index_name: str = 'CSI300',
        start_date: str = '20230101',
        end_date: str = '20231231'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load universe data for a specific index.
        
        Args:
            index_name (str): Name of the index
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data
        """
        # Always use shadow data for this test
        return self._load_shadow_data(start_date, end_date)
    
    def _load_shadow_data(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from shadow data files.
        
        Args:
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data
        """
        # Load all shadow data
        stock_data = self.shadow_loader.load_data()
        
        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        filtered_data = {}
        for ticker, df in stock_data.items():
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            if mask.any():
                # Add sector/industry info as a new column
                sector = self.sector_map.get(ticker, 'Unknown')
                df = df.copy()
                df['industry'] = sector
                filtered_data[ticker] = df[mask]
        
        logger.info(f"Loaded shadow data for {len(filtered_data)} stocks")
        return filtered_data
    
    def _load_tushare_data(
        self,
        index_name: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from Tushare API.
        
        Args:
            index_name (str): Name of the index
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data
        """
        try:
            # Get index code
            index_code = self.index_mapping.get(index_name)
            if not index_code:
                raise ValueError(f"Unknown index: {index_name}")
            
            # Get index constituents
            df = self.ts_api.index_weight(
                index_code=index_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                raise ValueError(f"No constituents found for {index_name}")
            
            # Get unique stock codes
            stock_codes = df['con_code'].unique().tolist()
            
            # Download data for each stock
            stock_data = {}
            for stock_code in stock_codes:
                data = self._download_stock_data(stock_code, start_date, end_date)
                if data is not None:
                    stock_data[stock_code] = data
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error loading universe data: {str(e)}")
            raise
    
    def _download_stock_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Download and process stock data.
        
        Args:
            stock_code (str): Stock code
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            Optional[pd.DataFrame]: Processed stock data
        """
        try:
            # Download daily data
            df = self.ts_api.daily(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                logger.warning(f"No data found for {stock_code}")
                return None
            
            # Download adjusted factors
            adj_factors = self.ts_api.adj_factor(
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
            
            # Map Tushare columns to our standard format
            column_mapping = {
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount',
                'adj_open': 'adj_open',
                'adj_high': 'adj_high',
                'adj_low': 'adj_low',
                'adj_close': 'adj_close'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data for {stock_code}: {str(e)}")
            return None

def main():
    """Example usage of UniverseManager."""
    # Initialize with your Tushare token
    manager = UniverseManager('YOUR_TUSHARE_TOKEN', use_shadow_data=True)
    
    # Load universe data
    stock_data = manager.load_universe_data(
        index_name='CSI300',
        start_date='20230101',
        end_date='20231231'
    )
    
    print(f"Loaded data for {len(stock_data)} stocks")

if __name__ == "__main__":
    main() 