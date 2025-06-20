"""
Script to create shadow training data from Tushare for offline testing.
"""

import tushare as ts
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

class ShadowDataCreator:
    def __init__(self, tushare_token: str, data_dir: str = 'data/shadow'):
        """
        Initialize shadow data creator.
        
        Args:
            tushare_token (str): Tushare API token
            data_dir (str): Directory to store shadow data
        """
        self.data_dir = data_dir
        self.ts_api = ts.pro_api(tushare_token)
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Index mapping
        self.index_mapping = {
            'CSI300': '000300.SH',
            'CSI800': '000906.SH',
            'GEM': '399006.SZ'
        }
    
    def download_stock_data(
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
    
    def create_shadow_data(
        self,
        index_name: str = 'CSI300',
        start_date: str = '20230101',
        end_date: str = '20231231',
        sample_size: int = 50
    ):
        """
        Create shadow training data for a specific index.
        
        Args:
            index_name (str): Name of the index
            start_date (str): Start date
            end_date (str): End date
            sample_size (int): Number of stocks to sample
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
            
            # Randomly sample stocks
            if len(stock_codes) > sample_size:
                np.random.seed(42)  # For reproducibility
                stock_codes = np.random.choice(stock_codes, sample_size, replace=False)
            
            logger.info(f"Downloading data for {len(stock_codes)} stocks")
            
            # Download data for each stock
            stock_data = {}
            for stock_code in tqdm(stock_codes, desc="Downloading stock data"):
                data = self.download_stock_data(stock_code, start_date, end_date)
                if data is not None:
                    stock_data[stock_code] = data
            
            if not stock_data:
                raise ValueError("No data downloaded for any stocks")
            
            # Save data
            output_file = f"{self.data_dir}/{index_name}_shadow_data.pkl"
            pd.to_pickle(stock_data, output_file)
            
            # Save metadata
            metadata = {
                'index_name': index_name,
                'start_date': start_date,
                'end_date': end_date,
                'num_stocks': len(stock_data),
                'stock_codes': list(stock_data.keys()),
                'data_points': sum(len(df) for df in stock_data.values()),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(f"{self.data_dir}/{index_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Shadow data created successfully: {output_file}")
            logger.info(f"Total stocks: {len(stock_data)}")
            logger.info(f"Total data points: {metadata['data_points']}")
            
        except Exception as e:
            logger.error(f"Error creating shadow data: {str(e)}")
            raise

def main():
    """Example usage of ShadowDataCreator."""
    # Initialize with your Tushare token
    creator = ShadowDataCreator('YOUR_TUSHARE_TOKEN')
    
    # Create shadow data for CSI300
    creator.create_shadow_data(
        index_name='CSI300',
        start_date='20230101',
        end_date='20231231',
        sample_size=50
    )

if __name__ == "__main__":
    main() 