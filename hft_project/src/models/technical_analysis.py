"""
Technical analysis module for the HFT project.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize technical analysis with price data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        """
        self.data = data.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Validate that required columns exist in the data."""
        required_columns = ['open', 'high', 'low', 'close', 'vol']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def calculate_ma(self, periods: List[int]) -> Dict[str, pd.Series]:
        """
        Calculate Moving Averages for given periods.
        
        Args:
            periods (List[int]): List of periods for MA calculation
            
        Returns:
            Dict[str, pd.Series]: Dictionary of MA series
        """
        ma_dict = {}
        for period in periods:
            ma_dict[f'MA{period}'] = self.data['close'].rolling(window=period).mean()
        return ma_dict
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            fast (int): Fast period
            slow (int): Slow period
            signal (int): Signal period
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing MACD line, signal line, and histogram
        """
        exp1 = self.data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.data['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            period (int): Moving average period
            std_dev (float): Number of standard deviations
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing upper band, middle band, and lower band
        """
        middle_band = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'Upper': upper_band,
            'Middle': middle_band,
            'Lower': lower_band
        }
    
    def get_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators and return as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with all technical indicators
        """
        # Calculate all indicators
        ma_dict = self.calculate_ma([5, 10, 20, 60])
        rsi = self.calculate_rsi()
        macd_dict = self.calculate_macd()
        bb_dict = self.calculate_bollinger_bands()
        
        # Combine all indicators
        indicators = pd.DataFrame(index=self.data.index)
        indicators.update(ma_dict)
        indicators['RSI'] = rsi
        indicators.update(macd_dict)
        indicators.update(bb_dict)
        
        return indicators

if __name__ == "__main__":
    # Example usage
    from src.data.market_data import MarketDataCollector
    
    # Get some sample data
    collector = MarketDataCollector()
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime('%Y%m%d')
    
    data = collector.get_daily_data('000001.SZ', start_date, end_date)
    if data is not None:
        # Calculate technical indicators
        ta = TechnicalAnalysis(data)
        indicators = ta.get_all_indicators()
        print(indicators.head()) 