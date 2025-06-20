"""
Stock ranking and signal generation module for the HFT project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from config.config import PORTFOLIO
from .factors import FactorCalculator

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self, stock_data: Dict[str, pd.DataFrame]):
        """
        Initialize signal generator with stock data.
        
        Args:
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock codes to their price data
        """
        self.stock_data = stock_data
        self.factor_scores = {}
        self.rankings = {}
    
    def calculate_factors(self) -> pd.DataFrame:
        """
        Calculate factors for all stocks.
        
        Returns:
            pd.DataFrame: Combined factor scores for all stocks
        """
        all_scores = []
        
        for stock_code, data in self.stock_data.items():
            try:
                factor_calc = FactorCalculator(data)
                scores = factor_calc.calculate_combined_score()
                scores['stock_code'] = stock_code
                all_scores.append(scores)
            except Exception as e:
                logger.error(f"Error calculating factors for {stock_code}: {str(e)}")
                continue
        
        if not all_scores:
            raise ValueError("No factor scores were calculated successfully")
        
        return pd.concat(all_scores, axis=0)
    
    def rank_stocks(self, date: str) -> Tuple[List[str], List[str]]:
        """
        Rank stocks based on their factor scores for a given date.
        
        Args:
            date (str): Date to rank stocks for
            
        Returns:
            Tuple[List[str], List[str]]: Lists of top and bottom stock codes
        """
        try:
            if not self.factor_scores:
                self.factor_scores = self.calculate_factors()
            
            # Get scores for the specific date
            date_scores = self.factor_scores[self.factor_scores.index == date]
            
            if date_scores.empty:
                logger.warning(f"No factor scores available for date {date}")
                return [], []
            
            # Handle NaN values
            date_scores = date_scores.fillna(0)
            
            # Sort stocks by combined score
            ranked_stocks = date_scores.sort_values('combined_score', ascending=False)
            
            # Select top and bottom stocks
            top_n = min(PORTFOLIO['top_n'], len(ranked_stocks))
            bottom_n = min(PORTFOLIO['bottom_n'], len(ranked_stocks))
            
            top_stocks = ranked_stocks.head(top_n)['stock_code'].tolist()
            bottom_stocks = ranked_stocks.tail(bottom_n)['stock_code'].tolist()
            
            return top_stocks, bottom_stocks
            
        except Exception as e:
            logger.error(f"Error ranking stocks: {str(e)}")
            return [], []
    
    def generate_signal(self, factors: pd.DataFrame, config: Optional[Dict] = None) -> pd.Series:
        """
        Generate trading signals based on factor scores.
        
        Args:
            factors (pd.DataFrame): Factor scores for all stocks
            config (Dict, optional): Strategy configuration
            
        Returns:
            pd.Series: Trading signals for each stock (-1 to 1)
        """
        if factors.empty:
            logger.warning("Empty factor data received")
            return pd.Series()
        logger.debug(f"generate_signal: factors shape={factors.shape}, head=\n{factors.head()}")
        try:
            # Validate required columns
            required_columns = ['ticker', 'combined_score']
            missing_columns = [col for col in required_columns if col not in factors.columns]
            if missing_columns:
                logger.error(f"Missing required columns in factor data: {missing_columns}")
                return pd.Series()
            
            # Get the latest factor scores for each ticker
            latest_factors = factors.groupby('ticker').last()
            
            # Handle NaN values
            if latest_factors.isna().any().any():
                logger.warning("NaN values found in factor scores, using 0 as default")
                latest_factors = latest_factors.fillna(0)
            
            # Validate combined scores
            if latest_factors['combined_score'].isna().any():
                logger.warning("NaN values in combined scores after filling, using 0")
                latest_factors['combined_score'] = latest_factors['combined_score'].fillna(0)
            
            # Calculate signals based on combined score
            signals = pd.Series(index=latest_factors.index)
            
            # Apply signal thresholds if provided in config
            if config and 'signal_threshold' in config:
                threshold = config['signal_threshold']
            else:
                threshold = 0.0  # Lowered from 0.5 to 0.0 for more trades
            signals[latest_factors['combined_score'] > threshold] = 1.0
            signals[latest_factors['combined_score'] < -threshold] = -1.0
            signals[(latest_factors['combined_score'] >= -threshold) & \
                   (latest_factors['combined_score'] <= threshold)] = 0.0
            
            # Log signal distribution and head
            logger.info(f"Signal distribution: {signals.value_counts().to_dict()}")
            logger.debug(f"Signals head: \n{signals.head()}")
            
            return signals
                
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return pd.Series()
    
    def generate_signals(self, date: str) -> Dict[str, float]:
        """
        Generate trading signals for a given date.
        
        Args:
            date (str): Date to generate signals for
            
        Returns:
            Dict[str, float]: Dictionary of stock codes to their signal values (1 for long, -1 for short)
        """
        top_stocks, bottom_stocks = self.rank_stocks(date)
        
        signals = {}
        # Long signals for top stocks
        for stock in top_stocks:
            signals[stock] = 1.0
        
        # Short signals for bottom stocks
        for stock in bottom_stocks:
            signals[stock] = -1.0
        
        return signals
    
    def get_portfolio_weights(self, date: str) -> Dict[str, float]:
        """
        Calculate portfolio weights based on signals.
        
        Args:
            date (str): Date to calculate weights for
            
        Returns:
            Dict[str, float]: Dictionary of stock codes to their portfolio weights
        """
        signals = self.generate_signals(date)
        
        if not signals:
            return {}
        
        # Calculate equal weights for long and short positions
        long_weight = 1.0 / len([s for s in signals.values() if s > 0])
        short_weight = -1.0 / len([s for s in signals.values() if s < 0])
        
        weights = {}
        for stock, signal in signals.items():
            weights[stock] = long_weight if signal > 0 else short_weight
        
        return weights

if __name__ == "__main__":
    # Example usage
    from src.data.market_data import MarketDataCollector
    
    # Get sample data for multiple stocks
    collector = MarketDataCollector()
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime('%Y%m%d')
    
    # Example stock list (you would typically get this from Tushare)
    stock_list = ['000001.SZ', '600000.SH', '601318.SH']
    
    # Collect data for all stocks
    stock_data = {}
    for stock in stock_list:
        data = collector.get_daily_data(stock, start_date, end_date)
        if data is not None:
            stock_data[stock] = data
    
    if stock_data:
        # Generate signals
        signal_gen = SignalGenerator(stock_data)
        latest_date = end_date
        signals = signal_gen.generate_signals(latest_date)
        weights = signal_gen.get_portfolio_weights(latest_date)
        
        print("Trading Signals:")
        print(signals)
        print("\nPortfolio Weights:")
        print(weights) 