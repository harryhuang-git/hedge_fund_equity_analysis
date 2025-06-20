"""
Module for automatically testing strategies on index constituents.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data.market_data import MarketDataCollector
from ..models.backtest_engine import BacktestEngine
from ..models.report import PerformanceReport

logger = logging.getLogger(__name__)

class IndexTester:
    def __init__(
        self,
        index_name: str,
        start_date: str,
        end_date: str,
        strategy_config: Dict,
        initial_capital: float = 1000000,
        max_workers: int = 4
    ):
        """
        Initialize index tester.
        
        Args:
            index_name (str): Name of the index (CSI300, CSI800, GEM)
            start_date (str): Start date for backtest
            end_date (str): End date for backtest
            strategy_config (Dict): Strategy configuration
            initial_capital (float): Initial capital for each stock
            max_workers (int): Maximum number of parallel workers
        """
        self.index_name = index_name
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_config = strategy_config
        self.initial_capital = initial_capital
        self.max_workers = max_workers
        
        self.collector = MarketDataCollector()
        self.results = []
        
    def get_index_constituents(self) -> List[str]:
        """Get list of index constituents."""
        # This is a placeholder - you'll need to implement the actual index constituent fetching
        # based on your data source (e.g., Tushare API)
        if self.index_name == "CSI300":
            return self.collector.get_csi300_constituents()
        elif self.index_name == "CSI800":
            return self.collector.get_csi800_constituents()
        elif self.index_name == "GEM":
            return self.collector.get_gem_constituents()
        else:
            raise ValueError(f"Unknown index: {self.index_name}")
    
    def run_single_backtest(self, ticker: str) -> Optional[Dict]:
        """
        Run backtest for a single stock.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            Optional[Dict]: Backtest results
        """
        try:
            # Fetch stock data
            data = self.collector.get_daily_data(
                ticker,
                self.start_date,
                self.end_date
            )
            
            if data is None or len(data) < 60:  # Minimum data requirement
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            # Run backtest
            backtest = BacktestEngine(
                {ticker: data},
                initial_capital=self.initial_capital
            )
            
            results = backtest.run_backtest(
                self.start_date,
                self.end_date,
                self.strategy_config
            )
            
            # Add ticker to results
            results['ticker'] = ticker
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest for {ticker}: {str(e)}")
            return None
    
    def run_index_backtest(self):
        """Run backtest for all index constituents."""
        # Get index constituents
        constituents = self.get_index_constituents()
        logger.info(f"Running backtest for {len(constituents)} stocks in {self.index_name}")
        
        # Run backtests in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.run_single_backtest, ticker): ticker
                for ticker in constituents
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result is not None:
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics for all backtests."""
        if not self.results:
            return {}
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        summary = {
            'index': self.index_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_stocks': len(self.results),
            'successful_stocks': len(results_df),
            'metrics': {
                'mean_return': results_df['return'].mean(),
                'median_return': results_df['return'].median(),
                'std_return': results_df['return'].std(),
                'sharpe_ratio': results_df['return'].mean() / results_df['return'].std() * np.sqrt(252),
                'win_rate': (results_df['return'] > 0).mean(),
                'max_drawdown': results_df['max_drawdown'].mean(),
                'avg_trades': results_df['trades'].apply(len).mean()
            },
            'top_performers': results_df.nlargest(5, 'return')[['ticker', 'return']].to_dict('records'),
            'bottom_performers': results_df.nsmallest(5, 'return')[['ticker', 'return']].to_dict('records')
        }
        
        # Save summary to file
        self.save_summary(summary)
        
        return summary
    
    def save_summary(self, summary: Dict):
        """Save summary results to file."""
        # Create output directory
        os.makedirs('outputs/index_tests', exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"outputs/index_tests/{self.index_name}_{timestamp}.json"
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Saved summary to {filename}")

def main():
    """Example usage of IndexTester."""
    # Load strategy configuration
    with open('config/strategies/momentum.json', 'r') as f:
        strategy_config = json.load(f)
    
    # Initialize index tester
    tester = IndexTester(
        index_name="CSI300",
        start_date="2023-01-01",
        end_date="2023-12-31",
        strategy_config=strategy_config
    )
    
    # Run backtest
    summary = tester.run_index_backtest()
    
    # Print summary
    print("\nIndex Backtest Summary:")
    print(f"Index: {summary['index']}")
    print(f"Period: {summary['start_date']} to {summary['end_date']}")
    print(f"Total Stocks: {summary['total_stocks']}")
    print(f"Successful Tests: {summary['successful_stocks']}")
    print("\nPerformance Metrics:")
    for metric, value in summary['metrics'].items():
        print(f"{metric}: {value:.2%}")
    
    print("\nTop Performers:")
    for stock in summary['top_performers']:
        print(f"{stock['ticker']}: {stock['return']:.2%}")
    
    print("\nBottom Performers:")
    for stock in summary['bottom_performers']:
        print(f"{stock['ticker']}: {stock['return']:.2%}")

if __name__ == "__main__":
    main() 