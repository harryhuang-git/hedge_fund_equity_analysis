"""
Pipeline script for running the HFT strategy evaluation process.
"""

import argparse
import logging
import os
from datetime import datetime
import pandas as pd
from typing import List, Dict
import json

from data.market_data import MarketDataCollector
from models.factors import FactorCalculator
from models.rank_signals import SignalGenerator
from models.backtest_engine import BacktestEngine
from models.report import PerformanceReport

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{log_dir}/pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run HFT strategy evaluation pipeline')
    
    parser.add_argument('--tickers', type=str, required=True,
                      help='Comma-separated list of stock tickers')
    parser.add_argument('--start', type=str, required=True,
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                      help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, default='default',
                      help='Strategy configuration name')
    parser.add_argument('--initial-capital', type=float, default=1000000,
                      help='Initial capital for backtesting')
    
    return parser.parse_args()

def load_strategy_config(strategy_name: str) -> Dict:
    """
    Load strategy configuration from file.
    
    Args:
        strategy_name (str): Name of the strategy configuration
        
    Returns:
        Dict: Strategy configuration
    """
    config_file = f"config/strategies/{strategy_name}.json"
    
    if not os.path.exists(config_file):
        logger.warning(f"Strategy config {config_file} not found, using default")
        return {
            "momentum": ["RSI", "ROC"],
            "trend": ["SMA", "ADX"],
            "volatility": ["ATR"],
            "weights": {
                "RSI": 0.3,
                "ROC": 0.2,
                "SMA": 0.2,
                "ADX": 0.3
            }
        }
    
    with open(config_file, 'r') as f:
        return json.load(f)

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for the specified stocks.
    
    Args:
        tickers (List[str]): List of stock tickers
        start_date (str): Start date
        end_date (str): End date
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of stock data
    """
    collector = MarketDataCollector()
    stock_data = {}
    
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}")
        data = collector.get_daily_data(ticker, start_date, end_date)
        if data is not None:
            stock_data[ticker] = data
        else:
            logger.warning(f"Failed to fetch data for {ticker}")
    
    return stock_data

def run_pipeline(args):
    """Run the complete strategy evaluation pipeline."""
    # Load strategy configuration
    strategy_config = load_strategy_config(args.strategy)
    logger.info(f"Loaded strategy configuration: {args.strategy}")
    
    # Fetch stock data
    tickers = args.tickers.split(',')
    stock_data = fetch_stock_data(tickers, args.start, args.end)
    
    if not stock_data:
        logger.error("No stock data available for analysis")
        return
    
    # Run backtest
    logger.info("Running backtest...")
    backtest = BacktestEngine(stock_data, initial_capital=args.initial_capital)
    backtest.run_backtest(args.start, args.end)
    
    # Generate performance report
    logger.info("Generating performance report...")
    report = PerformanceReport(
        pd.DataFrame(backtest.equity_curve),
        backtest.trades
    )
    metrics = report.generate_report(args.strategy)
    
    # Save summary to CSV
    summary_file = f"outputs/metrics/summary_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([metrics]).to_csv(summary_file, index=False)
    logger.info(f"Saved summary to {summary_file}")

def main():
    """Main entry point."""
    logger = setup_logging()
    args = parse_args()
    
    try:
        logger.info("Starting strategy evaluation pipeline")
        run_pipeline(args)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 