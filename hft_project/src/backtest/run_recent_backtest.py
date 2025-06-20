"""
Script to run a one-month backtest using recent market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import tushare as ts
import os

from src.data.universe_manager import UniverseManager
from src.models.backtest_engine import BacktestEngine
from src.portfolio.optimizer import PortfolioOptimizer
from src.research.alpha_decay import AlphaDecayModel

logger = logging.getLogger(__name__)

def run_recent_backtest(tushare_token: str):
    """
    Run backtest for the most recent period.
    
    Args:
        tushare_token (str): Tushare API token
    """
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Initialize universe manager
        universe_manager = UniverseManager(tushare_token)
        
        # Get date range (last 5 trading days)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Load universe data
        universe_data = universe_manager.load_universe_data('CSI300', start_date, end_date)
        if not universe_data:
            logger.error("No universe data available for backtest")
            return
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(
            stock_data=universe_data,
            initial_capital=1000000,  # 1 million CNY
            transaction_cost=0.0003,  # 0.03% transaction cost
            slippage=0.0001  # 0.01% slippage
        )
        
        # Run backtest
        results = backtest_engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            strategy_config={
                'factors': ['momentum', 'volatility', 'volume'],
                'signal_threshold': 0.1,
                'position_sizing': {
                    'max_position_size': 0.1,
                    'min_position_size': 0.01
                },
                'max_positions': 10,
                'max_daily_trades': 10
            }
        )
        
        # Generate performance report
        if 'error' in results:
            logger.error(f"Error in backtest results: {results['error']}")
            return
            
        # Create output directory if it doesn't exist
        output_dir = os.path.join('output', 'backtest')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'backtest_results_{timestamp}.json')
        
        # Convert results to JSON-serializable format
        serializable_results = {
            'initial_capital': float(results['initial_capital']),
            'final_capital': float(results['final_capital']),
            'total_return': float(results['total_return']),
            'annualized_return': float(results['annualized_return']),
            'max_drawdown': float(results['max_drawdown']),
            'win_rate': float(results['win_rate']),
            'num_trades': results['num_trades'],
            'sharpe_ratio': float(results['sharpe_ratio']),
            'trades': [
                {
                    'ticker': trade['ticker'],
                    'timestamp': trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'direction': trade['direction'],
                    'shares': int(trade['shares']),
                    'price': float(trade['price']),
                    'cost': float(trade['cost']),
                    'total': float(trade['total'])
                }
                for trade in results['trades']
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Backtest results saved to {output_file}")
        
        # Print summary
        print("\nBacktest Summary:")
        print(f"Initial Capital: ¥{results['initial_capital']:,.2f}")
        print(f"Final Capital: ¥{results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Annualized Return: {results['annualized_return']*100:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise

def main():
    """Run the backtest."""
    # Get Tushare token from environment variable
    tushare_token = os.getenv('TUSHARE_TOKEN')
    
    if not tushare_token:
        raise ValueError("Please set TUSHARE_TOKEN environment variable")
    
    # Run backtest
    run_recent_backtest(tushare_token)

if __name__ == "__main__":
    main() 