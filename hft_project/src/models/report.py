"""
Performance reporting module for the HFT project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class PerformanceReport:
    def __init__(self, equity_curve: pd.DataFrame, trades: List[Dict], output_dir: str = 'outputs'):
        """
        Initialize performance report generator.
        
        Args:
            equity_curve (pd.DataFrame): DataFrame with date and equity columns
            trades (List[Dict]): List of trade dictionaries
            output_dir (str): Directory to save reports and plots
        """
        self.equity_curve = equity_curve.copy()
        self.trades = pd.DataFrame(trades)
        self.output_dir = output_dir
        self._setup_directories()
        
        # Set style for plots
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def _setup_directories(self):
        """Create necessary directories for outputs."""
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        # Calculate returns
        self.equity_curve['returns'] = self.equity_curve['equity'].pct_change()
        daily_returns = self.equity_curve['returns'].dropna()
        
        # Basic metrics
        total_return = (self.equity_curve['equity'].iloc[-1] / self.equity_curve['equity'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(self.equity_curve)) - 1
        annual_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # Drawdown metrics
        self.equity_curve['peak'] = self.equity_curve['equity'].cummax()
        self.equity_curve['drawdown'] = (self.equity_curve['equity'] - self.equity_curve['peak']) / self.equity_curve['peak']
        max_drawdown = self.equity_curve['drawdown'].min()
        
        # Win rate
        if not self.trades.empty:
            winning_trades = self.trades[self.trades['pnl'] > 0]
            win_rate = len(winning_trades) / len(self.trades)
        else:
            win_rate = 0
        
        # Daily turnover
        if not self.trades.empty:
            daily_turnover = self.trades.groupby('date')['transaction_cost'].sum().mean()
        else:
            daily_turnover = 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'daily_turnover': daily_turnover,
            'total_trades': len(self.trades),
            'profitable_trades': len(winning_trades) if not self.trades.empty else 0
        }
        
        return metrics
    
    def plot_equity_curve(self, save: bool = True):
        """Plot equity curve."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['equity'], label='Portfolio Value')
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (CNY)')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(f"{self.output_dir}/plots/equity_curve.png")
            plt.close()
        else:
            plt.show()
    
    def plot_drawdown(self, save: bool = True):
        """Plot drawdown curve."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve['drawdown'] * 100, label='Drawdown', color='red')
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(f"{self.output_dir}/plots/drawdown.png")
            plt.close()
        else:
            plt.show()
    
    def plot_monthly_returns(self, save: bool = True):
        """Plot monthly returns heatmap."""
        # Calculate monthly returns
        monthly_returns = self.equity_curve['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create monthly returns matrix
        returns_matrix = monthly_returns.to_frame()
        returns_matrix.index = pd.to_datetime(returns_matrix.index)
        returns_matrix['year'] = returns_matrix.index.year
        returns_matrix['month'] = returns_matrix.index.month
        returns_matrix = returns_matrix.pivot(index='year', columns='month', values='returns')
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(returns_matrix * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0)
        plt.title('Monthly Returns (%)')
        
        if save:
            plt.savefig(f"{self.output_dir}/plots/monthly_returns.png")
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, strategy_name: str = "default"):
        """
        Generate complete performance report.
        
        Args:
            strategy_name (str): Name of the strategy for report identification
        """
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Generate plots
        self.plot_equity_curve()
        self.plot_drawdown()
        self.plot_monthly_returns()
        
        # Save metrics to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = f"{self.output_dir}/metrics/{strategy_name}_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log metrics
        logger.info(f"Performance Report for {strategy_name}:")
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {metrics['annual_return']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        
        return metrics

if __name__ == "__main__":
    # Example usage
    from src.data.market_data import MarketDataCollector
    from src.models.backtest_engine import BacktestEngine
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/backtest.log'),
            logging.StreamHandler()
        ]
    )
    
    # Get sample data and run backtest
    collector = MarketDataCollector()
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime('%Y%m%d')
    
    stock_list = ['000001.SZ', '600000.SH', '601318.SH']
    stock_data = {}
    
    for stock in stock_list:
        data = collector.get_daily_data(stock, start_date, end_date)
        if data is not None:
            stock_data[stock] = data
    
    if stock_data:
        # Run backtest
        backtest = BacktestEngine(stock_data)
        backtest.run_backtest(start_date, end_date)
        
        # Generate report
        report = PerformanceReport(
            pd.DataFrame(backtest.equity_curve),
            backtest.trades
        )
        report.generate_report("sample_strategy") 