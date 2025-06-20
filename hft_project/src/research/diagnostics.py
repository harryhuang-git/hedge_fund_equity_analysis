"""
Research diagnostics module for analyzing strategy performance and factor contributions.
"""

import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch

from src.portfolio.optimizer import PortfolioOptimizer
from src.research.alpha_decay import AlphaDecayModel

logger = logging.getLogger(__name__)

class ResearchDiagnostics:
    def __init__(
        self,
        output_dir: str = 'outputs/diagnostics',
        plots_dir: str = 'outputs/plots'
    ):
        """
        Initialize research diagnostics.
        
        Args:
            output_dir (str): Directory for diagnostic outputs
            plots_dir (str): Directory for diagnostic plots
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = Path(plots_dir)
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.portfolio_optimizer = PortfolioOptimizer()
        self.alpha_decay_model = AlphaDecayModel()
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def analyze_trades(
        self,
        trades_file: str,
        output_file: Optional[str] = None
    ) -> Dict:
        """
        Analyze trading history.
        
        Args:
            trades_file (str): Path to trades JSON file
            output_file (str, optional): Path to save analysis
            
        Returns:
            Dict: Trade analysis metrics
        """
        # Load trades
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df.reset_index(drop=True)
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calculate decay-adjusted metrics
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        decay_weights = self.alpha_decay_model.calculate_decay_weights(
            trades_df['date'],
            trades_df['date'].max()
        ).values  # get as numpy array
        trades_df['decay_weight'] = decay_weights
        
        decayed_pnl = trades_df['pnl'] * trades_df['decay_weight']
        decayed_win_rate = len(decayed_pnl[decayed_pnl > 0]) / len(decayed_pnl)
        
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'decayed_win_rate': decayed_win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': trades_df['pnl'].sum(),
            'decayed_pnl': decayed_pnl.sum()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=4)
        
        return metrics
    
    def analyze_factor_contributions(
        self,
        factor_scores_file: str,
        returns_file: str,
        output_file: Optional[str] = None
    ) -> Dict:
        """
        Analyze factor contributions to strategy performance.
        
        Args:
            factor_scores_file (str): Path to factor scores JSON file
            returns_file (str): Path to returns JSON file
            output_file (str, optional): Path to save analysis
            
        Returns:
            Dict: Factor contribution metrics
        """
        # Load data
        with open(factor_scores_file, 'r') as f:
            factor_scores = json.load(f)
        
        with open(returns_file, 'r') as f:
            returns = json.load(f)
        
        # Convert to DataFrames
        factor_df = pd.DataFrame(factor_scores)
        returns_df = pd.DataFrame(returns)
        
        # Calculate factor correlations
        factor_corr = factor_df.corrwith(returns_df.shift(-1))
        
        # Calculate factor contributions
        factor_contrib = {}
        for factor in factor_df.columns:
            # Calculate decay-adjusted factor scores
            decayed_scores = self.alpha_decay_model.apply_decay(
                factor_df[[factor]],
                factor_df.index[-1]
            )
            
            # Calculate factor contribution
            factor_contrib[factor] = {
                'ic': factor_corr[factor],
                'decayed_ic': decayed_scores.corr(returns_df.shift(-1).iloc[:, 0]),
                'contribution': (factor_df[factor] * returns_df.shift(-1)).mean()
            }
        
        metrics = {
            'factor_correlations': factor_corr.to_dict(),
            'factor_contributions': factor_contrib
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=4)
        
        return metrics
    
    def plot_rolling_contributions(
        self,
        factor_scores_file: str,
        returns_file: str,
        window: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot rolling factor contributions.
        
        Args:
            factor_scores_file (str): Path to factor scores JSON file
            returns_file (str): Path to returns JSON file
            window (int): Rolling window size
            save_path (str, optional): Path to save plot
        """
        # Load data
        with open(factor_scores_file, 'r') as f:
            factor_scores = json.load(f)
        
        with open(returns_file, 'r') as f:
            returns = json.load(f)
        
        # Convert to DataFrames
        factor_df = pd.DataFrame(factor_scores)
        returns_df = pd.DataFrame(returns)
        
        # Calculate rolling contributions
        rolling_contrib = {}
        for factor in factor_df.columns:
            rolling_contrib[factor] = (
                factor_df[factor].rolling(window)
                .corr(returns_df.shift(-1))
            )
        
        # Plot
        plt.figure(figsize=(12, 6))
        for factor, contrib in rolling_contrib.items():
            plt.plot(contrib, label=factor)
        
        plt.title(f'Rolling {window}-Day Factor Contributions')
        plt.xlabel('Date')
        plt.ylabel('Information Coefficient')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_correlation_heatmap(
        self,
        factor_scores_file: str,
        save_path: Optional[str] = None
    ):
        """
        Plot factor correlation heatmap.
        
        Args:
            factor_scores_file (str): Path to factor scores JSON file
            save_path (str, optional): Path to save plot
        """
        # Load data
        with open(factor_scores_file, 'r') as f:
            factor_scores = json.load(f)
        
        # Convert to DataFrame
        factor_df = pd.DataFrame(factor_scores)
        
        # Calculate correlation matrix
        corr_matrix = factor_df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        
        plt.title('Factor Correlation Heatmap')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_trade_heatmap(
        self,
        trades_file: str,
        save_path: Optional[str] = None
    ):
        """
        Plot trade PnL heatmap.
        
        Args:
            trades_file (str): Path to trades JSON file
            save_path (str, optional): Path to save plot
        """
        # Load trades
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        # Create pivot table
        pivot_table = trades_df.pivot_table(
            values='pnl',
            index='date',
            columns='symbol',
            aggfunc='sum'
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_table,
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'PnL'}
        )
        
        plt.title('Trade PnL Heatmap')
        plt.xlabel('Symbol')
        plt.ylabel('Date')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_portfolio_comparison(
        self,
        returns_file: str,
        factor_scores_file: str,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of optimized vs unoptimized portfolio returns.
        
        Args:
            returns_file (str): Path to returns JSON file
            factor_scores_file (str): Path to factor scores JSON file
            save_path (str, optional): Path to save plot
        """
        # Load data
        with open(returns_file, 'r') as f:
            returns = json.load(f)
        
        with open(factor_scores_file, 'r') as f:
            factor_scores = json.load(f)
        
        # Convert to DataFrames
        returns_df = pd.DataFrame(returns)
        factor_df = pd.DataFrame(factor_scores)
        
        # Calculate unoptimized portfolio returns
        unopt_weights = pd.Series(1/len(returns_df.columns), index=returns_df.columns)
        unopt_returns = (returns_df * unopt_weights).sum(axis=1)
        
        # Calculate optimized portfolio returns
        opt_weights = self.portfolio_optimizer.optimize_portfolio(
            returns_df,
            factor_df.iloc[-1],
            method='risk_parity'
        )
        opt_returns = (returns_df * opt_weights).sum(axis=1)
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot((1 + unopt_returns).cumprod(), label='Unoptimized')
        plt.plot((1 + opt_returns).cumprod(), label='Optimized')
        
        plt.title('Portfolio Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_daily_turnover(
        self,
        trades_file: str,
        save_path: Optional[str] = None
    ):
        """
        Plot daily portfolio turnover.
        
        Args:
            trades_file (str): Path to trades JSON file
            save_path (str, optional): Path to save plot
        """
        # Load trades
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        # Calculate daily turnover
        daily_turnover = trades_df.groupby('date')['volume'].sum()
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(daily_turnover)
        
        plt.title('Daily Portfolio Turnover')
        plt.xlabel('Date')
        plt.ylabel('Turnover')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def main():
    """Example usage of ResearchDiagnostics."""
    # Initialize diagnostics
    diagnostics = ResearchDiagnostics()
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 10
    n_days = 252
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    
    # Generate factor scores
    factor_scores = pd.DataFrame(
        np.random.normal(0, 1, (n_days, n_assets)),
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Generate returns
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.02, (n_days, n_assets)),
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Generate trades
    trades = []
    for date in dates:
        for asset in range(n_assets):
            if np.random.random() < 0.1:  # 10% chance of trade
                trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': f'Asset_{asset}',
                    'volume': np.random.randint(100, 1000),
                    'pnl': np.random.normal(0, 1000)
                })
    
    # Save sample data
    factor_scores.index = factor_scores.index.astype(str)
    returns.index = returns.index.astype(str)
    with open('outputs/diagnostics/factor_scores.json', 'w') as f:
        json.dump(factor_scores.to_dict(), f)
    with open('outputs/diagnostics/returns.json', 'w') as f:
        json.dump(returns.to_dict(), f)
    
    with open('outputs/diagnostics/trades.json', 'w') as f:
        json.dump(trades, f)
    
    # Run diagnostics
    trade_metrics = diagnostics.analyze_trades('outputs/diagnostics/trades.json')
    factor_metrics = diagnostics.analyze_factor_contributions(
        'outputs/diagnostics/factor_scores.json',
        'outputs/diagnostics/returns.json'
    )
    
    print("\nTrade Metrics:")
    print(trade_metrics)
    
    print("\nFactor Metrics:")
    print(factor_metrics)
    
    # Generate plots
    diagnostics.plot_rolling_contributions(
        'outputs/diagnostics/factor_scores.json',
        'outputs/diagnostics/returns.json'
    )
    
    diagnostics.plot_correlation_heatmap('outputs/diagnostics/factor_scores.json')
    
    diagnostics.plot_trade_heatmap('outputs/diagnostics/trades.json')
    
    diagnostics.plot_portfolio_comparison(
        'outputs/diagnostics/returns.json',
        'outputs/diagnostics/factor_scores.json'
    )
    
    diagnostics.plot_daily_turnover('outputs/diagnostics/trades.json')

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    main() 