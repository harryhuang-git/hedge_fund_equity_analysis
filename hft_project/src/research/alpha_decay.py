"""
Alpha decay modeling module for handling signal decay and turnover optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy.optimize import minimize
import torch

logger = logging.getLogger(__name__)

class AlphaDecayModel:
    def __init__(
        self,
        decay_half_life: float = 5.0,
        min_signal_strength: float = 0.1,
        max_turnover: float = 0.2
    ):
        """
        Initialize alpha decay model.
        
        Args:
            decay_half_life (float): Half-life of alpha signals in days
            min_signal_strength (float): Minimum signal strength to consider
            max_turnover (float): Maximum daily turnover allowed
        """
        self.decay_half_life = decay_half_life
        self.min_signal_strength = min_signal_strength
        self.max_turnover = max_turnover
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def calculate_decay_weights(
        self,
        dates: pd.DatetimeIndex,
        current_date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculate exponential decay weights for signals.
        
        Args:
            dates (pd.DatetimeIndex): Historical dates
            current_date (pd.Timestamp): Current date
            
        Returns:
            pd.Series: Decay weights for each date
        """
        # Ensure dates is a DatetimeIndex
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        days_diff = (current_date - dates) // pd.Timedelta('1D')
        decay_weights = np.exp(-np.log(2) * days_diff / self.decay_half_life)
        return pd.Series(decay_weights, index=dates)
    
    def apply_decay(
        self,
        signals: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> pd.Series:
        """
        Apply decay to historical signals.
        
        Args:
            signals (pd.DataFrame): Historical signals
            current_date (pd.Timestamp): Current date
            
        Returns:
            pd.Series: Decay-adjusted signals
        """
        if not isinstance(current_date, pd.Timestamp):
            current_date = pd.to_datetime(current_date)
        decay_weights = self.calculate_decay_weights(signals.index, current_date)
        decayed_signals = signals.multiply(decay_weights, axis=0)
        return decayed_signals.sum()
    
    def optimize_turnover(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        costs: pd.Series
    ) -> pd.Series:
        """
        Optimize portfolio weights considering turnover costs.
        
        Args:
            current_weights (pd.Series): Current portfolio weights
            target_weights (pd.Series): Target portfolio weights
            costs (pd.Series): Trading costs per asset
            
        Returns:
            pd.Series: Optimized weights
        """
        def objective(weights):
            # Calculate turnover cost
            turnover = np.abs(weights - current_weights)
            cost = (turnover * costs).sum()
            
            # Calculate tracking error
            tracking_error = np.sum((weights - target_weights) ** 2)
            
            return cost + tracking_error
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda x: self.max_turnover - np.sum(np.abs(x - current_weights))}  # turnover constraint
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(len(current_weights))]
        
        # Initial guess
        x0 = current_weights.values
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Turnover optimization did not converge")
            return current_weights
        
        return pd.Series(result.x, index=current_weights.index)
    
    def calculate_decay_metrics(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame
    ) -> Dict:
        """
        Calculate metrics for alpha decay analysis.
        
        Args:
            signals (pd.DataFrame): Historical signals
            returns (pd.DataFrame): Asset returns
            
        Returns:
            Dict: Decay metrics
        """
        # Calculate signal decay over time
        decay_rates = []
        for i in range(1, len(signals)):
            current_signals = signals.iloc[i]
            prev_signals = signals.iloc[i-1]
            decay_rate = 1 - np.corrcoef(current_signals, prev_signals)[0, 1]
            decay_rates.append(decay_rate)
        
        # Calculate signal-to-noise ratio
        signal_std = signals.std()
        noise_std = (signals - signals.shift(1)).std()
        snr = signal_std / noise_std
        
        # Calculate predictive power
        forward_returns = returns.shift(-1)
        ic = signals.corrwith(forward_returns).mean()
        
        metrics = {
            'mean_decay_rate': np.mean(decay_rates),
            'signal_to_noise': snr.mean(),
            'information_coefficient': ic,
            'half_life_estimate': -np.log(2) / np.mean(decay_rates)
        }
        
        return metrics
    
    def plot_decay_analysis(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot alpha decay analysis.
        
        Args:
            signals (pd.DataFrame): Historical signals
            returns (pd.DataFrame): Asset returns
            save_path (str, optional): Path to save plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Calculate decay metrics
        metrics = self.calculate_decay_metrics(signals, returns)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Signal decay over time
        decay_rates = []
        for i in range(1, len(signals)):
            current_signals = signals.iloc[i]
            prev_signals = signals.iloc[i-1]
            decay_rate = 1 - np.corrcoef(current_signals, prev_signals)[0, 1]
            decay_rates.append(decay_rate)
        
        axes[0, 0].plot(decay_rates)
        axes[0, 0].set_title('Signal Decay Rate Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Decay Rate')
        
        # Plot 2: Signal-to-noise ratio
        snr = signals.std() / (signals - signals.shift(1)).std()
        sns.histplot(snr, ax=axes[0, 1])
        axes[0, 1].set_title('Signal-to-Noise Ratio Distribution')
        axes[0, 1].set_xlabel('SNR')
        
        # Plot 3: Information coefficient
        ic = signals.corrwith(returns.shift(-1))
        sns.histplot(ic, ax=axes[1, 0])
        axes[1, 0].set_title('Information Coefficient Distribution')
        axes[1, 0].set_xlabel('IC')
        
        # Plot 4: Decay-adjusted returns
        decayed_signals = self.apply_decay(signals, signals.index[-1])
        forward_returns = returns.shift(-1).iloc[-1]
        axes[1, 1].scatter(decayed_signals, forward_returns)
        axes[1, 1].set_title('Decay-Adjusted Signals vs Returns')
        axes[1, 1].set_xlabel('Decay-Adjusted Signal')
        axes[1, 1].set_ylabel('Forward Return')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

def main():
    """Example usage of AlphaDecayModel."""
    # Initialize model
    model = AlphaDecayModel()
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 10
    n_days = 252
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    
    signals = pd.DataFrame(
        np.random.normal(0, 1, (n_days, n_assets)),
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.02, (n_days, n_assets)),
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Calculate decay-adjusted signals
    current_date = dates[-1]
    decayed_signals = model.apply_decay(signals, current_date)
    
    # Calculate metrics
    metrics = model.calculate_decay_metrics(signals, returns)
    
    print("\nDecay Metrics:")
    print(metrics)
    
    # Plot analysis
    model.plot_decay_analysis(signals, returns)

if __name__ == "__main__":
    main() 