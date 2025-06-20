"""
Portfolio optimization module implementing risk parity and Black-Litterman methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import cvxpy as cp
from scipy.optimize import minimize
import torch

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        tau: float = 0.05,
        omega_scale: float = 0.05
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate (float): Risk-free rate for Black-Litterman
            tau (float): Prior uncertainty scaling for Black-Litterman
            omega_scale (float): View uncertainty scaling for Black-Litterman
        """
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.omega_scale = omega_scale
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def risk_parity_weights(
        self,
        returns: pd.DataFrame,
        target_risk: Optional[float] = None,
        max_weight: float = 0.2
    ) -> pd.Series:
        """
        Calculate risk parity weights.
        
        Args:
            returns (pd.DataFrame): Asset returns
            target_risk (float, optional): Target portfolio volatility
            max_weight (float): Maximum weight per asset
            
        Returns:
            pd.Series: Asset weights
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Define optimization problem
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        # Risk contribution of each asset
        risk_contrib = []
        for i in range(n_assets):
            risk_contrib.append(
                cp.multiply(weights[i], cp.quad_form(weights, cov_matrix))
            )
        
        # Objective: minimize sum of squared differences in risk contributions
        objective = cp.Minimize(
            cp.sum_squares(
                cp.hstack([
                    risk_contrib[i] - risk_contrib[j]
                    for i in range(n_assets)
                    for j in range(i + 1, n_assets)
                ])
            )
        )
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # weights sum to 1
            weights >= 0,  # no shorting
            weights <= max_weight  # maximum weight constraint
        ]
        
        if target_risk is not None:
            constraints.append(
                cp.quad_form(weights, cov_matrix) <= target_risk**2
            )
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != 'optimal':
            logger.warning("Risk parity optimization did not converge")
            return pd.Series(1/n_assets, index=returns.columns)
        
        return pd.Series(weights.value, index=returns.columns)
    
    def black_litterman_weights(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series,
        views: Dict[str, float],
        view_confidences: Dict[str, float],
        risk_aversion: float = 2.5
    ) -> pd.Series:
        """
        Calculate Black-Litterman portfolio weights.
        
        Args:
            returns (pd.DataFrame): Asset returns
            market_caps (pd.Series): Market capitalization weights
            views (Dict[str, float]): Expected returns for assets with views
            view_confidences (Dict[str, float]): Confidence in each view
            risk_aversion (float): Risk aversion parameter
            
        Returns:
            pd.Series: Asset weights
        """
        # Calculate market equilibrium returns
        cov_matrix = returns.cov()
        pi = risk_aversion * cov_matrix @ market_caps
        
        # Create view matrix
        n_assets = len(returns.columns)
        P = np.zeros((len(views), n_assets))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))
        
        for i, (asset, view) in enumerate(views.items()):
            P[i, returns.columns.get_loc(asset)] = 1
            Q[i] = view
            Omega[i, i] = 1 / view_confidences[asset]
        
        # Calculate posterior returns and covariance
        tau_sigma = self.tau * cov_matrix
        omega_inv = np.linalg.inv(Omega)
        
        # Black-Litterman formula
        post_cov = np.linalg.inv(
            np.linalg.inv(tau_sigma) + P.T @ omega_inv @ P
        )
        post_returns = post_cov @ (
            np.linalg.inv(tau_sigma) @ pi + P.T @ omega_inv @ Q
        )
        
        # Calculate optimal weights
        weights = np.linalg.inv(risk_aversion * post_cov) @ post_returns
        
        # Normalize weights
        weights = weights / np.sum(np.abs(weights))
        
        return pd.Series(weights, index=returns.columns)
    
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        alpha_scores: pd.Series,
        market_caps: Optional[pd.Series] = None,
        method: str = 'risk_parity',
        **kwargs
    ) -> pd.Series:
        """
        Optimize portfolio weights using specified method.
        
        Args:
            returns (pd.DataFrame): Asset returns
            alpha_scores (pd.Series): Alpha scores for each asset
            market_caps (pd.Series, optional): Market capitalization weights
            method (str): Optimization method ('risk_parity' or 'black_litterman')
            **kwargs: Additional arguments for optimization
            
        Returns:
            pd.Series: Asset weights
        """
        if method == 'risk_parity':
            return self.risk_parity_weights(returns, **kwargs)
        
        elif method == 'black_litterman':
            if market_caps is None:
                raise ValueError("Market caps required for Black-Litterman")
            
            # Convert alpha scores to views
            views = {}
            view_confidences = {}
            
            for asset, score in alpha_scores.items():
                if abs(score) > 0.1:  # Only use significant signals
                    views[asset] = score * 0.1  # Scale to expected return
                    view_confidences[asset] = abs(score)
            
            return self.black_litterman_weights(
                returns,
                market_caps,
                views,
                view_confidences,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def calculate_portfolio_metrics(
        self,
        weights: pd.Series,
        returns: pd.DataFrame
    ) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights (pd.Series): Asset weights
            returns (pd.DataFrame): Asset returns
            
        Returns:
            Dict: Portfolio metrics
        """
        # Calculate portfolio returns
        port_returns = (returns * weights).sum(axis=1)
        
        # Calculate metrics
        metrics = {
            'volatility': port_returns.std() * np.sqrt(252),
            'sharpe': (
                (port_returns.mean() * 252 - self.risk_free_rate) /
                (port_returns.std() * np.sqrt(252))
            ),
            'max_drawdown': self._calculate_drawdown(port_returns),
            'turnover': self._calculate_turnover(weights, returns)
        }
        
        return metrics
    
    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (running_max - cum_returns) / running_max
        return drawdown.max()
    
    def _calculate_turnover(
        self,
        weights: pd.Series,
        returns: pd.DataFrame
    ) -> float:
        """Calculate portfolio turnover."""
        # Calculate daily weight changes
        weight_changes = weights.diff().abs()
        return weight_changes.mean()

def main():
    """Example usage of PortfolioOptimizer."""
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 10
    n_days = 252
    
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.02, (n_days, n_assets)),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    alpha_scores = pd.Series(
        np.random.normal(0, 1, n_assets),
        index=returns.columns
    )
    
    market_caps = pd.Series(
        np.random.lognormal(10, 1, n_assets),
        index=returns.columns
    )
    market_caps = market_caps / market_caps.sum()
    
    # Calculate weights using different methods
    risk_parity_weights = optimizer.optimize_portfolio(
        returns,
        alpha_scores,
        method='risk_parity'
    )
    
    black_litterman_weights = optimizer.optimize_portfolio(
        returns,
        alpha_scores,
        market_caps,
        method='black_litterman'
    )
    
    # Calculate metrics
    risk_parity_metrics = optimizer.calculate_portfolio_metrics(
        risk_parity_weights,
        returns
    )
    
    black_litterman_metrics = optimizer.calculate_portfolio_metrics(
        black_litterman_weights,
        returns
    )
    
    print("\nRisk Parity Metrics:")
    print(risk_parity_metrics)
    
    print("\nBlack-Litterman Metrics:")
    print(black_litterman_metrics)

if __name__ == "__main__":
    main() 