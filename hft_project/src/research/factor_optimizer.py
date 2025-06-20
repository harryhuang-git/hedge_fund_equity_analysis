import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from typing import Dict, Any, Callable

class FactorOptimizer:
    def __init__(self, method='grid', risk_adjusted=True, risk_penalty=0.1, custom_objective: Callable = None):
        self.method = method
        self.risk_adjusted = risk_adjusted
        self.risk_penalty = risk_penalty
        self.custom_objective = custom_objective

    def normalize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        return (factor_df - factor_df.mean()) / (factor_df.std() + 1e-8)

    def risk_adjust(self, signal: pd.Series, returns: pd.Series) -> float:
        # Sharpe ratio minus penalty for turnover (or other risk metric)
        sharpe = signal.corr(returns) / (signal.std() + 1e-8)
        penalty = self.risk_penalty * np.abs(signal.diff()).mean()
        return sharpe - penalty

    def optimize(self, factor_df: pd.DataFrame, returns: pd.Series, param_grid: Dict[str, Any], max_iter=50) -> Dict[str, Any]:
        factor_df = self.normalize_factors(factor_df)
        best_score = -np.inf
        best_params = None
        best_signal = None
        if self.method == 'grid':
            for params in ParameterGrid(param_grid):
                signal = sum(params[f] * factor_df[f] for f in factor_df.columns)
                score = self.risk_adjust(signal, returns) if self.risk_adjusted else signal.corr(returns)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_signal = signal
        elif self.method == 'genetic':
            # Placeholder: You can plug in DEAP or other GA library here
            raise NotImplementedError('Genetic optimization not implemented yet.')
        elif self.method == 'bayesian':
            # Placeholder: You can plug in skopt or optuna here
            raise NotImplementedError('Bayesian optimization not implemented yet.')
        elif self.custom_objective:
            # Custom optimizer
            best_params, best_score, best_signal = self.custom_objective(factor_df, returns, param_grid)
        else:
            raise ValueError('Unknown optimization method.')
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_signal': best_signal
        }

    def composite_alpha(self, factor_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        factor_df = self.normalize_factors(factor_df)
        return sum(weights[f] * factor_df[f] for f in factor_df.columns) 