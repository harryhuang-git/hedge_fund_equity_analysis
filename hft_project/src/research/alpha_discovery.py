"""
Alpha discovery module for strategy generation and testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
from itertools import product
from joblib import Parallel, delayed, Memory
import traceback

from ..data.universe_manager import UniverseManager
from ..models.factors import FactorCalculator
from ..models.rank_signals import SignalGenerator

logger = logging.getLogger(__name__)

# Set up a disk cache for factor calculations
memory = Memory(location='outputs/research/factor_cache', verbose=0)

class AlphaDiscovery:
    def __init__(
        self,
        universe_manager: UniverseManager,
        base_config: Dict,
        output_dir: str = 'outputs/research'
    ):
        """
        Initialize alpha discovery module.
        
        Args:
            universe_manager (UniverseManager): Universe manager instance
            base_config (Dict): Base strategy configuration
            output_dir (str): Directory for research output
        """
        self.universe_manager = universe_manager
        self.base_config = base_config
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/grid_search", exist_ok=True)
        os.makedirs(f"{output_dir}/ml_models", exist_ok=True)
    
    def calculate_daily_sector_metrics(self, stock_data: dict) -> pd.DataFrame:
        # Aggregate daily returns to sector level
        sector_metrics = {}
        for ticker, df in stock_data.items():
            sector = df['industry'].iloc[0]
            if sector not in sector_metrics:
                sector_metrics[sector] = []
            daily_return = df['close'].pct_change().rename(ticker)
            sector_metrics[sector].append(daily_return)
        sector_df = {}
        for sector, returns_list in sector_metrics.items():
            sector_df[sector] = pd.concat(returns_list, axis=1).mean(axis=1)
        sector_df = pd.DataFrame(sector_df)
        return sector_df

    def get_dynamic_hot_sectors(self, sector_df, lookback=5, top_n=3):
        # Calculate rolling momentum
        momentum = sector_df.rolling(window=lookback).mean().iloc[-1]
        hot_sectors = momentum.sort_values(ascending=False).head(top_n).index.tolist()
        return hot_sectors

    def grid_search(
        self,
        param_grid: Dict,
        start_date: str,
        end_date: str,
        universe: str = 'CSI300',
        n_splits: int = 5,
        focus_sectors: list = None,
        dynamic_sector: bool = True,
        sector_lookback: int = 5,
        sector_top_n: int = 3
    ) -> pd.DataFrame:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid (Dict): Parameter grid to search
            start_date (str): Start date
            end_date (str): End date
            universe (str): Universe to test on
            n_splits (int): Number of time series splits
            focus_sectors (list): List of sectors to focus on
            dynamic_sector (bool): Whether to dynamically select hot sectors
            sector_lookback (int): Lookback period for dynamic sector selection
            sector_top_n (int): Number of top sectors to select
            
        Returns:
            pd.DataFrame: Grid search results
        """
        print('Loading universe data...')
        logger.info(f'Loading universe data for {universe} from {start_date} to {end_date}')
        # Get universe data
        try:
            stock_data = self.universe_manager.load_universe_data(
                universe,
                start_date,
                end_date
            )
        except Exception as e:
            logger.error(f'Exception during universe data loading: {e}')
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        # Dynamic sector selection
        if dynamic_sector:
            sector_df = self.calculate_daily_sector_metrics(stock_data)
            hot_sectors = self.get_dynamic_hot_sectors(sector_df, lookback=sector_lookback, top_n=sector_top_n)
            stock_data = {k: v for k, v in stock_data.items() if v['industry'].iloc[0] in hot_sectors}
            logger.info(f'Dynamically selected hot sectors: {hot_sectors} (lookback={sector_lookback}, top_n={sector_top_n})')
        elif focus_sectors:
            stock_data = {k: v for k, v in stock_data.items() if v['industry'].iloc[0] in focus_sectors}
            logger.info(f'Filtered to {len(stock_data)} stocks in focus sectors: {focus_sectors}')
        print(f'Loaded universe data: {len(stock_data)} stocks')
        logger.info(f'Loaded universe data: {len(stock_data)} stocks')
        if not stock_data:
            logger.error("No data available for grid search")
            return pd.DataFrame()
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        print(f'Generated {len(param_combinations)} parameter combinations')
        logger.info(f'Generated {len(param_combinations)} parameter combinations')
        
        # Prepare results storage
        results = []
        
        # Time series cross-validation
        dates = sorted(list(stock_data[list(stock_data.keys())[0]].index))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for i, params in enumerate(param_combinations):
            print(f'Starting parameter combination {i+1}/{len(param_combinations)}: {params}')
            logger.info(f'Starting parameter combination {i+1}/{len(param_combinations)}: {params}')
            # Create strategy config with current parameters
            config = self.base_config.copy()
            for name, value in zip(param_names, params):
                config[name] = value
            
            # Cross-validation
            cv_results = []
            
            for j, (train_idx, test_idx) in enumerate(tscv.split(dates)):
                print(f'  CV split {j+1}/{n_splits}')
                logger.info(f'  CV split {j+1}/{n_splits}')
                train_dates = [dates[k] for k in train_idx]
                test_dates = [dates[k] for k in test_idx]
                print(f'    Calling _evaluate_strategy for split {j+1}')
                logger.info(f'    Calling _evaluate_strategy for split {j+1}')
                try:
                    metrics = self._evaluate_strategy(
                        stock_data,
                        config,
                        train_dates,
                        test_dates
                    )
                except Exception as e:
                    logger.error(f'Exception in _evaluate_strategy: {e}')
                    logger.error(traceback.format_exc())
                    metrics = {'sharpe': 0, 'returns': 0, 'drawdown': 0, 'win_rate': 0}
                cv_results.append(metrics)
            
            # Aggregate results
            avg_metrics = {
                'sharpe': np.mean([r['sharpe'] for r in cv_results]),
                'returns': np.mean([r['returns'] for r in cv_results]),
                'drawdown': np.mean([r['drawdown'] for r in cv_results]),
                'win_rate': np.mean([r['win_rate'] for r in cv_results])
            }
            
            # Attribute by sector (simple: assign all stocks in sector same metrics)
            sector_metrics = {}
            for ticker, df in stock_data.items():
                sector = df['industry'].iloc[0]
                if sector not in sector_metrics:
                    sector_metrics[sector] = {'returns': [], 'sharpe': [], 'drawdown': [], 'win_rate': []}
            for j, (train_idx, test_idx) in enumerate(tscv.split(dates)):
                # ... existing code ...
                # After metrics = ...
                # Attribute by sector (simple: assign all stocks in sector same metrics)
                for ticker, df in stock_data.items():
                    sector = df['industry'].iloc[0]
                    sector_metrics[sector]['returns'].append(metrics['returns'])
                    sector_metrics[sector]['sharpe'].append(metrics['sharpe'])
                    sector_metrics[sector]['drawdown'].append(metrics['drawdown'])
                    sector_metrics[sector]['win_rate'].append(metrics['win_rate'])
            # Aggregate sector metrics
            for sector in sector_metrics:
                sector_metrics[sector] = {k: float(np.mean(v)) for k, v in sector_metrics[sector].items()}
            
            # Add parameters to results
            result = {**dict(zip(param_names, params)), **avg_metrics, **sector_metrics}
            results.append(result)
            logger.info(f'Finished parameter combination {i+1}/{len(param_combinations)}: {params} with metrics {avg_metrics}')
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(
            f"{self.output_dir}/grid_search/results_{datetime.now().strftime('%Y%m%d')}.csv",
            index=False
        )
        logger.info(f'Saved grid search results to {self.output_dir}/grid_search/results_{datetime.now().strftime('%Y%m%d')}.csv')
        
        return results_df
    
    def train_ml_model(
        self,
        start_date: str,
        end_date: str,
        universe: str = 'CSI300',
        model_type: str = 'rf',
        target_horizon: int = 1,
        focus_sectors: list = None,
        dynamic_sector: bool = True,
        sector_lookback: int = 5,
        sector_top_n: int = 3
    ) -> Dict:
        """
        Train machine learning model for return prediction.
        
        Args:
            start_date (str): Start date
            end_date (str): End date
            universe (str): Universe to train on
            model_type (str): Model type ('rf' or 'xgb')
            target_horizon (int): Prediction horizon in days
            focus_sectors (list): List of sectors to focus on
            dynamic_sector (bool): Whether to dynamically select hot sectors
            sector_lookback (int): Lookback period for dynamic sector selection
            sector_top_n (int): Number of top sectors to select
            
        Returns:
            Dict: Model information and performance metrics
        """
        # Get universe data
        stock_data = self.universe_manager.load_universe_data(
            universe,
            start_date,
            end_date
        )
        # Dynamic sector selection
        if dynamic_sector:
            sector_df = self.calculate_daily_sector_metrics(stock_data)
            hot_sectors = self.get_dynamic_hot_sectors(sector_df, lookback=sector_lookback, top_n=sector_top_n)
            stock_data = {k: v for k, v in stock_data.items() if v['industry'].iloc[0] in hot_sectors}
            logger.info(f'Dynamically selected hot sectors: {hot_sectors} (lookback={sector_lookback}, top_n={sector_top_n})')
        elif focus_sectors:
            stock_data = {k: v for k, v in stock_data.items() if v['industry'].iloc[0] in focus_sectors}
            logger.info(f'Filtered to {len(stock_data)} stocks in focus sectors: {focus_sectors}')
        
        if not stock_data:
            logger.error("No data available for model training")
            return {}
        
        # Prepare features and targets
        X, y = self._prepare_ml_data(stock_data, target_horizon)
        
        # Add sector/industry as a categorical feature (one-hot)
        if not X.empty and 'industry' in X.columns:
            X = pd.get_dummies(X, columns=['industry'])
        
        if X.empty or y.empty:
            logger.error("Failed to prepare ML data")
            return {}
        
        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        if model_type == 'rf' or not XGBOOST_AVAILABLE:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:  # xgb
            if not XGBOOST_AVAILABLE:
                logger.warning('XGBoost not available, falling back to RandomForestRegressor')
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        metrics = {
            'train_r2': model.score(X_train, y_train),
            'test_r2': model.score(X_test, y_test),
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
        # Save model and metrics
        model_info = {
            'model_type': model_type,
            'train_date': start_date,
            'end_date': end_date,
            'universe': universe,
            'metrics': metrics,
            'feature_names': list(X.columns)
        }
        
        with open(
            f"{self.output_dir}/ml_models/model_{datetime.now().strftime('%Y%m%d')}.json",
            'w'
        ) as f:
            json.dump(model_info, f, indent=4)
        
        return model_info
    
    def _evaluate_strategy(
        self,
        stock_data: Dict[str, pd.DataFrame],
        config: Dict,
        train_dates: List[str],
        test_dates: List[str]
    ) -> Dict:
        print('  _evaluate_strategy: entered')
        # Calculate factors and signals
        signals = {}
        returns = []
        for date_idx, date in enumerate(test_dates):
            print(f'    Evaluating date {date_idx+1}/{len(test_dates)}: {date}')
            date_signals = {}
            # Multiprocessing per-ticker factor+signal calculation
            @memory.cache
            def cached_factors(ticker, data, config, date):
                # Only cache up to this date for this config
                return FactorCalculator(data.loc[:date]).calculate_factors(
                    data.loc[:date],
                    config
                )
            def process_ticker(ticker, data):
                try:
                    print(f'      Ticker: {ticker}')
                    print('        Calculating factors...')
                    factors = cached_factors(ticker, data, config, date)
                    print('        Factors calculated')
                    factors['ticker'] = ticker
                    print('        Generating signal...')
                    signal = SignalGenerator({ticker: data}).generate_signal(
                        factors,
                        config
                    )
                    print('        Signal generated')
                    if not signal.empty and signal.iloc[0] != 0:
                        return (ticker, signal, data.loc[date, 'close'])
                except Exception as e:
                    logger.error(f"Error evaluating {ticker} on {date}: {str(e)}")
                return None
            # Use joblib.Parallel for true multiprocessing
            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(process_ticker)(ticker, data) for ticker, data in stock_data.items()
            )
            for result in results:
                if result is not None:
                    ticker, signal, price = result
                    date_signals[ticker] = {
                        'signal': signal,
                        'price': price
                    }
            # Calculate daily returns
            if date_signals:
                daily_return = self._calculate_portfolio_return(
                    date_signals,
                    stock_data,
                    date
                )
                returns.append(daily_return)
        if not returns:
            return {
                'sharpe': 0,
                'returns': 0,
                'drawdown': 0,
                'win_rate': 0
            }
        # Calculate metrics
        returns = np.array(returns)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        total_return = np.prod(1 + returns) - 1
        drawdown = self._calculate_drawdown(returns)
        win_rate = np.mean(returns > 0)
        return {
            'sharpe': sharpe,
            'returns': total_return,
            'drawdown': drawdown,
            'win_rate': win_rate
        }
    
    def _prepare_ml_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        target_horizon: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning.
        
        Args:
            stock_data (Dict[str, pd.DataFrame]): Stock data
            target_horizon (int): Prediction horizon
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and targets
        """
        features = []
        targets = []
        
        for ticker, data in stock_data.items():
            try:
                # Calculate factors
                factors = FactorCalculator(data).calculate_factors(
                    data,
                    self.base_config
                )
                factors['ticker'] = ticker
                # Add industry as a feature
                if 'industry' in data.columns:
                    factors['industry'] = data['industry'].iloc[0]
                # Calculate future returns
                future_returns = data['close'].pct_change(target_horizon).shift(-target_horizon)
                # Combine features and targets
                df = pd.concat([factors, future_returns], axis=1)
                df = df.dropna()
                # Exclude 'ticker' from features
                feature_cols = [col for col in df.columns[:-1] if col != 'ticker']
                features.append(df[feature_cols])
                targets.append(df.iloc[:, -1])
            except Exception as e:
                logger.error(f"Error preparing ML data for {ticker}: {str(e)}")
        if not features:
            return pd.DataFrame(), pd.Series()
        X = pd.concat(features, axis=0)
        y = pd.concat(targets, axis=0)
        return X, y
    
    def _calculate_portfolio_return(
        self,
        signals: Dict,
        stock_data: Dict[str, pd.DataFrame],
        date: str
    ) -> float:
        """Calculate portfolio return for a given date."""
        returns = []
        weights = []
        
        for ticker, signal_data in signals.items():
            try:
                # Get next day's return
                next_date = pd.date_range(date, periods=2)[1].strftime('%Y%m%d')
                if next_date in stock_data[ticker].index:
                    price = signal_data['price']
                    next_price = stock_data[ticker].loc[next_date, 'close']
                    returns.append((next_price - price) / price)
                    weights.append(abs(signal_data['signal']))
            except Exception as e:
                logger.error(f"Error calculating return for {ticker}: {str(e)}")
        
        if not returns:
            return 0
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        return np.sum(np.array(returns) * weights)
    
    def _calculate_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (running_max - cum_returns) / running_max
        return np.max(drawdown)

def main():
    print('AlphaDiscovery main() started')
    # Initialize components
    universe_manager = UniverseManager('YOUR_TUSHARE_TOKEN')
    print('UniverseManager initialized')
    # Load base configuration
    with open('config/strategies/momentum.json', 'r') as f:
        base_config = json.load(f)
    print('Base config loaded')
    # Initialize alpha discovery
    discovery = AlphaDiscovery(
        universe_manager=universe_manager,
        base_config=base_config
    )
    print('AlphaDiscovery instance created')
    # Define parameter grid
    param_grid = {
        'rsi_period': [14, 21, 28],
        'roc_period': [10, 20, 30],
        'sma_period': [20, 50, 100]
    }
    print('Parameter grid defined')
    # Run grid search
    print('Starting grid search...')
    results = discovery.grid_search(
        param_grid=param_grid,
        start_date='20200101',
        end_date='20231231',
        universe='CSI300'
    )
    print('Grid search complete')
    # Train ML model
    print('Starting ML model training...')
    model_info = discovery.train_ml_model(
        start_date='20200101',
        end_date='20231231',
        universe='CSI300',
        model_type='rf'
    )
    print('ML model training complete')

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    main() 