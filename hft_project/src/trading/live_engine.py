"""
Real-time trading engine for live and paper trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime, timedelta
import json
import time
from abc import ABC, abstractmethod

from ..data.universe_manager import UniverseManager
from ..models.factors import FactorCalculator
from ..models.rank_signals import SignalGenerator

logger = logging.getLogger(__name__)

class ExecutionEngine(ABC):
    """Abstract base class for execution engines."""
    
    @abstractmethod
    def place_order(self, ticker: str, direction: str, shares: int, price: float) -> Dict:
        """Place an order."""
        pass
    
    @abstractmethod
    def get_position(self, ticker: str) -> int:
        """Get current position for a ticker."""
        pass
    
    @abstractmethod
    def get_account_value(self) -> float:
        """Get current account value."""
        pass

class PaperTradingEngine(ExecutionEngine):
    """Paper trading execution engine."""
    
    def __init__(self, initial_capital: float = 1000000):
        """
        Initialize paper trading engine.
        
        Args:
            initial_capital (float): Initial capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.orders = []
        
    def place_order(self, ticker: str, direction: str, shares: int, price: float) -> Dict:
        """Place a paper trading order."""
        order = {
            'timestamp': datetime.now(),
            'ticker': ticker,
            'direction': direction,
            'shares': shares,
            'price': price,
            'status': 'FILLED'  # Paper trading assumes immediate fill
        }
        
        # Update positions and capital
        if direction == 'BUY':
            cost = shares * price
            if cost > self.current_capital:
                order['status'] = 'REJECTED'
                order['reason'] = 'Insufficient capital'
                return order
            
            self.current_capital -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + shares
            
        else:  # SELL
            if ticker not in self.positions or self.positions[ticker] < shares:
                order['status'] = 'REJECTED'
                order['reason'] = 'Insufficient position'
                return order
            
            proceeds = shares * price
            self.current_capital += proceeds
            self.positions[ticker] -= shares
            
            if self.positions[ticker] == 0:
                del self.positions[ticker]
        
        self.orders.append(order)
        return order
    
    def get_position(self, ticker: str) -> int:
        """Get current position for a ticker."""
        return self.positions.get(ticker, 0)
    
    def get_account_value(self) -> float:
        """Get current account value."""
        return self.current_capital

class LiveTradingEngine(ExecutionEngine):
    """Live trading execution engine (placeholder for broker integration)."""
    
    def __init__(self, broker_config: Dict):
        """
        Initialize live trading engine.
        
        Args:
            broker_config (Dict): Broker configuration
        """
        self.broker_config = broker_config
        # Initialize broker connection here
        
    def place_order(self, ticker: str, direction: str, shares: int, price: float) -> Dict:
        """Place a live trading order."""
        # Implement broker-specific order placement
        raise NotImplementedError("Live trading not implemented yet")
    
    def get_position(self, ticker: str) -> int:
        """Get current position for a ticker."""
        # Implement broker-specific position query
        raise NotImplementedError("Live trading not implemented yet")
    
    def get_account_value(self) -> float:
        """Get current account value."""
        # Implement broker-specific account value query
        raise NotImplementedError("Live trading not implemented yet")

class LiveTradingManager:
    def __init__(
        self,
        universe_manager: UniverseManager,
        strategy_config: Dict,
        execution_engine: ExecutionEngine,
        output_dir: str = 'outputs/signals'
    ):
        """
        Initialize live trading manager.
        
        Args:
            universe_manager (UniverseManager): Universe manager instance
            strategy_config (Dict): Strategy configuration
            execution_engine (ExecutionEngine): Execution engine instance
            output_dir (str): Directory for signal output
        """
        self.universe_manager = universe_manager
        self.strategy_config = strategy_config
        self.execution_engine = execution_engine
        self.output_dir = output_dir
        
        self.factor_calculator = FactorCalculator()
        self.signal_generator = SignalGenerator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_signals(self, date: str) -> Dict:
        """
        Generate trading signals for a specific date.
        
        Args:
            date (str): Trading date
            
        Returns:
            Dict: Trading signals
        """
        # Get universe data
        universe = self.strategy_config.get('universe', 'CSI300')
        stock_data = self.universe_manager.load_universe_data(
            universe,
            date,
            date
        )
        
        if not stock_data:
            logger.warning(f"No data available for {date}")
            return {}
        
        # Calculate factors and generate signals
        signals = {}
        factor_scores = {}
        
        for ticker, data in stock_data.items():
            try:
                # Calculate factors
                factors = self.factor_calculator.calculate_factors(
                    data,
                    self.strategy_config
                )
                
                # Generate signal
                signal = self.signal_generator.generate_signal(
                    factors,
                    self.strategy_config
                )
                
                if signal != 0:  # Only include non-zero signals
                    signals[ticker] = {
                        'signal': signal,
                        'factors': factors,
                        'price': data['close'].iloc[-1]
                    }
                    factor_scores[ticker] = factors
                
            except Exception as e:
                logger.error(f"Error generating signal for {ticker}: {str(e)}")
        
        # Save signals to file
        self._save_signals(date, signals, factor_scores)
        
        return signals
    
    def _save_signals(self, date: str, signals: Dict, factor_scores: Dict):
        """Save trading signals to file."""
        output = {
            'date': date,
            'strategy': self.strategy_config['name'],
            'signals': signals,
            'factor_scores': factor_scores
        }
        
        filename = f"{self.output_dir}/{date}_{self.strategy_config['name']}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=4)
        
        logger.info(f"Saved signals to {filename}")
    
    def execute_signals(self, signals: Dict, max_positions: int = 10):
        """
        Execute trading signals.
        
        Args:
            signals (Dict): Trading signals
            max_positions (int): Maximum number of positions
        """
        # Sort signals by strength
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: abs(x[1]['signal']),
            reverse=True
        )
        
        # Get top N signals
        top_signals = sorted_signals[:max_positions]
        
        # Execute signals
        for ticker, signal_data in top_signals:
            current_position = self.execution_engine.get_position(ticker)
            signal = signal_data['signal']
            price = signal_data['price']
            
            if signal > 0 and current_position <= 0:
                # Buy signal
                shares = int(self.execution_engine.get_account_value() * 0.1 / price)
                if shares > 0:
                    self.execution_engine.place_order(ticker, 'BUY', shares, price)
                    
            elif signal < 0 and current_position >= 0:
                # Sell signal
                if current_position > 0:
                    self.execution_engine.place_order(ticker, 'SELL', current_position, price)
    
    def run_daily_trading(self, date: Optional[str] = None):
        """
        Run daily trading process.
        
        Args:
            date (str, optional): Trading date. If None, use current date.
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        logger.info(f"Running daily trading for {date}")
        
        # Generate signals
        signals = self.generate_signals(date)
        
        if signals:
            # Execute signals
            self.execute_signals(signals)
            
            # Log account status
            account_value = self.execution_engine.get_account_value()
            logger.info(f"Account value: ${account_value:,.2f}")
        else:
            logger.warning("No signals generated for today")

def main():
    """Example usage of LiveTradingManager."""
    # Initialize components
    universe_manager = UniverseManager('YOUR_TUSHARE_TOKEN')
    
    # Load strategy configuration
    with open('config/strategies/momentum.json', 'r') as f:
        strategy_config = json.load(f)
    
    # Initialize paper trading engine
    execution_engine = PaperTradingEngine(initial_capital=1000000)
    
    # Initialize trading manager
    manager = LiveTradingManager(
        universe_manager=universe_manager,
        strategy_config=strategy_config,
        execution_engine=execution_engine
    )
    
    # Run daily trading
    manager.run_daily_trading()

if __name__ == "__main__":
    main() 