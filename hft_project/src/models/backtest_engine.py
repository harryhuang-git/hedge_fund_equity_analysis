"""
Backtesting engine for HFT strategies with realistic execution simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import json
import os

from src.models.factors import FactorCalculator
from src.models.rank_signals import SignalGenerator
from src.data.universe_manager import UniverseManager

logger = logging.getLogger(__name__)

# Chinese market specific constants
MIN_TRADE_VALUE = 10000  # Minimum trade value in CNY (10,000 CNY)
MAX_POSITION_SIZE = 0.1  # Maximum position size as fraction of capital (10%)
MIN_POSITION_SIZE = 0.01  # Minimum position size as fraction of capital (1%)
MAX_DAILY_TRADES = 10  # Maximum number of trades per day
STOCK_PRICE_PRECISION = 2  # Price precision for Chinese stocks
MIN_STOCK_PRICE = 0.01  # Minimum stock price in CNY
MAX_STOCK_PRICE = 10000  # Maximum stock price in CNY

class BacktestEngine:
    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        initial_capital: float = 1000000,  # 1 million CNY
        transaction_cost: float = 0.0003,  # 0.03% transaction cost
        slippage: float = 0.0001,  # 0.01% slippage
        execution_delay: int = 1
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock data
            initial_capital (float): Initial capital in CNY
            transaction_cost (float): Transaction cost as a percentage
            slippage (float): Slippage as a percentage
            execution_delay (int): Number of periods delay in execution
        """
        self.stock_data = stock_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.execution_delay = execution_delay
        
        self.current_capital = initial_capital
        self.cash = initial_capital  # Track cash separately
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_trades = {}  # Track number of trades per day
        
    def _validate_price(self, price: float) -> bool:
        """
        Validate stock price according to Chinese market rules.
        
        Args:
            price (float): Stock price to validate
            
        Returns:
            bool: True if price is valid, False otherwise
        """
        if not (MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE):
            logger.warning(f"Invalid price {price} outside allowed range [{MIN_STOCK_PRICE}, {MAX_STOCK_PRICE}]")
            return False
        return True
    
    def calculate_position_size(
        self,
        signal: float,
        price: float,
        volatility: float,
        max_position_size: float = MAX_POSITION_SIZE
    ) -> int:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            signal (float): Trading signal (-1 to 1)
            price (float): Current price
            volatility (float): Current volatility
            max_position_size (float): Maximum position size as fraction of capital
            
        Returns:
            int: Number of shares to trade
        """
        if not self._validate_price(price):
            return 0
            
        if volatility < 0:
            logger.warning(f"Invalid volatility {volatility} for position sizing")
            volatility = 0.1
            
        # Adjust position size based on signal strength and volatility
        position_size = abs(signal) * max_position_size
        position_size = position_size / (1 + volatility)  # Reduce size for high volatility
        
        # Ensure minimum position size
        position_size = max(position_size, MIN_POSITION_SIZE)
        
        # Calculate number of shares with maximum position limit
        max_shares = int((self.current_capital * max_position_size) / price)
        shares = int((self.current_capital * position_size) / price)
        
        # Ensure shares don't exceed maximum position size
        shares = min(shares, max_shares)
        
        # Ensure minimum trade value
        if shares * price < MIN_TRADE_VALUE:
            logger.info(f"Trade value {shares * price} below minimum {MIN_TRADE_VALUE}")
            return 0
            
        return max(0, shares)  # Ensure non-negative shares
    
    def _check_daily_trade_limit(self, date: datetime) -> bool:
        """
        Check if daily trade limit has been reached.
        
        Args:
            date (datetime): Current date
            
        Returns:
            bool: True if more trades are allowed, False otherwise
        """
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in self.daily_trades:
            self.daily_trades[date_str] = 0
            
        if self.daily_trades[date_str] >= MAX_DAILY_TRADES:
            logger.warning(f"Daily trade limit reached for {date_str}")
            return False
            
        return True
    
    def simulate_execution(
        self,
        ticker: str,
        shares: int,
        price: float,
        timestamp: datetime,
        direction: str
    ) -> Optional[Dict]:
        """
        Simulate realistic trade execution with slippage and delay.
        
        Args:
            ticker (str): Stock ticker
            shares (int): Number of shares to trade
            price (float): Current price
            timestamp (datetime): Current timestamp
            direction (str): 'buy' or 'sell'
            
        Returns:
            Optional[Dict]: Trade execution details or None if trade fails
        """
        if not self._check_daily_trade_limit(timestamp):
            return None
            
        if direction not in ['buy', 'sell']:
            raise ValueError(f"Invalid direction: {direction}")
            
        if shares <= 0:
            raise ValueError(f"Invalid shares: {shares}")
            
        if not self._validate_price(price):
            return None
        
        # Apply slippage
        slippage_factor = 1 + self.slippage if direction == 'buy' else 1 - self.slippage
        execution_price = round(price * slippage_factor, STOCK_PRICE_PRECISION)
        
        # Calculate transaction cost
        cost = round(shares * execution_price * self.transaction_cost, 2)
        
        # Calculate total cost/proceeds
        if direction == 'buy':
            total_cost = shares * execution_price + cost
            if total_cost > self.current_capital:
                logger.warning(f"Insufficient capital for trade: {total_cost} > {self.current_capital}")
                return None
            self.current_capital -= total_cost
        else:
            total_proceeds = shares * execution_price - cost
            self.current_capital += total_proceeds
        
        # Update daily trade count
        date_str = timestamp.strftime('%Y-%m-%d')
        self.daily_trades[date_str] += 1
        
        return {
            'ticker': ticker,
            'timestamp': timestamp,
            'direction': direction,
            'shares': shares,
            'price': execution_price,
            'cost': cost,
            'total': total_cost if direction == 'buy' else total_proceeds
        }
    
    def update_positions(self, trade: Dict):
        """
        Update position tracking after trade execution.
        
        Args:
            trade (Dict): Trade execution details
        """
        if trade is None:
            return
            
        ticker = trade['ticker']
        if ticker not in self.positions:
            self.positions[ticker] = 0
        
        if trade['direction'] == 'buy':
            self.positions[ticker] += trade['shares']
        else:
            self.positions[ticker] -= trade['shares']
            
        # Ensure non-negative positions
        if self.positions[ticker] < 0:
            logger.warning(f"Negative position detected for {ticker}, setting to 0")
            self.positions[ticker] = 0
            
        if self.positions[ticker] == 0:
            del self.positions[ticker]
    
    @property
    def portfolio_value(self) -> float:
        """Calculate current portfolio value including cash and positions."""
        try:
            position_value = 0
            for ticker, shares in self.positions.items():
                if ticker in self.stock_data:
                    # Get the latest price for the ticker
                    latest_price = self.stock_data[ticker]['close'].iloc[-1]
                    position_value += shares * latest_price
                else:
                    logger.warning(f"Missing price data for {ticker}")
            
            return self.cash + position_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {str(e)}")
            return self.cash  # Return cash only in case of error

    def _calculate_position_size(self, ticker: str, signal: float, price: float) -> int:
        """
        Calculate position size based on signal and current price.
        
        Args:
            ticker (str): Stock ticker
            signal (float): Trading signal (-1 to 1)
            price (float): Current price
            
        Returns:
            int: Number of shares to trade
        """
        # Calculate target position value
        target_value = abs(signal) * self.current_capital * MAX_POSITION_SIZE
        
        # Calculate number of shares
        shares = int(target_value / price)
        
        # Apply minimum trade value
        if shares * price < MIN_TRADE_VALUE:
            return 0
        
        # Apply minimum position size
        if shares * price < self.current_capital * MIN_POSITION_SIZE:
            return 0
        
        return shares
    
    def _execute_single_trade(
        self,
        ticker: str,
        signal: float,
        shares: int,
        price: float,
        date: str
    ) -> Optional[Dict]:
        """
        Execute a single trade.
        
        Args:
            ticker (str): Stock ticker
            signal (float): Trading signal (-1 to 1)
            shares (int): Number of shares to trade
            price (float): Current price
            date (str): Trading date
            
        Returns:
            Optional[Dict]: Trade details if executed, None otherwise
        """
        if shares == 0:
            return None
        
        # Calculate total cost including fees and slippage
        total_cost = shares * price * (1 + self.transaction_cost + self.slippage)
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for trade: {ticker}")
            return None
        
        # Create trade record
        trade = {
            'date': date,
            'ticker': ticker,
            'signal': signal,
            'shares': shares if signal > 0 else -shares,
            'price': price,
            'total': total_cost,
            'fees': shares * price * self.transaction_cost,
            'slippage': shares * price * self.slippage
        }
        
        return trade

    def _execute_trades(self, signals: Dict[str, float], date: str) -> List[Dict]:
        """
        Execute trades based on signals.
        
        Args:
            signals (Dict[str, float]): Dictionary of ticker to signal values
            date (str): Trading date
            
        Returns:
            List[Dict]: List of executed trades
        """
        executed_trades = []
        
        for ticker, signal in signals.items():
            try:
                # Verify data exists for this ticker and date
                if ticker not in self.stock_data:
                    logger.warning(f"No data available for {ticker}")
                    continue
                    
                if date not in self.stock_data[ticker].index:
                    logger.warning(f"No data available for {ticker} on {date}")
                    continue
                
                # Get current price
                current_price = self.stock_data[ticker].loc[date, 'close']
                
                # Validate price
                if not self._validate_price(current_price):
                    logger.warning(f"Invalid price {current_price} for {ticker}")
                    continue
                
                # Calculate position size
                position_size = self._calculate_position_size(ticker, signal, current_price)
                
                if position_size == 0:
                    continue
                
                # Check if we have enough cash
                total_cost = position_size * current_price * (1 + self.transaction_cost + self.slippage)
                if total_cost > self.cash:
                    logger.warning(f"Insufficient cash for trade: {ticker} (needed: {total_cost:.2f}, available: {self.cash:.2f})")
                    continue
                
                # Execute trade
                trade = self._execute_single_trade(
                    ticker,
                    signal,
                    position_size,
                    current_price,
                    date
                )
                
                if trade:
                    executed_trades.append(trade)
                    self.trades.append(trade)
                    
                    # Update positions and cash
                    if ticker in self.positions:
                        self.positions[ticker] += trade['shares']
                        # Remove position if it becomes zero
                        if self.positions[ticker] == 0:
                            del self.positions[ticker]
                    else:
                        self.positions[ticker] = trade['shares']
                    
                    self.cash -= trade['total']
                    
                    # Update equity curve after trade
                    self.equity_curve[-1].update({
                        'timestamp': date,
                        'portfolio_value': self.portfolio_value,
                        'cash': self.cash,
                        'positions': self.positions.copy()
                    })
                    
                    logger.info(f"Executed trade for {ticker}: {trade}")
                    
            except Exception as e:
                logger.error(f"Error executing trade for {ticker}: {str(e)}")
                continue
        
        return executed_trades

    def _generate_results(self) -> Dict:
        """
        Generate backtest results and metrics.
        
        Returns:
            Dict: Backtest results and metrics
        """
        try:
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            
            # Ensure timestamp column exists and is properly formatted
            if 'timestamp' not in equity_df.columns:
                logger.error("Missing timestamp column in equity curve")
                return {'error': 'Missing timestamp column in equity curve'}
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(equity_df['timestamp']):
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            # Calculate returns
            equity_df['returns'] = equity_df['portfolio_value'].pct_change()
            
            # Calculate metrics
            total_return = (self.portfolio_value / self.initial_capital) - 1
            annualized_return = (1 + total_return) ** (252 / len(equity_df)) - 1
            
            # Calculate drawdown
            equity_df['cummax'] = equity_df['portfolio_value'].cummax()
            equity_df['drawdown'] = (equity_df['cummax'] - equity_df['portfolio_value']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].max()
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.03  # 3% annual risk-free rate
            excess_returns = equity_df['returns'] - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Calculate win rate
            trades_df = pd.DataFrame(self.trades)
            if not trades_df.empty:
                winning_trades = len(trades_df[trades_df['total'] > 0])
                win_rate = winning_trades / len(trades_df)
            else:
                win_rate = 0.0
            
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.portfolio_value,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'num_trades': len(self.trades),
                'trades': self.trades,
                'equity_curve': equity_df.reset_index().to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error generating results: {str(e)}")
            return {'error': str(e)}
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        strategy_config: Optional[Dict] = None
    ) -> Dict:
        """
        Run backtest for the specified date range.
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            strategy_config (Dict, optional): Strategy configuration
            
        Returns:
            Dict: Backtest results
        """
        try:
            logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Initialize signal generator
            signal_generator = SignalGenerator(self.stock_data)
            
            # Get date range
            dates = pd.date_range(start_date, end_date)
            
            # Track daily trade counts
            daily_trades = {}
            attempted_trades = set()
            
            for date in dates:
                date_str = date.strftime('%Y%m%d')
                daily_trades[date_str] = 0
                
                # Update equity curve at start of day
                self.equity_curve.append({
                    'timestamp': date_str,
                    'portfolio_value': self.portfolio_value,
                    'cash': self.cash,
                    'positions': self.positions.copy()
                })
                
                # Get current data for all stocks up to this date
                current_data = {}
                for ticker, data in self.stock_data.items():
                    if date_str in data.index:
                        # Ensure data has required columns
                        required_columns = ['open', 'high', 'low', 'close', 'volume']
                        if all(col in data.columns for col in required_columns):
                            current_data[ticker] = data.loc[:date_str]
                        else:
                            logger.warning(f"Missing required columns for {ticker}")
                            continue
                
                if not current_data:
                    logger.warning(f"No valid data available for {date_str}")
                    continue
                
                # Calculate factors for each stock separately
                all_factors = []
                for ticker, data in current_data.items():
                    try:
                        # Create a copy of data with required columns
                        factor_data = data[['open', 'high', 'low', 'close', 'volume']].copy()
                        factor_calc = FactorCalculator(factor_data)
                        factors = factor_calc.calculate_factors(factor_data, strategy_config)
                        factors['ticker'] = ticker
                        all_factors.append(factors)
                    except Exception as e:
                        logger.error(f"Error calculating factors for {ticker}: {str(e)}")
                        continue
                
                if not all_factors:
                    logger.warning(f"No factors calculated for {date_str}")
                    continue
                
                # Combine factors
                combined_factors = pd.concat(all_factors, axis=0)
                
                # Generate signals
                signals = signal_generator.generate_signal(combined_factors, strategy_config)
                
                if not signals.empty:
                    # Sort signals by absolute value to prioritize stronger signals
                    sorted_signals = signals[signals != 0].abs().sort_values(ascending=False)
                    
                    # Execute trades based on signals
                    for ticker, signal in sorted_signals.items():
                        if daily_trades[date_str] >= MAX_DAILY_TRADES:
                            logger.warning(f"Daily trade limit reached for {date_str}")
                            break
                            
                        if ticker not in attempted_trades:
                            self._execute_trades({ticker: signal}, date_str)
                            attempted_trades.add(ticker)
                            daily_trades[date_str] += 1
                
                # Reset attempted trades for next day
                attempted_trades.clear()
                
                # Log daily summary
                logger.info(f"Day {date_str} completed: Portfolio value = {self.portfolio_value:.2f}, Cash = {self.cash:.2f}")
            
            # Generate results
            results = self._generate_results()
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise

def main():
    """Run backtest with shadow data."""
    # Initialize universe manager with shadow data
    universe_manager = UniverseManager(
        tushare_token='2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211',
        use_shadow_data=True
    )
    
    # Load universe data
    stock_data = universe_manager.load_universe_data(
        index_name='CSI300',
        start_date='20230101',
        end_date='20231231'
    )
    
    if not stock_data:
        logger.error("No stock data loaded")
        return
    
    # Initialize backtest engine
    engine = BacktestEngine(
        stock_data=stock_data,
        initial_capital=1000000,  # 1 million CNY
        transaction_cost=0.0003,  # 0.03% transaction cost
        slippage=0.0001,  # 0.01% slippage
        execution_delay=1
    )
    
    # Run backtest
    results = engine.run_backtest(
        start_date='20230101',
        end_date='20231231',
        strategy_config={
            'lookback_period': 20,
            'signal_threshold': 0.5,
            'position_size': 0.1
        }
    )
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")

if __name__ == "__main__":
    main() 