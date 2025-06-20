"""
Configuration settings for the HFT project.
"""

# API Configuration
TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"

# Trading Parameters
TRADING_HOURS = {
    'start': '09:30:00',
    'end': '15:00:00'
}

# Data Parameters
DATA_DIR = 'data'
HISTORICAL_DATA_DIR = f'{DATA_DIR}/historical'
REALTIME_DATA_DIR = f'{DATA_DIR}/realtime'

# Technical Analysis Parameters
TECHNICAL_INDICATORS = {
    'MA': [5, 10, 20, 60],  # Moving Averages
    'RSI': 14,              # Relative Strength Index
    'MACD': {               # Moving Average Convergence Divergence
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'ROC': 10,              # Rate of Change
    'ADX': 14,              # Average Directional Index
    'ATR': 14,              # Average True Range
    'BB': {                 # Bollinger Bands
        'period': 20,
        'std_dev': 2.0
    },
    'OBV': True,            # On-Balance Volume
    'MFI': 14               # Money Flow Index
}

# Factor Weights for Multi-Factor Strategy
FACTOR_WEIGHTS = {
    'momentum': 0.4,        # Momentum factors (RSI, MACD, ROC)
    'trend': 0.3,           # Trend factors (MA, ADX)
    'volatility': 0.3       # Volatility factors (ATR, BB)
}

# Portfolio Parameters
PORTFOLIO = {
    'top_n': 10,            # Number of stocks to long
    'bottom_n': 10,         # Number of stocks to short
    'transaction_cost': 0.0005,  # 0.05% per trade
    'initial_capital': 1000000   # Initial capital in CNY
}

# Risk Management
RISK_PARAMETERS = {
    'max_position_size': 0.1,  # Maximum position size as fraction of portfolio
    'stop_loss': 0.02,        # Stop loss percentage
    'take_profit': 0.05,      # Take profit percentage
    'max_drawdown': 0.15      # Maximum drawdown threshold
}

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_DIR = 'logs' 