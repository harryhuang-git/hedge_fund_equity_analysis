"""
Advanced technical indicators and factor calculations for the HFT project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging
from config.config import TECHNICAL_INDICATORS, FACTOR_WEIGHTS

logger = logging.getLogger(__name__)

class FactorWeightOptimizer:
    def __init__(self, window=20, min_weight=0.05):
        self.window = window
        self.min_weight = min_weight
        self.factor_ic_history = {}  # {factor: [IC_t, ...]}
        self.current_weights = {}

    def update_ic(self, factor_df: pd.DataFrame, future_returns: pd.Series):
        # Calculate IC for each factor over the rolling window
        for col in factor_df.columns:
            if col not in self.factor_ic_history:
                self.factor_ic_history[col] = []
            # Rolling IC
            ic = factor_df[col].rolling(self.window).corr(future_returns)
            self.factor_ic_history[col].append(ic.iloc[-1] if not ic.empty else 0)

    def compute_weights(self):
        # Use the mean of recent ICs as the weight basis
        ic_means = {f: np.mean([x for x in ics if not np.isnan(x)]) for f, ics in self.factor_ic_history.items()}
        # Normalize and threshold
        total = sum(abs(v) for v in ic_means.values()) + 1e-8
        weights = {f: max(abs(v) / total, self.min_weight) for f, v in ic_means.items()}
        # Re-normalize to sum to 1
        total = sum(weights.values())
        self.current_weights = {f: w / total for f, w in weights.items()}
        return self.current_weights

    def get_weights(self):
        return self.current_weights

class FactorCalculator:
    def __init__(self, data: pd.DataFrame, weight_optimizer: FactorWeightOptimizer = None):
        """
        Initialize factor calculator with price data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            weight_optimizer (FactorWeightOptimizer, optional): Weight optimizer for dynamic weighting
        """
        self.data = data.copy()
        self._validate_data()
        self.indicators = {}
        self.weight_optimizer = weight_optimizer
        self.dynamic_weights = None
    
    def _validate_data(self):
        """Validate that required columns exist in the data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def calculate_momentum_factors(self) -> pd.DataFrame:
        """Calculate momentum-based factors."""
        factors = pd.DataFrame(index=self.data.index)
        
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=TECHNICAL_INDICATORS['RSI']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=TECHNICAL_INDICATORS['RSI']).mean()
        rs = gain / loss
        factors['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['close'].ewm(span=TECHNICAL_INDICATORS['MACD']['fast'], adjust=False).mean()
        exp2 = self.data['close'].ewm(span=TECHNICAL_INDICATORS['MACD']['slow'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=TECHNICAL_INDICATORS['MACD']['signal'], adjust=False).mean()
        factors['MACD'] = macd
        factors['MACD_Signal'] = signal
        factors['MACD_Hist'] = macd - signal
        
        # Rate of Change (ROC)
        n = TECHNICAL_INDICATORS['ROC']
        factors['ROC'] = self.data['close'].pct_change(periods=n) * 100
        
        return factors
    
    def calculate_trend_factors(self) -> pd.DataFrame:
        """Calculate trend-based factors."""
        factors = pd.DataFrame(index=self.data.index)
        
        # Moving Averages
        for period in TECHNICAL_INDICATORS['MA']:
            factors[f'SMA_{period}'] = self.data['close'].rolling(window=period).mean()
            factors[f'EMA_{period}'] = self.data['close'].ewm(span=period, adjust=False).mean()
        
        # ADX (Average Directional Index)
        period = TECHNICAL_INDICATORS['ADX']
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        factors['ATR'] = true_range.rolling(period).mean()
        
        # Calculate +DM and -DM
        up_move = self.data['high'] - self.data['high'].shift()
        down_move = self.data['low'].shift() - self.data['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate +DI and -DI
        tr = true_range.rolling(period).sum()
        plus_dm_series = pd.Series(plus_dm, index=self.data.index)
        minus_dm_series = pd.Series(minus_dm, index=self.data.index)
        plus_di = 100 * plus_dm_series.rolling(period).sum() / tr
        minus_di = 100 * minus_dm_series.rolling(period).sum() / tr
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        factors['ADX'] = dx.rolling(period).mean()
        
        return factors
    
    def calculate_volatility_factors(self) -> pd.DataFrame:
        """Calculate volatility-based factors."""
        factors = pd.DataFrame(index=self.data.index)
        
        # Bollinger Bands
        period = TECHNICAL_INDICATORS['BB']['period']
        std_dev = TECHNICAL_INDICATORS['BB']['std_dev']
        middle_band = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std()
        factors['BB_Upper'] = middle_band + (std * std_dev)
        factors['BB_Middle'] = middle_band
        factors['BB_Lower'] = middle_band - (std * std_dev)
        factors['BB_Width'] = (factors['BB_Upper'] - factors['BB_Lower']) / factors['BB_Middle']
        
        # ATR
        period = TECHNICAL_INDICATORS['ATR']
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        factors['ATR'] = true_range.rolling(period).mean()
        
        return factors
    
    def calculate_volume_factors(self) -> pd.DataFrame:
        """Calculate volume-based factors."""
        factors = pd.DataFrame(index=self.data.index)
        
        # On-Balance Volume (OBV)
        if TECHNICAL_INDICATORS['OBV']:
            obv = pd.Series(index=self.data.index, dtype='float64')
            obv.iloc[0] = 0.0
            for i in range(1, len(self.data)):
                if self.data['close'].iloc[i] > self.data['close'].iloc[i-1]:
                    obv.iloc[i] = float(obv.iloc[i-1]) + float(self.data['volume'].iloc[i])
                elif self.data['close'].iloc[i] < self.data['close'].iloc[i-1]:
                    obv.iloc[i] = float(obv.iloc[i-1]) - float(self.data['volume'].iloc[i])
                else:
                    obv.iloc[i] = float(obv.iloc[i-1])
            factors['OBV'] = obv
        
        # Money Flow Index (MFI)
        period = TECHNICAL_INDICATORS['MFI']
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        money_flow = typical_price * self.data['volume']
        
        positive_flow = pd.Series(0.0, index=self.data.index, dtype='float64')
        negative_flow = pd.Series(0.0, index=self.data.index, dtype='float64')
        
        for i in range(1, len(self.data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = float(money_flow.iloc[i])
            else:
                negative_flow.iloc[i] = float(money_flow.iloc[i])
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        factors['MFI'] = mfi
        
        return factors
    
    def calculate_hft_factors(self) -> pd.DataFrame:
        """Calculate high-frequency and microstructure factors."""
        factors = pd.DataFrame(index=self.data.index)
        # Trend Strength: (close - open) / sum(abs(price_t - price_{t-1}))
        net_move = self.data['close'] - self.data['open']
        total_path = self.data['close'].diff().abs().rolling(window=len(self.data)).sum()
        factors['trend_strength'] = net_move / total_path.replace(0, np.nan)
        # Improved Reversal: exclude overnight gap and first 30min (assume minute data, 30min=30 rows)
        if len(self.data) > 40:
            open_px = self.data['open'].iloc[0]
            close_px = self.data['close'].iloc[-1]
            after_30min = self.data.iloc[30:]
            reversal = (close_px - after_30min['close'].iloc[0]) / after_30min['close'].iloc[0]
            factors['improved_reversal'] = reversal
        else:
            factors['improved_reversal'] = 0
        # Intraday Skewness/Kurtosis
        returns = self.data['close'].pct_change().dropna()
        factors['intraday_skew'] = returns.rolling(window=len(returns)).apply(lambda x: pd.Series(x).skew(), raw=False)
        factors['intraday_kurt'] = returns.rolling(window=len(returns)).apply(lambda x: pd.Series(x).kurt(), raw=False)
        # Volume Profile: 10:00-11:00 and last 30min volume ratio (assume index is time or integer)
        n = len(self.data)
        if n > 60:
            factors['vol_10_11_ratio'] = self.data['volume'].iloc[30:60].sum() / self.data['volume'].sum()
            factors['vol_last30_ratio'] = self.data['volume'].iloc[-30:].sum() / self.data['volume'].sum()
        else:
            factors['vol_10_11_ratio'] = 0
            factors['vol_last30_ratio'] = 0
        # Price-Volume Correlation
        if self.data['volume'].std() > 0:
            factors['price_vol_corr'] = self.data['close'].rolling(window=len(self.data)).corr(self.data['volume'])
        else:
            factors['price_vol_corr'] = 0
        # Order Flow Proxy: volume imbalance (if bid/ask not available)
        if 'buy_volume' in self.data.columns and 'sell_volume' in self.data.columns:
            total = self.data['buy_volume'] + self.data['sell_volume']
            factors['order_imbalance'] = (self.data['buy_volume'] - self.data['sell_volume']) / total.replace(0, np.nan)
        else:
            # Use up-tick vs down-tick volume as a proxy
            up = self.data['close'].diff() > 0
            down = self.data['close'].diff() < 0
            up_vol = self.data['volume'][up].sum()
            down_vol = self.data['volume'][down].sum()
            total = up_vol + down_vol
            factors['order_imbalance'] = (up_vol - down_vol) / total if total != 0 else 0
        return factors
    
    def normalize_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize factors using z-score method.
        
        Args:
            factors (pd.DataFrame): Raw factor values
            
        Returns:
            pd.DataFrame: Normalized factor values
        """
        normalized = pd.DataFrame(index=factors.index, columns=factors.columns)
        for col in factors.columns:
            std = factors[col].std()
            if std == 0:
                normalized[col] = 0
            else:
                normalized[col] = (factors[col] - factors[col].mean()) / std
        return normalized
    
    def calculate_combined_score(self, future_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate combined factor score using weighted sum of normalized factors.
        
        Args:
            future_returns (pd.Series, optional): Future returns for dynamic weighting
            
        Returns:
            pd.DataFrame: Combined factor scores
        """
        # Calculate all factor groups
        momentum = self.calculate_momentum_factors()
        trend = self.calculate_trend_factors()
        volatility = self.calculate_volatility_factors()
        volume = self.calculate_volume_factors()
        hft = self.calculate_hft_factors()
        
        # Normalize all factors
        momentum_norm = self.normalize_factors(momentum)
        trend_norm = self.normalize_factors(trend)
        volatility_norm = self.normalize_factors(volatility)
        volume_norm = self.normalize_factors(volume)
        hft_norm = self.normalize_factors(hft)
        
        # Combine all normalized factors
        all_factors = pd.concat([
            momentum_norm.add_prefix('momentum_'),
            trend_norm.add_prefix('trend_'),
            volatility_norm.add_prefix('volatility_'),
            volume_norm.add_prefix('volume_'),
            hft_norm.add_prefix('hft_')
        ], axis=1)
        
        # Dynamic weighting
        if self.weight_optimizer is not None and future_returns is not None:
            self.weight_optimizer.update_ic(all_factors, future_returns)
            weights = self.weight_optimizer.compute_weights()
            self.dynamic_weights = weights
            # Weighted sum
            combined_score = all_factors.mul(pd.Series(weights)).sum(axis=1)
        else:
            # Static weights fallback
            combined_score = (
                FACTOR_WEIGHTS['momentum'] * momentum_norm.mean(axis=1) +
                FACTOR_WEIGHTS['trend'] * trend_norm.mean(axis=1) +
                FACTOR_WEIGHTS['volatility'] * volatility_norm.mean(axis=1) +
                FACTOR_WEIGHTS['volume'] * volume_norm.mean(axis=1) +
                FACTOR_WEIGHTS['hft'] * hft_norm.mean(axis=1)
            )
        
        return pd.DataFrame({'combined_score': combined_score})

    def get_dynamic_weights(self):
        return self.dynamic_weights

    def calculate_factors(self, data: pd.DataFrame, config: Optional[Dict] = None, future_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate all factors for the given data.
        
        Args:
            data (pd.DataFrame): Price data
            config (Dict, optional): Strategy configuration
            future_returns (pd.Series, optional): Future returns for dynamic weighting
            
        Returns:
            pd.DataFrame: Combined factor scores
        """
        # Update data if provided
        if data is not None:
            self.data = data.copy()
            self._validate_data()
        
        # Calculate all factor groups
        momentum = self.calculate_momentum_factors()
        trend = self.calculate_trend_factors()
        volatility = self.calculate_volatility_factors()
        volume = self.calculate_volume_factors()
        hft = self.calculate_hft_factors()
        
        # Combine all factors
        all_factors = pd.concat([
            momentum.add_prefix('momentum_'),
            trend.add_prefix('trend_'),
            volatility.add_prefix('volatility_'),
            volume.add_prefix('volume_'),
            hft.add_prefix('hft_')
        ], axis=1)
        
        # Calculate combined score with dynamic weights if available
        combined_score = self.calculate_combined_score(future_returns)
        all_factors['combined_score'] = combined_score['combined_score']
        
        # Handle NaN values
        all_factors = all_factors.fillna(0)
        
        return all_factors

if __name__ == "__main__":
    # Example usage
    from src.data.market_data import MarketDataCollector
    
    # Get some sample data
    collector = MarketDataCollector()
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime('%Y%m%d')
    
    data = collector.get_daily_data('000001.SZ', start_date, end_date)
    if data is not None:
        # Calculate factors
        factor_calc = FactorCalculator(data)
        factors = factor_calc.calculate_factors(data)
        print(factors.head()) 