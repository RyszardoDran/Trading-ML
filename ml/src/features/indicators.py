"""Technical indicators for XAU/USD 1-minute data.

Provides implementations of standard technical indicators:
- Trend indicators: EMA, ADX, MACD
- Momentum indicators: RSI, Stochastic, CCI, Williams %R
- Volatility indicators: ATR, Bollinger Bands
- Volume indicators: OBV
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute Exponential Moving Average.
    
    Args:
        series: Input price series
        span: Period (days/candles)
    
    Returns:
        EMA values aligned with input
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI).
    
    Args:
        close: Close prices
        period: Period for RSI calculation
    
    Returns:
        RSI values (0-100)
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    loss_safe = loss.mask(loss == 0, np.nan)
    rs = gain / loss_safe
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                       period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple:
    """Compute Stochastic Oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
        smooth_k: Smoothing period for K line
        smooth_d: Smoothing period for D line
    
    Returns:
        (stoch_k, stoch_d) tuple
    """
    low_period = low.rolling(period, min_periods=1).min()
    high_period = high.rolling(period, min_periods=1).max()
    
    stoch_k = 100 * (close - low_period) / (high_period - low_period + 1e-9)
    stoch_d = stoch_k.rolling(smooth_d, min_periods=1).mean()
    
    return stoch_k, stoch_d


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Commodity Channel Index (CCI).
    
    Important for commodities like gold. Measures deviation from average price.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
    
    Returns:
        CCI values
    """
    tp = (high + low + close) / 3  # Typical Price
    tp_sma = tp.rolling(period, min_periods=1).mean()
    mad = (tp - tp_sma).abs().rolling(period, min_periods=1).mean()
    cci = (tp - tp_sma) / (0.015 * (mad + 1e-9))
    return cci


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Williams %R momentum oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
    
    Returns:
        Williams %R values (-100 to 0)
    """
    high_period = high.rolling(period, min_periods=1).max()
    low_period = low.rolling(period, min_periods=1).min()
    
    williams_r = -100 * (high_period - close) / (high_period - low_period + 1e-9)
    return williams_r


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR).
    
    Measures volatility. Needed for SL/TP calculation.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (14 standard)
    
    Returns:
        ATR values
    """
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = true_range.rolling(period, min_periods=1).mean()
    return atr


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """Compute Average Directional Index (ADX) with +DI and -DI.
    
    ADX measures trend strength. +DI/-DI show trend direction.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (14 standard)
    
    Returns:
        (adx, plus_di, minus_di) tuple
    """
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movements
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Smoothed values
    tr_smooth = true_range.rolling(period, min_periods=1).mean()
    
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / (tr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / (tr_smooth + 1e-9))
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(period, min_periods=1).mean()
    
    return adx, plus_di, minus_di


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Compute MACD (Moving Average Convergence Divergence).
    
    Args:
        close: Close prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
    
    Returns:
        (macd_line, macd_signal, macd_hist) tuple
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


def compute_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2) -> tuple:
    """Compute Bollinger Bands.
    
    Args:
        close: Close prices
        period: Lookback period
        num_std: Number of standard deviations
    
    Returns:
        (bb_upper, bb_mid, bb_lower) tuple
    """
    bb_mid = close.rolling(period, min_periods=1).mean()
    bb_std = close.rolling(period, min_periods=1).std()
    bb_upper = bb_mid + (num_std * bb_std)
    bb_lower = bb_mid - (num_std * bb_std)
    
    return bb_upper, bb_mid, bb_lower


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV).
    
    Volume momentum indicator. Important for gold and commodities.
    
    Args:
        close: Close prices
        volume: Volume data
    
    Returns:
        OBV values (cumulative)
    """
    # Calculate returns to determine direction
    ret = close.diff()
    obv = (np.sign(ret) * volume).cumsum()
    return obv


def compute_cvd(open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute Cumulative Volume Delta (CVD) approximation.
    
    Uses intra-bar price action to estimate buy/sell volume delta.
    Formula: Delta = Volume * (Close - Open) / (High - Low)
    
    Args:
        open_p: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        
    Returns:
        CVD values (cumulative sum of estimated deltas)
    """
    # Calculate price range
    price_range = high - low
    
    # Estimate delta: (Close - Open) / (High - Low) * Volume
    # If High == Low, delta is 0
    delta = (close - open_p) / (price_range + 1e-9) * volume
    
    # Cumulative sum
    cvd = delta.cumsum()
    return cvd


def compute_roc(close: pd.Series, period: int = 5) -> pd.Series:
    """Compute Rate of Change (ROC).
    
    Measures momentum as percentage change over period.
    
    Args:
        close: Close prices
        period: Lookback period
    
    Returns:
        ROC values (percentage change)
    """
    return close.pct_change(period)


def compute_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    """Compute volatility (standard deviation of returns).
    
    Args:
        returns: Log returns
        period: Lookback period
    
    Returns:
        Volatility values
    """
    return returns.rolling(period, min_periods=1).std()

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV).
    
    OBV accumulates volume with sign based on price direction:
    - Add volume if close > previous close
    - Subtract volume if close < previous close
    
    Args:
        close: Close prices
        volume: Trading volume
    
    Returns:
        OBV values (cumulative)
    """
    obv = pd.Series(0.0, index=close.index)
    obv_val = 0.0
    
    for i in range(len(close)):
        if i == 0:
            obv_val = 0.0
        elif close.iloc[i] > close.iloc[i - 1]:
            obv_val += volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv_val -= volume.iloc[i]
        # If close == close, OBV stays same
        
        obv.iloc[i] = obv_val
    
    return obv


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Compute Money Flow Index (MFI).
    
    MFI combines price action with volume to create a momentum oscillator (0-100).
    Similar to RSI but volume-weighted.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        period: Lookback period (default: 14)
    
    Returns:
        MFI values (0-100)
    """
    # Typical Price = (High + Low + Close) / 3
    tp = (high + low + close) / 3.0
    
    # Raw Money Flow = Typical Price * Volume
    rmf = tp * volume
    
    # Positive/Negative Money Flow
    positive_mf = pd.Series(0.0, index=close.index)
    negative_mf = pd.Series(0.0, index=close.index)
    
    for i in range(1, len(tp)):
        if tp.iloc[i] > tp.iloc[i - 1]:
            positive_mf.iloc[i] = rmf.iloc[i]
        elif tp.iloc[i] < tp.iloc[i - 1]:
            negative_mf.iloc[i] = rmf.iloc[i]
        # If TP == TP, nothing is added (neutral)
    
    # Sum over period
    positive_flow = positive_mf.rolling(period, min_periods=1).sum()
    negative_flow = negative_mf.rolling(period, min_periods=1).sum()
    
    # Money Flow Ratio
    mfr = positive_flow / (negative_flow + 1e-9)
    
    # MFI = 100 - (100 / (1 + MFR))
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    
    return mfi.astype(np.float32)