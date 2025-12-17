"""Time-based features for the pipeline.

Provides encoding of temporal information that the model can use:
- Hour of day (sine/cosine encoding for cyclical nature)
- Minute of hour (sine/cosine encoding)
- Daily context (distance from daily open, previous day OHLC)
- London session open context
- Intraday highs/lows

These features help the model understand market microstructure and
intraday patterns without exposing it to absolute time values.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_time_features(df: pd.DataFrame, close: pd.Series) -> dict:
    """Compute time-based features for 1-minute data.
    
    Includes:
    - Hour/minute encoding (sine/cosine for cyclical nature)
    - Daily context (distance from daily open/previous day OHLC)
    - London session reference
    - Intraday high/low context
    - Long-term trend indicators (SMA 200, SMA 1440)
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        close: Close prices (pd.Series)
    
    Returns:
        Dictionary with feature names as keys and Series as values
    """
    logger.info("Computing time-based features...")
    
    hour = df.index.hour
    minute = df.index.minute
    
    # -------- Hour/Minute Encoding (Sine/Cosine) --------
    # Captures cyclical nature: 23:00 and 01:00 should be "close" in value
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)
    
    # -------- Daily Context --------
    # Distance from daily open
    daily_open = df["Open"].groupby(df.index.date).transform("first")
    dist_daily_open = (close - daily_open) / (daily_open + 1e-9)
    
    # -------- London Session Open (08:00 UTC) Context --------
    # Create a series where 08:00 has the open price, others NaN, then ffill within the day
    london_open_mask = (hour == 8) & (minute == 0)
    london_opens = df["Open"].where(london_open_mask).groupby(df.index.date).ffill()
    dist_london_open = (close - london_opens) / (london_opens + 1e-9)
    dist_london_open = dist_london_open.fillna(0)  # 0 if before 08:00
    
    # -------- Previous Day Context --------
    # Resample to daily to get previous day's High/Low/Close
    daily_stats = df.resample("D").agg({"High": "max", "Low": "min", "Close": "last"})
    daily_shifted = daily_stats.shift(1)
    
    # Broadcast back to minute data
    daily_features = daily_shifted.reindex(df.index, method="ffill")
    
    dist_prev_high = (close - daily_features["High"]) / (daily_features["High"] + 1e-9)
    dist_prev_low = (close - daily_features["Low"]) / (daily_features["Low"] + 1e-9)
    dist_prev_close = (close - daily_features["Close"]) / (daily_features["Close"] + 1e-9)
    
    # Fill NaNs
    dist_prev_high = dist_prev_high.fillna(0)
    dist_prev_low = dist_prev_low.fillna(0)
    dist_prev_close = dist_prev_close.fillna(0)
    
    # -------- Intraday High/Low So Far Context --------
    # Group by date and calculate expanding max/min
    day_high_so_far = df["High"].groupby(df.index.date).expanding().max().reset_index(level=0, drop=True)
    day_low_so_far = df["Low"].groupby(df.index.date).expanding().min().reset_index(level=0, drop=True)
    
    # Align indices just in case
    day_high_so_far = day_high_so_far.reindex(df.index)
    day_low_so_far = day_low_so_far.reindex(df.index)
    
    dist_day_high = (close - day_high_so_far) / (day_high_so_far + 1e-9)
    dist_day_low = (close - day_low_so_far) / (day_low_so_far + 1e-9)
    
    # Fill NaNs
    if dist_day_high.isnull().any():
        logger.warning("NaNs in dist_day_high, filling with 0")
        dist_day_high = dist_day_high.fillna(0)
    if dist_day_low.isnull().any():
        logger.warning("NaNs in dist_day_low, filling with 0")
        dist_day_low = dist_day_low.fillna(0)
    
    # -------- Long-term Trend Indicators (Macro Context) --------
    # SMA 200 (larger trend context)
    sma_200 = close.rolling(200, min_periods=1).mean()
    dist_sma_200 = (close - sma_200) / (sma_200 + 1e-9)
    dist_sma_200 = dist_sma_200.fillna(0)
    
    # SMA 1440 (daily SMA on 1-minute data)
    sma_1440 = close.rolling(1440, min_periods=1).mean()
    dist_sma_1440 = (close - sma_1440) / (sma_1440 + 1e-9)
    dist_sma_1440 = dist_sma_1440.fillna(0)
    
    # -------- Long-term Rate of Change --------
    delta_60 = close.diff(60)
    roc_60 = delta_60 / (close.shift(60) + 1e-9)
    roc_60 = roc_60.fillna(0)
    
    # -------- Long-term Volatility Ratio --------
    ret1 = np.log(close).diff()
    vol_60 = ret1.rolling(60, min_periods=1).std()
    vol_200 = ret1.rolling(200, min_periods=1).std()
    vol_ratio_60_200 = vol_60 / (vol_200 + 1e-9)
    vol_ratio_60_200 = vol_ratio_60_200.fillna(0)
    
    return {
        "hour_sin": pd.Series(hour_sin, index=df.index),
        "hour_cos": pd.Series(hour_cos, index=df.index),
        "minute_sin": pd.Series(minute_sin, index=df.index),
        "minute_cos": pd.Series(minute_cos, index=df.index),
        "dist_daily_open": dist_daily_open,
        "dist_london_open": dist_london_open,
        "dist_prev_high": dist_prev_high,
        "dist_prev_low": dist_prev_low,
        "dist_prev_close": dist_prev_close,
        "dist_day_high": dist_day_high,
        "dist_day_low": dist_day_low,
        "dist_sma_200": dist_sma_200,
        "dist_sma_1440": dist_sma_1440,
        "roc_60": roc_60,
        "vol_ratio_60_200": vol_ratio_60_200,
    }
