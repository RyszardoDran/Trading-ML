"""Main feature engineering function for sequence-based training.

This module provides the central `engineer_candle_features()` function which
orchestrates the computation of all technical and contextual features for
XAU/USD 1-minute data.

Features computed (~50 total):
1. Candle structure (5): Log return, range, body ratio, shadows
2. Volume features (2): Volume change, volume ratio
3. Technical indicators - Trend (6): EMA spread, ADX, +DI/-DI, MACD
4. Technical indicators - Momentum (8): RSI, Stochastic, CCI, Williams %R, ROC
5. Technical indicators - Volatility (5): Volatility, ATR, BB width, BB position
6. Volume indicators (2): OBV, Market structure (HH/LL)
7. Price action (1): Distance from MA
8. Time features (4): Hour/minute encoding
9. M5 Context (5): ATR_M5, RSI_M5, SMA20_M5, MACD_M5, BB_M5
10. Additional micro-structure and long-term features

Usage:
    from ml.src.features import engineer_candle_features
    
    features = engineer_candle_features(df)  # df is OHLCV DataFrame
"""

import logging
import numpy as np
import pandas as pd

from ml.src.features.indicators import (
    compute_rsi, compute_stochastic, compute_cci, compute_williams_r,
    compute_atr, compute_adx, compute_macd, compute_bollinger_bands, 
    compute_obv, compute_roc, compute_volatility, compute_ema
)
from ml.src.features.m5_context import compute_m5_context
from ml.src.features.time_features import compute_time_features

logger = logging.getLogger(__name__)


def engineer_candle_features(df: pd.DataFrame, window_size: int = 100) -> pd.DataFrame:
    """Engineer per-candle features suitable for windowing.

    Features computed for each candle (~50 features total):
    
    **Candle structure (5):**
    - Log return (1-minute)
    - Normalized range: (High - Low) / Close
    - Body ratio: abs(Close - Open) / (High - Low)
    - Upper/lower shadows normalized
    
    **Volume (2):**
    - Volume change: log(Volume / Volume_prev)
    - Volume ratio: Volume / MA(20)
    
    **Trend indicators (6):**
    - EMA spread (12, 26) normalized
    - ADX (Average Directional Index) - trend strength
    - +DI, -DI (Directional Indicators)
    - MACD line and histogram (normalized)
    
    **Momentum indicators (8):**
    - RSI(14) - Relative Strength Index
    - Stochastic K and D (14, 3, 3)
    - CCI (Commodity Channel Index) - specific for commodities like gold
    - Williams %R (14) - momentum oscillator
    - Rate of change 5-min and 20-min
    - Price position in 20-period range [0,1]
    
    **Volatility indicators (5):**
    - Volatility (20-period std)
    - ATR(14) absolute and normalized
    - Bollinger Band width (volatility measure)
    - Bollinger Band position (where price sits in bands)
    
    **Volume indicators (2):**
    - OBV normalized (On-Balance Volume momentum)
    - Market structure (Higher Highs / Lower Lows pattern)
    
    **Price action (1):**
    - Distance from 20-period MA (mean reversion potential)
    
    **Time features (4):**
    - Hour of day (sine/cosine encoding)
    - Minute of hour (sine/cosine encoding)
    
    **M5 Context (5):**
    - ATR_M5, RSI_M5, SMA20_M5, MACD_M5, BB position_M5
    
    **Micro-structure & Long-term:**
    - Multiple context features for trend, momentum, volatility

    Args:
        df: OHLCV DataFrame with datetime index
            Must contain columns: [Date, Open, High, Low, Close, Volume]
        window_size: Minimum lookback for rolling features

    Returns:
        DataFrame with per-candle features (NaNs dropped)

    Raises:
        ValueError: If resulting feature matrix is empty
    """
    logger.info(f"Engineering features for {len(df):,} candles...")
    
    # Use float32 for memory efficiency
    close = df["Close"].astype(np.float32).clip(lower=1e-9)
    open_ = df["Open"].astype(np.float32).clip(lower=1e-9)
    high = df["High"].astype(np.float32).clip(lower=1e-9)
    low = df["Low"].astype(np.float32).clip(lower=1e-9)
    volume = df["Volume"].astype(np.float32).clip(lower=1e-9)

    # ========== M5 Context Features (Resampling) ==========
    m5_features = compute_m5_context(df)

    # ========== Basic Price Features ==========
    logger.info("Calculating basic price features...")
    
    # Log returns
    logc = np.log(close)
    ret1 = logc.diff()
    
    # Candle structure
    logger.info("Calculating candle structure...")
    range_ = high - low
    range_safe = range_.mask(range_ == 0, np.nan)
    
    range_n = range_ / close
    body_ratio = np.abs(close - open_) / (range_safe + 1e-9)
    upper_shadow = (high - np.maximum(open_, close)) / (range_safe + 1e-9)
    lower_shadow = (np.minimum(open_, close) - low) / (range_safe + 1e-9)

    # Volume features
    logger.info("Calculating volume changes...")
    vol_prev = volume.shift(1)
    vol_prev = vol_prev.mask(vol_prev == 0, np.nan)
    vol_change = np.log(volume / vol_prev)

    # ========== Technical Indicators ==========
    logger.info("Calculating technical indicators...")
    
    # EMA spread
    ema_12 = compute_ema(close, 12)
    ema_26 = compute_ema(close, 26)
    ema_spread_n = (ema_12 - ema_26) / close

    # RSI
    rsi_14 = compute_rsi(close, period=14)
    
    # Stochastic Oscillator
    stoch_k, stoch_d = compute_stochastic(high, low, close, period=14)
    
    # CCI (Commodity Channel Index)
    cci = compute_cci(high, low, close, period=20)
    
    # Williams %R
    williams_r = compute_williams_r(high, low, close, period=14)

    # Volatility
    vol_20 = compute_volatility(ret1, period=20)
    
    # ATR (Average True Range)
    atr_14 = compute_atr(high, low, close, period=14)
    atr_n = atr_14 / close
    
    # ADX, +DI, -DI
    adx, plus_di, minus_di = compute_adx(high, low, close, period=14)
    
    # MACD
    macd_line, macd_signal, macd_hist = compute_macd(close, fast=12, slow=26, signal=9)
    macd_line_n = macd_line / close
    macd_hist_n = macd_hist / close
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close, period=20, num_std=2)
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ========== Momentum Indicators ==========
    logger.info("Calculating momentum indicators...")
    
    roc_5 = compute_roc(close, period=5)
    roc_20 = compute_roc(close, period=20)
    
    # Price position in 20-period range
    high_20 = high.rolling(20, min_periods=1).max()
    low_20 = low.rolling(20, min_periods=1).min()
    price_position = (close - low_20) / (high_20 - low_20 + 1e-9)
    
    # Distance from 20-period MA
    close_sma_20 = close.rolling(20, min_periods=1).mean()
    distance_from_ma = (close - close_sma_20) / close_sma_20

    # ========== Volume Indicators ==========
    logger.info("Calculating volume indicators...")
    
    vol_ma_20 = volume.rolling(20, min_periods=1).mean()
    vol_ratio = volume / (vol_ma_20 + 1e-9)
    
    obv = compute_obv(close, volume)
    obv_ma = obv.rolling(20, min_periods=1).mean()
    obv_normalized = (obv - obv_ma) / (obv_ma.abs() + 1e-9)
    
    # Market structure: Higher Highs, Lower Lows
    hh = (high > high.rolling(5, min_periods=1).max().shift(1)).astype(int)
    ll = (low < low.rolling(5, min_periods=1).min().shift(1)).astype(int)
    market_structure = hh - ll

    # ========== Time Features ==========
    time_features = compute_time_features(df, close)

    # ========== Micro-structure Features ==========
    logger.info("Calculating micro-structure features...")
    
    # Micro Volatility
    micro_vol_5 = ret1.rolling(5, min_periods=2).std().fillna(0)
    
    # Efficiency Ratio (Kaufman)
    net_move_5 = (close - close.shift(5)).abs()
    total_path_5 = close.diff().abs().rolling(5, min_periods=1).sum()
    efficiency_5 = net_move_5 / (total_path_5 + 1e-9)
    efficiency_5 = efficiency_5.fillna(0)
    
    # Fractal Dimension Proxy
    sum_range_5 = (high - low).rolling(5, min_periods=1).sum()
    range_5 = (high.rolling(5, min_periods=1).max() - low.rolling(5, min_periods=1).min())
    fractal_dim_5 = sum_range_5 / (range_5 + 1e-9)
    fractal_dim_5 = fractal_dim_5.fillna(1)
    
    # Trend Consistency
    trend_consistency_5 = np.sign(ret1).rolling(5, min_periods=1).sum().fillna(0)
    
    # Trend Slope
    slope_20 = (close - close.shift(20)) / (20 * atr_14 + 1e-9)
    slope_20 = slope_20.fillna(0)
    
    slope_60 = (close - close.shift(60)) / (60 * atr_14 + 1e-9)
    slope_60 = slope_60.fillna(0)

    # ========== Check for NaNs ==========
    z_score_20 = (close - close.rolling(20, min_periods=1).mean()) / (close.rolling(20, min_periods=2).std().fillna(0) + 1e-9)
    z_score_20 = z_score_20.fillna(0)
    
    vol_5 = ret1.rolling(5, min_periods=2).std().fillna(0)
    vol_ratio_5_20 = vol_5 / (vol_20.fillna(0) + 1e-9)
    vol_ratio_5_20 = vol_ratio_5_20.fillna(0)

    # ========== Create Features DataFrame ==========
    logger.info("Creating features DataFrame...")
    
    features = pd.DataFrame(
        {
            # M5 Context Features
            "atr_m5_n": m5_features["atr_m5_n"],
            "rsi_m5": m5_features["rsi_m5"],
            "dist_sma_20_m5": m5_features["dist_sma_20_m5"],
            "macd_n_m5": m5_features["macd_n_m5"],
            "bb_pos_m5": m5_features["bb_pos_m5"],
            
            # Micro-structure
            "micro_vol_5": micro_vol_5,
            "efficiency_5": efficiency_5,
            "fractal_dim_5": fractal_dim_5,
            "trend_consistency_5": trend_consistency_5,
            "slope_20": slope_20,
            "slope_60": slope_60,

            # Long-term features
            "dist_sma_200": time_features["dist_sma_200"],
            "dist_sma_1440": time_features["dist_sma_1440"],
            "roc_60": time_features["roc_60"],
            "vol_ratio_60_200": time_features["vol_ratio_60_200"],
            "dist_day_high": time_features["dist_day_high"],
            "dist_day_low": time_features["dist_day_low"],
            "z_score_20": z_score_20,
            "vol_ratio_5_20": vol_ratio_5_20,
            "dist_prev_high": time_features["dist_prev_high"],
            "dist_prev_low": time_features["dist_prev_low"],
            "dist_prev_close": time_features["dist_prev_close"],
            "dist_daily_open": time_features["dist_daily_open"],
            "dist_london_open": time_features["dist_london_open"],
            
            # Candle structure
            "ret_1": ret1,
            "range_n": range_n,
            "body_ratio": body_ratio,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            
            # Volume
            "vol_change": vol_change,
            "vol_ratio": vol_ratio,
            
            # Technical indicators - Trend
            "ema_spread_n": ema_spread_n,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "macd_line_n": macd_line_n,
            "macd_hist_n": macd_hist_n,
            
            # Technical indicators - Momentum
            "rsi_14": rsi_14,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "cci": cci,
            "williams_r": williams_r,
            "roc_5": roc_5,
            "roc_20": roc_20,
            "price_position": price_position,
            
            # Technical indicators - Volatility
            "vol_20": vol_20,
            "atr_14": atr_14,
            "atr_n": atr_n,
            "bb_width": bb_width,
            "bb_position": bb_position,
            
            # Volume indicators
            "obv_normalized": obv_normalized,
            "market_structure": market_structure,
            
            # Price action
            "distance_from_ma": distance_from_ma,
            
            # Time features
            "hour_sin": time_features["hour_sin"],
            "hour_cos": time_features["hour_cos"],
            "minute_sin": time_features["minute_sin"],
            "minute_cos": time_features["minute_cos"],
        },
        index=df.index,
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Use ffill instead of dropna to avoid losing samples on large datasets
    features = features.ffill().fillna(0)

    if features.empty:
        raise ValueError("Feature matrix is empty after feature engineering")
    
    logger.info(f"Feature engineering complete: {features.shape[0]} rows × {features.shape[1]} features")
    return features
