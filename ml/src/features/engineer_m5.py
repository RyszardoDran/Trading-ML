#!/usr/bin/env python3
"""M5 (5-minute) feature engineering for trading strategy.

**PURPOSE**: Generate features DIRECTLY on M5 timeframe instead of M1.
This matches the trading strategy which operates on 5-minute candles.

**KEY DIFFERENCE FROM engineer.py**:
- engineer.py: M1 candles with M5 indicators forward-filled
- engineer_m5.py: M5 candles with M5 indicators (THIS FILE)

**PROCESS**:
1. Load M1 raw data (e.g., 7 days = ~10,080 M1 candles)
2. Aggregate to M5 (10,080 / 5 = ~2,016 M5 candles)
3. Calculate features on M5 bars
4. Return M5 DataFrame (NOT M1!)

**USAGE**:
    from ml.src.features.engineer_m5 import engineer_m5_candle_features
    
    # Load 7 days of M1 data
    df_m1 = load_data(days=7)  # ~10,080 M1 candles
    
    # Engineer features on M5
    features_m5 = engineer_m5_candle_features(df_m1)  # ~2,016 M5 candles
    
    # Create sequences from M5 features
    X, y = create_sequences(features_m5, window_size=100)  # 100 M5 candles

**INPUTS**:
    - DataFrame with DatetimeIndex and columns: [Open, High, Low, Close, Volume]
    - M1 frequency (will be aggregated to M5)
    
**OUTPUTS**:
    - DataFrame with M5 features (1 row per 5-minute bar)
    - Includes: RSI, BB position, SMA distance, MACD, Stochastic, ATR, etc.
"""

import logging
import numpy as np
import pandas as pd

from .indicators import (
    compute_rsi, compute_stochastic, compute_macd, 
    compute_adx, compute_bollinger_bands, compute_atr,
    compute_cvd, compute_obv, compute_mfi
)
from ..utils.risk_config import (
    CVD_LOOKBACK_WINDOW, ENABLE_CVD_INDICATOR,
    FEAT_ENABLE_RSI, FEAT_ENABLE_BB_POS, FEAT_ENABLE_SMA_DIST,
    FEAT_ENABLE_STOCH, FEAT_ENABLE_MACD, FEAT_ENABLE_ATR,
    FEAT_ENABLE_ADX, FEAT_ENABLE_SMA200, FEAT_ENABLE_RETURNS,
    FEAT_ENABLE_VOLUME_M5, FEAT_ENABLE_VOLUME_M15,
    FEAT_ENABLE_OBV, FEAT_ENABLE_MFI,
    FEAT_ENABLE_M15_CONTEXT, FEAT_ENABLE_M60_CONTEXT
)

logger = logging.getLogger(__name__)


def aggregate_to_m5(
    df_m1: pd.DataFrame,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """Aggregate M1 OHLCV data to M5 (5-minute) bars, with optional date filtering to prevent data leakage.
    
    Args:
        df_m1: M1 DataFrame with DatetimeIndex and [Open, High, Low, Close, Volume]
        start_date: Optional, filter data from this date (inclusive)
        end_date: Optional, filter data up to this date (inclusive)
    Returns:
        M5 DataFrame with aggregated OHLCV data
    Raises:
        ValueError: If required columns are missing
    """
    # Filter by date if specified
    if start_date is not None:
        df_m1 = df_m1[df_m1.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_m1 = df_m1[df_m1.index <= pd.to_datetime(end_date)]

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = set(required_cols) - set(df_m1.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Aggregating {len(df_m1)} M1 candles to M5...")
    
    # Aggregation dictionary
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Resample to 5-minute bars
    df_m5 = df_m1[required_cols].resample('5min').agg(agg_dict)
    
    # Drop incomplete bars (last bar may be incomplete)
    df_m5 = df_m5.dropna()
    
    logger.info(f"Aggregated to {len(df_m5)} M5 candles ({len(df_m1)/len(df_m5):.1f}x compression)")
    
    return df_m5


def engineer_m5_candle_features(df_m1: pd.DataFrame) -> pd.DataFrame:
    """Engineer features directly on M5 (5-minute) candles.
    
    **PURPOSE**: Generate trading features on M5 timeframe to match
    strategy that operates on 5-minute candles.
    
    **PROCESS**:
    1. Aggregate M1 → M5 (5-minute bars)
    2. Calculate technical indicators on M5
    3. Calculate multi-timeframe context (M15, M60) from M5
    4. Return M5 DataFrame with features
    
    **FEATURES** (~15 total):
    - M5 Primary: RSI, BB position, SMA distance, Stochastic, MACD, ATR
    - M15 Context: RSI, BB position, SMA distance (from M5)
    - M60 Context: RSI, BB position (from M5)
    - Returns and volatility
    
    Args:
        df_m1: M1 DataFrame with [Open, High, Low, Close, Volume]
        
    Returns:
        M5 DataFrame with engineered features (1 row per 5-minute bar)
        
    Raises:
        ValueError: If input data is insufficient or invalid
        
    Notes:
        - Requires sufficient M1 data for indicator warmup (recommend 7 days)
        - Output is ~1/5 size of input (10,080 M1 → 2,016 M5)
        - All features calculated on M5 timeframe (no forward-fill from M1)
        - Multi-timeframe indicators (M15, M60) resampled from M5
        
    Examples:
        >>> # Load 7 days of M1 data
        >>> df_m1 = load_data(days=7)
        >>> len(df_m1)
        10080
        
        >>> # Engineer M5 features
        >>> features_m5 = engineer_m5_candle_features(df_m1)
        >>> len(features_m5)
        2016  # ~10,080 / 5
        
        >>> # Create 100-candle sequences (100 × 5min = 500 minutes)
        >>> X, y = create_sequences(features_m5, window_size=100)
    """
    logger.info("Engineering M5 features...")
    
    # Step 1: Aggregate M1 → M5
    df_m5 = aggregate_to_m5(df_m1)
    
    if len(df_m5) < 200:
        raise ValueError(f"Insufficient M5 data: need at least 200 bars, got {len(df_m5)}")
    
    # Extract M5 price series
    close = df_m5["Close"].astype(np.float32).clip(lower=1e-9)
    high = df_m5["High"].astype(np.float32).clip(lower=1e-9)
    low = df_m5["Low"].astype(np.float32).clip(lower=1e-9)
    
    # ========== M5 Primary Features ==========
    logger.info("Computing M5 primary features...")
    
    # ATR (14-period on M5 = 70 minutes)
    atr_14 = compute_atr(high, low, close, period=14)
    atr_norm = atr_14 / atr_14.rolling(14, min_periods=1).mean()
    
    # RSI (14-period on M5)
    rsi_14 = compute_rsi(close, period=14)
    
    # Bollinger Bands (20-period on M5 = 100 minutes)
    bb_upper, bb_sma, bb_lower = compute_bollinger_bands(close, period=20, num_std=2)
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # SMA distance (20-period on M5)
    sma_20 = close.rolling(20, min_periods=1).mean()
    dist_sma_20 = (close - sma_20) / (atr_14 + 1e-9)
    
    # Stochastic (14-period on M5)
    stoch_k, stoch_d = compute_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3)
    
    # MACD on M5
    macd_line, macd_signal, macd_hist = compute_macd(close, fast=12, slow=26, signal=9)
    macd_hist_norm = macd_hist / (atr_14 + 1e-9)
    
    # ADX (14-period on M5) - returns tuple (adx, plus_di, minus_di)
    adx, _, _ = compute_adx(high, low, close, period=14)
    
    # Volume Analysis M5
    volume_m5 = df_m5["Volume"].astype(np.float32)
    volume_m5_norm = volume_m5 / (volume_m5.rolling(20).mean() + 1e-9)
    
    # CVD (Cumulative Volume Delta) on M5
    if ENABLE_CVD_INDICATOR:
        open_p = df_m5["Open"].astype(np.float32)
        volume = df_m5["Volume"].astype(np.float32)
        cvd = compute_cvd(open_p, high, low, close, volume)
        # Normalize CVD (z-score over rolling window to make it stationary for ML)
        cvd_rolling_mean = cvd.rolling(CVD_LOOKBACK_WINDOW, min_periods=1).mean()
        cvd_rolling_std = cvd.rolling(CVD_LOOKBACK_WINDOW, min_periods=1).std()
        cvd_norm = (cvd - cvd_rolling_mean) / (cvd_rolling_std + 1e-9)
    else:
        cvd_norm = pd.Series(0.0, index=df_m5.index)
    
    # SMA 200 on M5 (200-period on M5 = 1000 minutes = 16.7 hours)
    sma_200 = close.rolling(200, min_periods=1).mean()
    dist_sma_200_m5 = (close - sma_200) / (atr_14 + 1e-9)
    
    # No M15 alignment here (handled later after M15 is computed)
    # Continue with M5-derived short-term features
    ret_1 = close.pct_change().fillna(0)
    ret_1 = close.pct_change().fillna(0)
    
    # OBV (On-Balance Volume) on M5
    if FEAT_ENABLE_OBV:
        obv_m5 = compute_obv(close, volume)
        obv_m5_norm = (obv_m5 - obv_m5.rolling(50, min_periods=1).mean()) / (obv_m5.rolling(50, min_periods=1).std() + 1e-9)
    else:
        obv_m5_norm = pd.Series(0.0, index=df_m5.index)
    
    # MFI (Money Flow Index) on M5
    if FEAT_ENABLE_MFI:
        mfi_m5 = compute_mfi(high, low, close, volume, period=14)
        mfi_m5_norm = (mfi_m5 - 50.0) / 25.0  # Normalize 0-100 range to approximately -2..2
    else:
        mfi_m5_norm = pd.Series(0.0, index=df_m5.index)
    
    # ========== M15 Context (from M5) ==========
    logger.info("Computing M15 context from M5...")
    
    # Resample M5 → M15 (every 3 M5 bars = 1 M15 bar)
    df_m15 = df_m5.resample('15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    close_m15 = df_m15["Close"].astype(np.float32).clip(lower=1e-9)
    high_m15 = df_m15["High"].astype(np.float32).clip(lower=1e-9)
    low_m15 = df_m15["Low"].astype(np.float32).clip(lower=1e-9)
    
    # M15 indicators
    atr_m15 = compute_atr(high_m15, low_m15, close_m15, period=14)
    rsi_m15 = compute_rsi(close_m15, period=14)
    bb_upper_m15, bb_sma_m15, bb_lower_m15 = compute_bollinger_bands(close_m15, period=20, num_std=2)
    bb_pos_m15 = (close_m15 - bb_lower_m15) / (bb_upper_m15 - bb_lower_m15 + 1e-9)
    sma_20_m15 = close_m15.rolling(20, min_periods=1).mean()
    dist_sma_20_m15 = (close_m15 - sma_20_m15) / (atr_m15 + 1e-9)
    
    # Volume Analysis M15
    volume_m15 = df_m15["Volume"].astype(np.float32)
    volume_m15_norm = volume_m15 / (volume_m15.rolling(20).mean() + 1e-9)
    
    # CVD (Cumulative Volume Delta) on M15
    if ENABLE_CVD_INDICATOR:
        open_m15 = df_m15["Open"].astype(np.float32)
        cvd_m15 = compute_cvd(open_m15, high_m15, low_m15, close_m15, volume_m15)
        cvd_m15_rolling_mean = cvd_m15.rolling(CVD_LOOKBACK_WINDOW, min_periods=1).mean()
        cvd_m15_rolling_std = cvd_m15.rolling(CVD_LOOKBACK_WINDOW, min_periods=1).std()
        cvd_m15_norm = (cvd_m15 - cvd_m15_rolling_mean) / (cvd_m15_rolling_std + 1e-9)
    else:
        cvd_m15_norm = pd.Series(0.0, index=df_m15.index)
    
    # OBV (On-Balance Volume) on M15
    if FEAT_ENABLE_OBV:
        obv_m15 = compute_obv(close_m15, volume_m15)
        obv_m15_norm = (obv_m15 - obv_m15.rolling(50, min_periods=1).mean()) / (obv_m15.rolling(50, min_periods=1).std() + 1e-9)
    else:
        obv_m15_norm = pd.Series(0.0, index=df_m15.index)
    
    # MFI (Money Flow Index) on M15
    if FEAT_ENABLE_MFI:
        mfi_m15 = compute_mfi(high_m15, low_m15, close_m15, volume_m15, period=14)
        mfi_m15_norm = (mfi_m15 - 50.0) / 25.0
    else:
        mfi_m15_norm = pd.Series(0.0, index=df_m15.index)
    
    # SMA 200 on M15 (200-period on M15 = 3000 minutes = 50 hours) - longer trend context
    sma_200_m15 = close_m15.rolling(200, min_periods=1).mean()
    dist_sma_200_m15 = (close_m15 - sma_200_m15) / (atr_m15 + 1e-9)
    
    # Align M15 to M5 index (backward-fill to avoid lookahead)
    # Use bfill instead of ffill: at time T, use PREVIOUS M15 bar that closed BEFORE T
    # Use forward-fill (ffill) to align higher-timeframe bars to the current M5 timestamp
    # At time T we want the last closed M15/M60 bar that closed BEFORE T (ffill),
    # not the next bar in the future (bfill) which introduces lookahead.
    try:
        rsi_m15 = rsi_m15.reindex(df_m5.index, method='ffill').fillna(50)
    except Exception:
        rsi_m15 = pd.Series(50.0, index=df_m5.index)
    try:
        bb_pos_m15 = bb_pos_m15.reindex(df_m5.index, method='ffill').fillna(0.5)
    except Exception:
        bb_pos_m15 = pd.Series(0.5, index=df_m5.index)
    try:
        dist_sma_20_m15 = dist_sma_20_m15.reindex(df_m5.index, method='ffill').fillna(0)
    except Exception:
        dist_sma_20_m15 = pd.Series(0.0, index=df_m5.index)
    try:
        dist_sma_200 = dist_sma_200_m15.reindex(df_m5.index, method='ffill').fillna(0)
    except Exception:
        dist_sma_200 = pd.Series(0.0, index=df_m5.index)
    try:
        volume_m15_norm = volume_m15_norm.reindex(df_m5.index, method='ffill').fillna(1.0)
    except Exception:
        volume_m15_norm = pd.Series(1.0, index=df_m5.index)
    try:
        cvd_m15_norm = cvd_m15_norm.reindex(df_m5.index, method='ffill').fillna(0)
    except Exception:
        cvd_m15_norm = pd.Series(0.0, index=df_m15.index)
    
    # ========== M60 Context (from M5) ==========
    logger.info("Computing M60 context from M5...")
    
    # Resample M5 → M60 (every 12 M5 bars = 1 M60 bar)
    df_m60 = df_m5.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    close_m60 = df_m60["Close"].astype(np.float32).clip(lower=1e-9)
    high_m60 = df_m60["High"].astype(np.float32).clip(lower=1e-9)
    low_m60 = df_m60["Low"].astype(np.float32).clip(lower=1e-9)
    
    # M60 indicators
    rsi_m60 = compute_rsi(close_m60, period=14)
    bb_upper_m60, bb_sma_m60, bb_lower_m60 = compute_bollinger_bands(close_m60, period=20, num_std=2)
    bb_pos_m60 = (close_m60 - bb_lower_m60) / (bb_upper_m60 - bb_lower_m60 + 1e-9)
    
    # CVD (Cumulative Volume Delta) on M60
    if ENABLE_CVD_INDICATOR:
        open_m60 = df_m60["Open"].astype(np.float32)
        volume_m60 = df_m60["Volume"].astype(np.float32)
        cvd_m60 = compute_cvd(open_m60, high_m60, low_m60, close_m60, volume_m60)
        cvd_m60_rolling_mean = cvd_m60.rolling(CVD_LOOKBACK_WINDOW, min_periods=1).mean()
        cvd_m60_rolling_std = cvd_m60.rolling(CVD_LOOKBACK_WINDOW, min_periods=1).std()
        cvd_m60_norm = (cvd_m60 - cvd_m60_rolling_mean) / (cvd_m60_rolling_std + 1e-9)
    else:
        cvd_m60_norm = pd.Series(0.0, index=df_m60.index)
    
    # OBV (On-Balance Volume) on M60
    if FEAT_ENABLE_OBV:
        obv_m60 = compute_obv(close_m60, volume_m60)
        obv_m60_norm = (obv_m60 - obv_m60.rolling(50, min_periods=1).mean()) / (obv_m60.rolling(50, min_periods=1).std() + 1e-9)
    else:
        obv_m60_norm = pd.Series(0.0, index=df_m60.index)
    
    # MFI (Money Flow Index) on M60
    if FEAT_ENABLE_MFI:
        mfi_m60 = compute_mfi(high_m60, low_m60, close_m60, volume_m60, period=14)
        mfi_m60_norm = (mfi_m60 - 50.0) / 25.0
    else:
        mfi_m60_norm = pd.Series(0.0, index=df_m60.index)
    
    # Align M60 to M5 index using forward-fill to avoid lookahead
    # Use ffill so that at time T we use the most recent M60 bar that closed before T
    rsi_m60 = rsi_m60.reindex(df_m5.index, method='ffill').fillna(50)
    bb_pos_m60 = bb_pos_m60.reindex(df_m5.index, method='ffill').fillna(0.5)
    cvd_m60_norm = cvd_m60_norm.reindex(df_m5.index, method='ffill').fillna(0)
    obv_m60_norm = obv_m60_norm.reindex(df_m5.index, method='ffill').fillna(0)
    mfi_m60_norm = mfi_m60_norm.reindex(df_m5.index, method='ffill').fillna(0)
    
    # ========== Create M5 Features DataFrame ==========
    features_dict = {}
    
    # M5 Primary (calculated on M5 bars)
    if FEAT_ENABLE_RSI:
        features_dict["rsi_m5"] = rsi_14.fillna(50)
    else:
        features_dict["rsi_m5"] = pd.Series(50.0, index=df_m5.index)
        
    if FEAT_ENABLE_BB_POS:
        features_dict["bb_pos_m5"] = bb_position.fillna(0.5)
    else:
        features_dict["bb_pos_m5"] = pd.Series(0.5, index=df_m5.index)
        
    if FEAT_ENABLE_SMA_DIST:
        features_dict["dist_sma_20_m5"] = dist_sma_20.fillna(0)
    else:
        features_dict["dist_sma_20_m5"] = pd.Series(0.0, index=df_m5.index)
        
    if FEAT_ENABLE_STOCH:
        features_dict["stoch_k_m5"] = stoch_k.fillna(50)
        features_dict["stoch_d_m5"] = stoch_d.fillna(50)
    else:
        features_dict["stoch_k_m5"] = pd.Series(50.0, index=df_m5.index)
        features_dict["stoch_d_m5"] = pd.Series(50.0, index=df_m5.index)
        
    if FEAT_ENABLE_MACD:
        features_dict["macd_hist_m5"] = macd_hist_norm.fillna(0)
    else:
        features_dict["macd_hist_m5"] = pd.Series(0.0, index=df_m5.index)
        
    if FEAT_ENABLE_ATR:
        features_dict["atr_norm_m5"] = atr_norm.fillna(1.0)
        # Also expose raw ATR (absolute units) required by regime filters and target logic
        features_dict["atr_m5"] = atr_14.fillna(0.0)
    else:
        features_dict["atr_norm_m5"] = pd.Series(1.0, index=df_m5.index)
        features_dict["atr_m5"] = pd.Series(0.0, index=df_m5.index)
    
    if FEAT_ENABLE_ADX:
        features_dict["adx"] = adx.fillna(20)
    else:
        features_dict["adx"] = pd.Series(20.0, index=df_m5.index)

    # Provide raw SMA200 level as well (regime filter expects 'sma_200')
    features_dict["sma_200"] = sma_200.ffill().fillna(sma_200.mean() if not sma_200.empty else 0.0)

    if ENABLE_CVD_INDICATOR:
        features_dict["cvd_m5"] = cvd_norm.fillna(0)
    else:
        features_dict["cvd_m5"] = pd.Series(0.0, index=df_m5.index)
        
    if FEAT_ENABLE_SMA200:
        # Keep original M5 SMA200 distance for backwards compatibility
        features_dict["dist_sma_200_m5"] = dist_sma_200_m5.fillna(0)
        # Add M15 SMA200 distance as a separate feature for model to learn longer-term trend
        features_dict["dist_sma_200_m15"] = dist_sma_200_m15.reindex(df_m5.index, method='ffill').fillna(0)
        # Preserve legacy 'dist_sma_200' name as the M5 version (used by filters by default)
        features_dict["dist_sma_200"] = dist_sma_200_m5.fillna(0)
    else:
        features_dict["dist_sma_200_m5"] = pd.Series(0.0, index=df_m5.index)
        features_dict["dist_sma_200_m15"] = pd.Series(0.0, index=df_m5.index)
        features_dict["dist_sma_200"] = pd.Series(0.0, index=df_m5.index)
        
    if FEAT_ENABLE_RETURNS:
        features_dict["ret_1_m5"] = ret_1.fillna(0)
    else:
        features_dict["ret_1_m5"] = pd.Series(0.0, index=df_m5.index)
        
    if FEAT_ENABLE_OBV:
        features_dict["obv_m5"] = obv_m5_norm.fillna(0)
    else:
        features_dict["obv_m5"] = pd.Series(0.0, index=df_m5.index)
        
    if FEAT_ENABLE_MFI:
        features_dict["mfi_m5"] = mfi_m5_norm.fillna(0)
    else:
        features_dict["mfi_m5"] = pd.Series(0.0, index=df_m5.index)
        
    if FEAT_ENABLE_VOLUME_M5:
        features_dict["volume_m5_norm"] = volume_m5_norm.fillna(1.0)
    else:
        features_dict["volume_m5_norm"] = pd.Series(1.0, index=df_m5.index)
        
    # M15 Context
    if FEAT_ENABLE_M15_CONTEXT:
        features_dict["rsi_m15"] = rsi_m15
        features_dict["bb_pos_m15"] = bb_pos_m15
        features_dict["dist_sma_20_m15"] = dist_sma_20_m15
        if ENABLE_CVD_INDICATOR:
            features_dict["cvd_m15"] = cvd_m15_norm
        if FEAT_ENABLE_OBV:
            features_dict["obv_m15"] = obv_m15_norm
        if FEAT_ENABLE_MFI:
            features_dict["mfi_m15"] = mfi_m15_norm
    else:
        features_dict["rsi_m15"] = pd.Series(50.0, index=df_m5.index)
        features_dict["bb_pos_m15"] = pd.Series(0.5, index=df_m5.index)
        features_dict["dist_sma_20_m15"] = pd.Series(0.0, index=df_m5.index)
        if ENABLE_CVD_INDICATOR:
            features_dict["cvd_m15"] = pd.Series(0.0, index=df_m5.index)
        if FEAT_ENABLE_OBV:
            features_dict["obv_m15"] = pd.Series(0.0, index=df_m5.index)
        if FEAT_ENABLE_MFI:
            features_dict["mfi_m15"] = pd.Series(0.0, index=df_m5.index)
        
    if FEAT_ENABLE_VOLUME_M15:
        features_dict["volume_m15_norm"] = volume_m15_norm
    else:
        features_dict["volume_m15_norm"] = pd.Series(1.0, index=df_m5.index)
        
    # M60 Context
    if FEAT_ENABLE_M60_CONTEXT:
        features_dict["rsi_m60"] = rsi_m60
        features_dict["bb_pos_m60"] = bb_pos_m60
        if ENABLE_CVD_INDICATOR:
            features_dict["cvd_m60"] = cvd_m60_norm
        if FEAT_ENABLE_OBV:
            features_dict["obv_m60"] = obv_m60_norm
        if FEAT_ENABLE_MFI:
            features_dict["mfi_m60"] = mfi_m60_norm
    else:
        features_dict["rsi_m60"] = pd.Series(50.0, index=df_m5.index)
        features_dict["bb_pos_m60"] = pd.Series(0.5, index=df_m5.index)
        if ENABLE_CVD_INDICATOR:
            features_dict["cvd_m60"] = pd.Series(0.0, index=df_m5.index)
        if FEAT_ENABLE_OBV:
            features_dict["obv_m60"] = pd.Series(0.0, index=df_m5.index)
        if FEAT_ENABLE_MFI:
            features_dict["mfi_m60"] = pd.Series(0.0, index=df_m5.index)

    features_m5 = pd.DataFrame(features_dict, index=df_m5.index)
    
    # Clean NaN and inf values
    features_m5.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Use forward-fill only to propagate historical values; drop initial warmup rows that still contain NaN
    features_m5 = features_m5.ffill()
    features_m5 = features_m5.dropna()
    # Final safety: fill any remaining tiny gaps with zero (should be none)
    features_m5 = features_m5.fillna(0)
    
    # Final validation
    if features_m5.empty:
        raise ValueError("M5 feature matrix is empty")
    
    if features_m5.isnull().any().any():
        raise ValueError("M5 feature matrix contains NaN after cleaning")
    
    logger.info(f"M5 feature engineering complete: {features_m5.shape[0]} rows × {features_m5.shape[1]} features")
    logger.info(f"Timeframe: M5 (5-minute candles)")
    logger.info(f"Date range: {features_m5.index.min()} to {features_m5.index.max()}")
    
    return features_m5
