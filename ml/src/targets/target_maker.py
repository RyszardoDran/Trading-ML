"""Target (label) creation for sequence classification.

Creates binary targets based on SL/TP simulation:
- 1 (TP hit first) = positive signal
- 0 (SL hit first or timeout) = negative signal
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_target(
    df: pd.DataFrame,
    atr_multiplier_sl: float = 2.0,
    atr_multiplier_tp: float = 4.0,
    min_hold_minutes: int = 10,
    max_horizon: int = 120,
    *,
    slippage: float = 0.0,
) -> pd.Series:
    """Create binary classification target based on SL/TP hit logic (Vectorized).

    Notes:
        - When `df` is on M5 timeframe the `min_hold_minutes` and `max_horizon`
          parameters should be provided in M5 steps (e.g., min_hold_minutes=3 = 3 M5 candles).
        - We treat tie (TP and SL hit in the same future candle) conservatively as a LOSS
          (i.e., TP must strictly occur earlier than SL to be considered a win).
        - Optional `slippage` (same units as price) can be specified to make targets
          more realistic (reduce TP and increase chance of SL to account for spread/slippage).

    Definition:
        For each candle, simulate a trade with:
        - SL = entry_price - (ATR × atr_multiplier_sl) + slippage
        - TP = entry_price + (ATR × atr_multiplier_tp) - slippage
        - RR Ratio = atr_multiplier_tp / atr_multiplier_sl (default 1:2)
        
        Target = 1 if TP is hit before SL within max_horizon steps
        Target = 0 if SL hit first or neither hit within max_horizon

    Args:
        df: OHLCV DataFrame with datetime index
        atr_multiplier_sl: ATR multiplier for Stop Loss (default: 2.0)
        atr_multiplier_tp: ATR multiplier for Take Profit (default: 4.0)
        min_hold_minutes: Minimum steps (M5 candles) to wait before TP can occur
        max_horizon: Maximum forward steps (M5 candles) to simulate
        slippage: Optional price slippage/spread to subtract from TP and add to SL

    Returns:
        Binary series (0/1) aligned to original index

    Raises:
        ValueError: If parameters invalid or resulting target is empty
    """
    logger.info(f"Creating targets (SL/TP simulation) for {len(df):,} candles...")
    logger.info(f"Parameters: SL={atr_multiplier_sl}×ATR, TP={atr_multiplier_tp}×ATR, min_hold={min_hold_minutes}min, max_horizon={max_horizon}min")
    
    if atr_multiplier_sl <= 0 or atr_multiplier_tp <= 0:
        raise ValueError("ATR multipliers must be positive")
    if min_hold_minutes < 1:
        raise ValueError(f"min_hold_minutes must be >=1, got {min_hold_minutes}")
    if max_horizon < min_hold_minutes:
        raise ValueError(f"max_horizon ({max_horizon}) must be >= min_hold_minutes ({min_hold_minutes})")

    # Calculate ATR
    # Use float32 to save memory and improve speed
    high_series = df["High"].astype(np.float32)
    low_series = df["Low"].astype(np.float32)
    close_series = df["Close"].astype(np.float32)
    
    # Prefer explicit 'atr_m5' if present (engineer_m5 now provides this). Fallback to other names
    if "atr_m5" in df.columns:
        logger.info("Using pre-calculated M5 ATR (atr_m5) for SL/TP targets")
        atr_series = df["atr_m5"].astype(np.float32)
    elif "ATR_M5" in df.columns:
        logger.info("Using pre-calculated M5 ATR (ATR_M5) for SL/TP targets")
        atr_series = df["ATR_M5"].astype(np.float32)
    else:
        logger.info("No precomputed ATR column found; calculating ATR (14) on provided timeframe")
        tr1 = high_series - low_series
        tr2 = np.abs(high_series - close_series.shift(1))
        tr3 = np.abs(low_series - close_series.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_series = true_range.rolling(14, min_periods=1).mean()
    
    # Prepare arrays for vectorization
    closes = close_series.values
    highs = high_series.values
    lows = low_series.values
    atrs = atr_series.values
    
    # Calculate levels
    # Apply optional slippage conservatively: increase sl (closer) and decrease tp (harder to hit)
    sl_levels = closes - (atrs * atr_multiplier_sl) + slippage
    tp_levels = closes + (atrs * atr_multiplier_tp) - slippage
    
    # Vectorized look-forward using stride_tricks
    from numpy.lib.stride_tricks import as_strided
    
    n_samples = len(df)
    valid_samples = n_samples - max_horizon
    
    if valid_samples <= 0:
        raise ValueError(f"Not enough data for max_horizon={max_horizon}")

    # Create strided views for High and Low
    # Shape: (valid_samples, max_horizon)
    # We start looking from index 1 (next candle) relative to current i
    # So future_highs[i, j] corresponds to highs[i + 1 + j]
    
    itemsize = highs.itemsize
    shape = (valid_samples, max_horizon)
    strides = (itemsize, itemsize)
    
    future_highs = as_strided(highs[1:], shape=shape, strides=strides)
    future_lows = as_strided(lows[1:], shape=shape, strides=strides)
    
    # Slice to respect min_hold_minutes
    # We want to check from i + min_hold_minutes to i + max_horizon
    # In our window (0-based), index j corresponds to i + 1 + j
    # We want i + 1 + j >= i + min_hold_minutes => j >= min_hold_minutes - 1
    start_idx = min_hold_minutes - 1
    
    future_highs = future_highs[:, start_idx:]
    future_lows = future_lows[:, start_idx:]
    
    # Align levels to valid_samples and broadcast
    tp_levels_valid = tp_levels[:valid_samples, None]
    sl_levels_valid = sl_levels[:valid_samples, None]
    
    # Check hits (boolean matrices)
    hit_tp = future_highs >= tp_levels_valid
    hit_sl = future_lows <= sl_levels_valid
    
    # Find first occurrence index
    # argmax returns index of first True, or 0 if all False
    tp_idx = np.argmax(hit_tp, axis=1)
    sl_idx = np.argmax(hit_sl, axis=1)
    
    # Check if any hit occurred
    tp_any = hit_tp.max(axis=1)
    sl_any = hit_sl.max(axis=1)
    
    # Determine target (CONSERVATIVE):
    # TP wins only if TP was hit and it was strictly before SL. If both hit in same candle, treat as LOSS.
    tp_wins = tp_any & (~sl_any | (tp_idx < sl_idx))
    
    # Initialize target array
    target = np.zeros(n_samples, dtype=np.float32)
    target[:valid_samples] = tp_wins.astype(np.float32)
    
    # Set invalid/future targets to NaN
    target[valid_samples:] = np.nan
    
    # Convert to Series and drop NaNs
    target_series = pd.Series(target, index=df.index)
    target_series = target_series.dropna().astype(int)
    
    if target_series.empty:
        raise ValueError("Target series is empty; check data quality or adjust parameters")
    
    logger.info(f"Target creation complete: {len(target_series):,} valid targets")
    return target_series
