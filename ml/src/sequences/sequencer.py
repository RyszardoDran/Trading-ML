"""Sequence creation from feature and target data.

Creates sliding windows of features with corresponding targets for model training.
Includes optional M5 alignment, trend, and pullback filters.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import SequenceFilterConfig

logger = logging.getLogger(__name__)


def create_sequences(
    features: pd.DataFrame,
    targets: pd.Series,
    window_size: int = 100,
    session: str = "all",
    custom_start: int = None,
    custom_end: int = None,
    filter_config: Optional[SequenceFilterConfig] = None,
    max_windows: int = 200000,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create sliding windows of features with corresponding targets.

    Optimized for memory:
    1. Drops NaNs from inputs first
    2. Filters by session BEFORE creating full matrix
    3. Uses stride tricks to avoid copies until necessary
    4. Caps number of windows to max_windows to prevent OOM

    Args:
        features: Per-candle feature matrix (n_samples, n_features)
        targets: Binary labels aligned with features
        window_size: Number of candles in each window
        session: Session filter ('london', 'ny', 'asian', 'london_ny', 'all', 'custom')
        custom_start: Start hour for custom session
        custom_end: End hour for custom session
        filter_config: Optional configuration controlling alignment and feature-based
            trade filters. When None, defaults replicate legacy behaviour.
        max_windows: Maximum number of windows to create

    Returns:
        X: (n_windows, window_size * n_features) array
        y: (n_windows,) array of binary labels
        timestamps: DatetimeIndex of window end times
    """
    # 1. Pre-clean data to avoid checking NaNs on huge X matrix later
    if features.isnull().values.any():
        logger.warning("Input features contain NaNs. Dropping rows with NaNs...")
        features = features.dropna()
    
    # Align features and targets
    common = features.index.intersection(targets.index)
    features = features.loc[common]
    targets = targets.loc[common]

    if len(features) < window_size:
        raise ValueError(f"Need at least {window_size} samples, got {len(features)}")

    config = filter_config or SequenceFilterConfig()
    n_features = features.shape[1]
    n_windows = len(features) - window_size + 1

    # Prepare arrays
    features_array = np.ascontiguousarray(features.values, dtype=np.float32)
    targets_array = targets.values.astype(np.int32)
    timestamps_array = features.index.values

    # 2. Calculate timestamps and targets for all potential windows
    # Timestamps aligned to the END of the window
    timestamp_indices = np.arange(window_size - 1, window_size - 1 + n_windows)
    timestamps = pd.DatetimeIndex(timestamps_array[timestamp_indices])
    
    # Targets aligned to the END of the window
    y = targets_array[window_size - 1 : window_size - 1 + n_windows]

    # 3. Apply Session Filter (Indices)
    # We calculate the mask on timestamps BEFORE creating the heavy X matrix
    hours = timestamps.hour
    if session == "london":
        mask = (hours >= 8) & (hours < 16)
    elif session == "ny":
        mask = (hours >= 13) & (hours < 22)
    elif session == "asian":
        mask = (hours >= 0) & (hours < 9)
    elif session == "london_ny":
        mask = (hours >= 8) & (hours < 22)
    elif session == "all":
        mask = np.ones(len(timestamps), dtype=bool)
    elif session == "custom":
        if custom_start is None or custom_end is None:
            raise ValueError("Must provide custom_start and custom_end for 'custom' session")
        if custom_start < custom_end:
            mask = (hours >= custom_start) & (hours < custom_end)
        else:
            mask = (hours >= custom_start) | (hours < custom_end)
    else:
        raise ValueError(f"Unknown session: {session}")

    # --- M5 ALIGNMENT FIX ---
    # Only take trades that align with M5 candles (0, 5, 10, 15...)
    # We want to trade at the OPEN of a new M5 candle.
    # The decision is made after the CLOSE of the previous M5 candle.
    # M5 candle 10:00 covers 10:00-10:04. It closes at 10:05:00.
    # So we need the window to end at minute 4, 9, 14, etc.
    if config.enable_m5_alignment:
        minutes = timestamps.minute
        m5_mask = ((minutes + 1) % 5 == 0)
        mask = mask & m5_mask
        logger.info("Applied M5 alignment filter (keeping only candles ending at 4, 9, 14... to trade at 0, 5, 10...)")
    else:
        logger.info("M5 alignment filter disabled; keeping all timestamps irrespective of minute alignment")
    # ------------------------

    # --- TREND FILTER (Long Only) ---
    # Only take trades where Price > SMA 200 (Uptrend) AND ADX > 15 (Trend exists)
    # We use 'dist_sma_200' > 0 and 'adx' > 15
    if config.enable_trend_filter:
        if "dist_sma_200" in features.columns and "adx" in features.columns:
            sma_idx = features.columns.get_loc("dist_sma_200")
            adx_idx = features.columns.get_loc("adx")

            dist_sma_values = features_array[timestamp_indices, sma_idx]
            adx_values = features_array[timestamp_indices, adx_idx]

            trend_mask = np.ones(len(timestamps), dtype=bool)
            if config.trend_min_dist_sma200 is not None:
                trend_mask &= dist_sma_values > config.trend_min_dist_sma200
            if config.trend_min_adx is not None:
                trend_mask &= adx_values > config.trend_min_adx

            mask = mask & trend_mask
            kept = int(trend_mask.sum())
            total = len(trend_mask)
            ratio = kept / total if total else 0.0
            logger.info(
                "Applied Trend Filter (dist_sma_200 > %s, ADX > %s): kept %s/%s (%.1f%%)"
                % (
                    (
                        f"{config.trend_min_dist_sma200:.2f}"
                        if config.trend_min_dist_sma200 is not None
                        else "disabled"
                    ),
                    (
                        f"{config.trend_min_adx:.2f}"
                        if config.trend_min_adx is not None
                        else "disabled"
                    ),
                    kept,
                    total,
                    ratio * 100,
                )
            )
        else:
            logger.warning("dist_sma_200 or adx feature not found; skipping trend filter")
    else:
        logger.info("Trend filter disabled; not enforcing SMA/ADX conditions")
    
    # --- PULLBACK FILTER ---
    # Avoid buying tops. Only buy if RSI_M5 is not overbought (< 75).
    # Ideally we want RSI < 60 for a pullback, but let's start with < 75 to avoid cutting too much.
    if config.enable_pullback_filter:
        if "rsi_m5" in features.columns:
            rsi_idx = features.columns.get_loc("rsi_m5")
            rsi_values = features_array[timestamp_indices, rsi_idx]

            pullback_mask = np.ones(len(timestamps), dtype=bool)
            if config.pullback_max_rsi_m5 is not None:
                pullback_mask &= rsi_values < config.pullback_max_rsi_m5

            mask = mask & pullback_mask
            kept = int(pullback_mask.sum())
            total = len(pullback_mask)
            ratio = kept / total if total else 0.0
            logger.info(
                "Applied Pullback Filter (RSI_M5 < %s): kept %s/%s (%.1f%%)"
                % (
                    (
                        f"{config.pullback_max_rsi_m5:.2f}"
                        if config.pullback_max_rsi_m5 is not None
                        else "disabled"
                    ),
                    kept,
                    total,
                    ratio * 100,
                )
            )
        else:
            logger.warning("rsi_m5 feature not found; skipping pullback filter")
    else:
        logger.info("Pullback filter disabled; not constraining RSI_M5")
    # --------------------------------

    if mask.sum() == 0:
        logger.warning(f"Session filter '{session}' removed all data!")
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    logger.info(f"Session filter '{session}': keeping {mask.sum():,} / {len(timestamps):,} windows ({mask.mean():.1%})")

    # Cap number of windows if needed (to avoid OOM during matrix creation)
    if max_windows is not None and mask.sum() > max_windows:
        logger.warning(
            f"Too many windows ({mask.sum():,}) for memory safety. "
            f"Capping to last {max_windows:,} windows."
        )
        # Find indices of True values
        true_indices = np.where(mask)[0]
        # Keep only the last max_windows indices
        keep_indices = true_indices[-max_windows:]
        # Create new mask
        new_mask = np.zeros_like(mask)
        new_mask[keep_indices] = True
        mask = new_mask
        logger.info(f"After capping: {mask.sum():,} windows")

    # Filter y and timestamps
    y = y[mask]
    timestamps = timestamps[mask]
    
    # 4. Create X only for valid windows
    # Use stride tricks to get a view, then slice with mask, then reshape
    from numpy.lib.stride_tricks import as_strided

    shape = (n_windows, window_size, n_features)
    strides = (features_array.strides[0], features_array.strides[0], features_array.strides[1])

    try:
        # Create view of ALL windows (no copy)
        windowed_view = as_strided(features_array, shape=shape, strides=strides, writeable=False)
        
        # Apply mask to view (creates copy of ONLY valid windows)
        # Shape becomes (n_valid, window_size, n_features)
        X_valid = windowed_view[mask]
        
        # Flatten (creates copy of valid windows)
        # Shape becomes (n_valid, window_size * n_features)
        X = X_valid.reshape(X_valid.shape[0], -1)
        
    except Exception as e:
        logger.warning(f"Stride trick optimization failed: {e}. Falling back to loop...")
        # Fallback: iterate only over valid indices
        valid_indices = np.where(mask)[0]
        n_valid = len(valid_indices)
        X = np.zeros((n_valid, window_size * n_features), dtype=np.float32)
        
        for i, idx in enumerate(valid_indices):
            # idx is the window index. Window starts at idx in features_array
            X[i] = features_array[idx : idx + window_size].flatten()

    return X, y, timestamps
