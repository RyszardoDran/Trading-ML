"""Chronological train/val/test splitting for sequence data.

Purpose:
    Split sequences while maintaining temporal order (no data leakage).

Example:
    >>> X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test = split_sequences(
    ...     X, y, timestamps
    ... )
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    train_until: str = "2023-12-31 23:59:00",
    val_until: str = "2024-06-30 23:59:00",
    test_until: str = "2025-12-31 23:59:00",
    gap_days: int = 7,  # Dodaj gap w dniach między splitami, aby uniknąć data leakage
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    """Chronological split of sequence data.

    Args:
        X: Feature array (n_windows, n_features)
        y: Target array (n_windows,)
        timestamps: Timestamps corresponding to each window
        train_until: End of train period
        val_until: End of validation period
        test_until: End of test period
        gap_days: Number of days to skip between splits to prevent data leakage

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, timestamps_train, timestamps_val, timestamps_test

    Raises:
        ValueError: If any split is empty or insufficient coverage
    """
    # Validate input shapes
    if len(X) != len(y) or len(X) != len(timestamps):
        raise ValueError(
            f"Shape mismatch: X={len(X)}, y={len(y)}, timestamps={len(timestamps)}"
        )
    
    if len(X) == 0:
        raise ValueError("Empty input arrays provided")
    
    train_end = pd.Timestamp(train_until)
    val_end = pd.Timestamp(val_until)
    test_end = pd.Timestamp(test_until)
    
    # Validate chronological order
    if train_end >= val_end:
        raise ValueError(f"train_until ({train_until}) must be before val_until ({val_until})")
    if val_end >= test_end:
        raise ValueError(f"val_until ({val_until}) must be before test_until ({test_until})")

    # Dodaj gap między splitami
    val_start = train_end + pd.Timedelta(days=gap_days)
    test_start = val_end + pd.Timedelta(days=gap_days)

    train_mask = timestamps <= train_end
    val_mask = (timestamps > val_start) & (timestamps <= val_end)
    test_mask = (timestamps > test_start) & (timestamps <= test_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if min(map(len, [X_train, X_val, X_test])) == 0:
        raise ValueError(
            f"One of the splits is empty; verify data coverage for train/val/test ranges. "
            f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
    
    # Log split statistics
    logger.info(
        f"Chronological split complete: "
        f"Train={len(X_train)} ({len(X_train)/len(X)*100:.1f}%), "
        f"Val={len(X_val)} ({len(X_val)/len(X)*100:.1f}%), "
        f"Test={len(X_test)} ({len(X_test)/len(X)*100:.1f}%)"
    )
    logger.info(
        f"Time ranges: "
        f"Train=[{timestamps[train_mask].min()} to {timestamps[train_mask].max()}], "
        f"Val=[{timestamps[val_mask].min()} to {timestamps[val_mask].max()}], "
        f"Test=[{timestamps[test_mask].min()} to {timestamps[test_mask].max()}]"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, timestamps[train_mask], timestamps[val_mask], timestamps[test_mask]
