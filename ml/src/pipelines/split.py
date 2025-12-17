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
    train_until: str = "2022-12-31 23:59:00",
    val_until: str = "2023-12-31 23:59:00",
    test_until: str = "2024-12-31 23:59:00",
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

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, timestamps_train, timestamps_val, timestamps_test

    Raises:
        ValueError: If any split is empty or insufficient coverage
    """
    train_end = pd.Timestamp(train_until)
    val_end = pd.Timestamp(val_until)
    test_end = pd.Timestamp(test_until)

    train_mask = timestamps <= train_end
    val_mask = (timestamps > train_end) & (timestamps <= val_end)
    test_mask = (timestamps > val_end) & (timestamps <= test_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if min(map(len, [X_train, X_val, X_test])) == 0:
        raise ValueError("One of the splits is empty; verify data coverage for train/val/test ranges")

    return X_train, X_val, X_test, y_train, y_val, y_test, timestamps[train_mask], timestamps[val_mask], timestamps[test_mask]
