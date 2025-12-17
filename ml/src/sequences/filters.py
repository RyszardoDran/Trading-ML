"""Filters for sequence data (session, trend, pullback).

Applies various filters to sequences before training:
- Session filters (London, NY, Asian, Custom)
- Trend filters (SMA/ADX based)
- Pullback filters (RSI-based)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_by_session(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    session: str,
    custom_start: int = None,
    custom_end: int = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Filter data by trading session.

    Sessions (UTC approx):
    - london: 08:00 - 16:00
    - ny: 13:00 - 22:00
    - asian: 00:00 - 09:00
    - london_ny: 08:00 - 22:00
    - all: No filter
    - custom: Use custom_start and custom_end
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        timestamps: DatetimeIndex aligned with X and y
        session: Session name ('london', 'ny', 'asian', 'london_ny', 'all', 'custom')
        custom_start: Start hour for custom session (0-23)
        custom_end: End hour for custom session (0-23)
    
    Returns:
        (X_filtered, y_filtered, timestamps_filtered)
    
    Raises:
        ValueError: On invalid session or missing custom parameters
    """
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
        logger.info("Session filter: 'all' (keeping all data)")
        return X, y, timestamps
    elif session == "custom":
        if custom_start is None or custom_end is None:
            raise ValueError("Must provide custom_start and custom_end for 'custom' session")
        if custom_start < custom_end:
            mask = (hours >= custom_start) & (hours < custom_end)
        else:  # Crosses midnight
            mask = (hours >= custom_start) | (hours < custom_end)
    else:
        raise ValueError(f"Unknown session: {session}")

    if mask.sum() == 0:
        logger.warning(f"Session filter '{session}' removed all data! Proceeding without filter.")
        return X, y, timestamps

    logger.info(f"Session filter '{session}': kept {mask.sum()} / {len(X)} ({mask.mean():.1%})")
    return X[mask], y[mask], timestamps[mask]
