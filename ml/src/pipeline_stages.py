"""Pipeline execution stages for sequence XGBoost training.

Purpose:
    Encapsulate each major stage of the training pipeline:
    1. Load and prepare data
    2. Engineer features
    3. Create targets
    4. Build sequences
    5. Split and scale data
    6. Train and evaluate model
    
    Each stage is a separate function with clear input/output contracts,
    comprehensive logging, and type hints.

Usage:
    >>> from ml.src.pipeline_stages import (
    ...     load_and_prepare_data,
    ...     engineer_features_stage,
    ...     create_targets_stage,
    ...     build_sequences_stage,
    ...     split_and_scale_stage,
    ...     train_and_evaluate_stage,
    ... )
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .data_loading import load_all_years
from .features.engineer_m5 import aggregate_to_m5, engineer_m5_candle_features
from .pipelines.sequence_split import split_sequences
from .sequences import SequenceFilterConfig, create_sequences
from .targets import make_target
from .training import evaluate, evaluate_with_fixed_threshold, optimize_threshold_on_val, save_artifacts, train_xgb
from .utils.timeseries_validation import TimeSeriesValidator, validate_train_test_boundary

logger = logging.getLogger(__name__)


def validate_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    tolerance_pct: float = 10.0,
) -> None:
    """Validate that class distribution is similar across train/val/test splits.
    
    **PURPOSE**: Ensure temporal stability of target distribution. Trading signals
    should have similar win/loss ratios across different time periods. Large
    differences indicate potential issues with market regime changes or data quality.
    
    Args:
        y_train: Training targets (0/1)
        y_val: Validation targets (0/1) 
        y_test: Test targets (0/1)
        tolerance_pct: Maximum allowed difference in positive class % (default 10%)
        
    Raises:
        ValueError: If class distributions differ by more than tolerance_pct
        
    Notes:
        - Logs detailed statistics for each split
        - Warns if differences exceed tolerance but doesn't fail (for CV)
        - Critical for ensuring model generalizes across time periods
        
    Examples:
        >>> validate_class_distribution(y_train, y_val, y_test)
        TRAIN: pos=1,234 (35.6%), neg=2,234 (64.4%)
        VAL:   pos=456 (32.1%), neg=967 (67.9%)  
        TEST:  pos=789 (38.2%), neg=1,278 (61.8%)
        ✅ Class distributions are stable (max diff: 6.1% < 10.0%)
    """
    def get_stats(y, name):
        pos = int(y.sum())
        neg = int((y == 0).sum())
        pos_pct = 100 * pos / len(y) if len(y) > 0 else 0
        return pos, neg, pos_pct, f"{name}: pos={pos:,} ({pos_pct:.1f}%), neg={neg:,} ({100-pos_pct:.1f}%)"
    
    train_pos, train_neg, train_pct, train_msg = get_stats(y_train, "TRAIN")
    val_pos, val_neg, val_pct, val_msg = get_stats(y_val, "VAL")
    test_pos, test_neg, test_pct, test_msg = get_stats(y_test, "TEST")
    
    logger.info("Class distribution validation:")
    logger.info(f"  {train_msg}")
    logger.info(f"  {val_msg}")
    logger.info(f"  {test_msg}")
    
    # Calculate differences
    train_val_diff = abs(train_pct - val_pct)
    train_test_diff = abs(train_pct - test_pct)
    val_test_diff = abs(val_pct - test_pct)
    max_diff = max(train_val_diff, train_test_diff, val_test_diff)
    
    logger.info(f"  Max difference in positive class: {max_diff:.1f}% (tolerance: {tolerance_pct:.1f}%)")
    
    if max_diff > tolerance_pct:
        logger.warning(
            f"⚠️  Class distributions differ significantly (max diff: {max_diff:.1f}% > {tolerance_pct:.1f}%)"
        )
        logger.warning("   This may indicate market regime changes or data quality issues")
        logger.warning("   Model performance may be unstable across time periods")
    else:
        logger.info(f"  ✅ Class distributions are stable (max diff: {max_diff:.1f}% < {tolerance_pct:.1f}%)")


def validate_sequence_boundaries(
    timestamps: pd.DatetimeIndex,
    train_end_idx: int,
    val_end_idx: int,
    window_size: int,
) -> list[int]:
    """Validate that sequences don't cross train/val/test boundaries.
    
    **PURPOSE**: Prevent data leakage where sequences span across temporal splits.
    A sequence ending near a boundary might include data from both sides, causing
    the model to learn patterns that won't exist in production.
    
    Args:
        timestamps: Datetime index for all sequences
        train_end_idx: Last index of training set
        val_end_idx: Last index of validation set  
        window_size: Number of candles per sequence
        
    Returns:
        List of sequence indices that cross boundaries and should be removed
        
    Notes:
        - Returns indices of sequences to remove to prevent data leakage
        - Critical for time series validation integrity
        
    Examples:
        >>> to_remove = validate_sequence_boundaries(timestamps, 1000, 1200, 100)
        >>> if to_remove:
        ...     # Remove crossing sequences
        ...     X = np.delete(X, to_remove, axis=0)
        ...     y = np.delete(y, to_remove, axis=0)
        ...     timestamps = timestamps.delete(to_remove)
    """
    # Check sequences that might cross train/val boundary
    train_boundary_crossings = []
    for i in range(max(0, train_end_idx - window_size + 1), min(train_end_idx + 1, len(timestamps))):
        seq_start = timestamps[i] if i < len(timestamps) else None
        seq_end = timestamps[min(i + window_size - 1, len(timestamps) - 1)] if i < len(timestamps) else None
        if seq_start is not None and seq_end is not None:
            # If sequence spans the boundary, mark for removal
            if i <= train_end_idx < i + window_size - 1:
                train_boundary_crossings.append(i)
    
    # Check sequences that might cross val/test boundary  
    val_boundary_crossings = []
    for i in range(max(0, val_end_idx - window_size + 1), min(val_end_idx + 1, len(timestamps))):
        seq_start = timestamps[i] if i < len(timestamps) else None
        seq_end = timestamps[min(i + window_size - 1, len(timestamps) - 1)] if i < len(timestamps) else None
        if seq_start is not None and seq_end is not None:
            # If sequence spans the boundary, mark for removal
            if i <= val_end_idx < i + window_size - 1:
                val_boundary_crossings.append(i)
    
    # Combine all crossing sequences
    to_remove = sorted(set(train_boundary_crossings + val_boundary_crossings))
    
    if to_remove:
        logger.warning(f"⚠️  {len(to_remove)} sequences cross temporal boundaries (potential data leakage)")
        logger.warning(f"   Removing {len(to_remove)} sequences to prevent data leakage")
    else:
        logger.info("✅ No sequences cross temporal boundaries")
    
    return to_remove


def load_and_prepare_data(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    tolerance_pct: float = 10.0,
) -> None:
    """Validate that class distribution is similar across train/val/test splits.
    
    **PURPOSE**: Ensure temporal stability of target distribution. Trading signals
    should have similar win/loss ratios across different time periods. Large
    differences indicate potential issues with market regime changes or data quality.
    
    Args:
        y_train: Training targets (0/1)
        y_val: Validation targets (0/1) 
        y_test: Test targets (0/1)
        tolerance_pct: Maximum allowed difference in positive class % (default 10%)
        
    Raises:
        ValueError: If class distributions differ by more than tolerance_pct
        
    Notes:
        - Logs detailed statistics for each split
        - Warns if differences exceed tolerance but doesn't fail (for CV)
        - Critical for ensuring model generalizes across time periods
        
    Examples:
        >>> validate_class_distribution(y_train, y_val, y_test)
        TRAIN: pos=1,234 (35.6%), neg=2,234 (64.4%)
        VAL:   pos=456 (32.1%), neg=967 (67.9%)  
        TEST:  pos=789 (38.2%), neg=1,278 (61.8%)
        ✅ Class distributions are stable (max diff: 6.1% < 10.0%)
    """
    def get_stats(y, name):
        pos = int(y.sum())
        neg = int((y == 0).sum())
        pos_pct = 100 * pos / len(y) if len(y) > 0 else 0
        return pos, neg, pos_pct, f"{name}: pos={pos:,} ({pos_pct:.1f}%), neg={neg:,} ({100-pos_pct:.1f}%)"
    
    train_pos, train_neg, train_pct, train_msg = get_stats(y_train, "TRAIN")
    val_pos, val_neg, val_pct, val_msg = get_stats(y_val, "VAL")
    test_pos, test_neg, test_pct, test_msg = get_stats(y_test, "TEST")
    
    logger.info("Class distribution validation:")
    logger.info(f"  {train_msg}")
    logger.info(f"  {val_msg}")
    logger.info(f"  {test_msg}")
    
    # Calculate differences
    train_val_diff = abs(train_pct - val_pct)
    train_test_diff = abs(train_pct - test_pct)
    val_test_diff = abs(val_pct - test_pct)
    max_diff = max(train_val_diff, train_test_diff, val_test_diff)
    
    logger.info(f"  Max difference in positive class: {max_diff:.1f}% (tolerance: {tolerance_pct:.1f}%)")
    
    if max_diff > tolerance_pct:
        logger.warning(
            f"⚠️  Class distributions differ significantly (max diff: {max_diff:.1f}% > {tolerance_pct:.1f}%)"
        )
        logger.warning("   This may indicate market regime changes or data quality issues")
        logger.warning("   Model performance may be unstable across time periods")
    else:
        logger.info(f"  ✅ Class distributions are stable (max diff: {max_diff:.1f}% < {tolerance_pct:.1f}%)")


def load_and_prepare_data(
    data_dir: Path,
    year_filter: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Load and validate raw OHLCV data from CSV files.
    
    **PURPOSE**: Load historical XAU/USD 1-minute OHLCV data and ensure
    data quality and schema validity.
    
    Args:
        data_dir: Path to directory containing XAU_1m_data_*.csv files
        year_filter: Optional list of years to load (None = all years)
        
    Returns:
        DataFrame with columns [Date, Open, High, Low, Close, Volume]
        and datetime index sorted chronologically
        
    Raises:
        FileNotFoundError: If data_dir doesn't exist or contains no CSVs
        ValueError: If data validation fails
        
    Notes:
        - Data is automatically sorted by Date
        - Logs number of rows loaded and year range
        
    Examples:
        >>> df = load_and_prepare_data(Path('ml/src/data'))
        >>> df.shape
        (1234567, 5)  # 5 columns: Open, High, Low, Close, Volume
    """
    logger.info(f"Loading data from {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    df = load_all_years(data_dir, year_filter=year_filter)
    
    if len(df) == 0:
        raise ValueError(f"No data loaded from {data_dir}")
    
    logger.info(f"Loaded {len(df):,} rows from {data_dir}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def engineer_features_stage(
    df: pd.DataFrame,
    window_size: int,
    feature_version: str = "v2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Engineer technical features on M5 timeframe.
    
    **PURPOSE**: Aggregate M1 data to M5 bars, then engineer technical
    indicators on true M5 timeframe. This provides proper multi-timeframe
    context (M5→M15→M60) for superior predictive power.
    
    **ARCHITECTURE**: M1 raw → M5 aggregation → M5 features → M15/M60 context
    - 10,080 M1 candles (7 days) → 2,016 M5 candles
    - Model sees 100 true M5 candles = 500 minutes context (vs 100 M1 = 100 min)
    
    Args:
        df: Raw M1 OHLCV DataFrame with datetime index
        window_size: (Deprecated) Kept for backward compatibility with config
        feature_version: (Deprecated) Always uses M5 aggregation version
        
    Returns:
        Tuple of (features, df_m5):
        - features: DataFrame with engineered features on M5 timeframe
        - df_m5: DataFrame with aggregated M5 OHLCV data
        
    Raises:
        ValueError: If df is empty
        
    Notes:
        - **CRITICAL**: Aggregates M1→M5 first, then calculates indicators
        - Features are on true M5 bars (proper OHLCV aggregation)
        - Multi-timeframe: M5→M15 (3:1), M5→M60 (12:1)
        - NaN values cleaned automatically (ffill + fillna(0))
        - Compression: ~5x fewer candles (10,080 M1 → 2,016 M5)
        
    Examples:
        >>> features, df_m5 = engineer_features_stage(df_m1, window_size=60)
        >>> features.shape
        (2016, 15)  # 15 features on M5 timeframe (5x compression)
    """
    if len(df) == 0:
        raise ValueError("Cannot engineer features on empty DataFrame")
    
    # Log deprecation warning for unused parameters
    if window_size != 60:
        logger.warning(f"window_size parameter is deprecated and ignored (passed: {window_size})")
    if feature_version != "v2":
        logger.warning(f"feature_version parameter is deprecated and ignored (passed: {feature_version})")
    
    logger.info(f"Aggregating M1 data to M5 timeframe...")
    
    # Step 1: Aggregate M1 → M5 (proper OHLCV aggregation)
    df_m5 = aggregate_to_m5(df)
    logger.info(f"Aggregated {len(df):,} M1 candles → {len(df_m5):,} M5 candles ({len(df_m5)/len(df)*100:.1f}% compression)")
    
    # Step 2: Engineer features on M5 timeframe
    logger.info(f"Engineering technical features on M5 timeframe (with M15/M60 context)...")
    features = engineer_m5_candle_features(df_m5)
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Feature columns: {len(features.columns)}")
    
    return features, df_m5


def create_targets_stage(
    df_m5: pd.DataFrame,
    features: pd.DataFrame,
    atr_multiplier_sl: float,
    atr_multiplier_tp: float,
    min_hold_minutes: int,
    max_horizon: int,
) -> pd.Series:
    """Create binary target labels based on SL/TP simulation on M5 timeframe.
    
    **PURPOSE**: Simulate trading positions with ATR-based SL/TP levels.
    Assign binary labels: 1=position would be profitable (TP hit first),
    0=position would lose (SL hit first).
    
    **IMPORTANT**: Operates on M5 timeframe data (aggregated bars).
    - min_hold_minutes on M5: min_hold_minutes=5 means 1 M5 candle (5 minutes)
    - max_horizon on M5: max_horizon=60 means 60 M5 candles (300 minutes = 5 hours)
    
    Args:
        df_m5: M5 OHLCV DataFrame with datetime index (aggregated from M1)
        features: Engineered features on M5 timeframe (used only to align index)
        atr_multiplier_sl: ATR multiplier for stop-loss distance
        atr_multiplier_tp: ATR multiplier for take-profit distance
        min_hold_minutes: Minimum M5 candles to hold position (1 candle = 5 minutes)
        max_horizon: Maximum forward M5 candles to simulate (1 candle = 5 minutes)
        
    Returns:
        Series with binary labels (0/1) aligned to features index
        
    Raises:
        ValueError: If parameters are invalid or data mismatch
        
    Notes:
        - Labels are 0/1 only (binary classification)
        - Forward simulation ensures no lookahead bias
        - Class imbalance is logged for awareness
        - **TIME SCALE**: All parameters are in M5 candle units (not minutes)
        
    Examples:
        >>> targets = create_targets_stage(df_m5, features, 1.0, 2.0, 1, 60)
        >>> # min_hold=1 M5 candle (5 min), max_horizon=60 M5 candles (300 min)
        >>> targets.sum()  # Number of positive examples
        12345
        >>> targets.mean()  # Class balance
        0.35
    """
    logger.info(
        f"Creating targets on M5 timeframe (SL={atr_multiplier_sl}×ATR, TP={atr_multiplier_tp}×ATR, "
        f"min_hold={min_hold_minutes} M5 candles, max_horizon={max_horizon} M5 candles)"
    )
    
    # Use df_m5 aligned to features index
    df_aligned = df_m5.loc[features.index]
    targets = make_target(
        df_aligned,
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
        min_hold_minutes=min_hold_minutes,
        max_horizon=max_horizon,
    )
    
    logger.info(f"Targets shape: {len(targets)}")
    logger.info(f"Positive class (win): {targets.sum():,} ({targets.mean():.2%})")
    logger.info(f"Negative class (loss): {(1 - targets).sum():,} ({(1 - targets).mean():.2%})")
    
    return targets


def build_sequences_stage(
    features: pd.DataFrame,
    targets: pd.Series,
    df_dates: pd.DatetimeIndex,
    window_size: int,
    session: str,
    custom_start_hour: Optional[int],
    custom_end_hour: Optional[int],
    enable_m5_alignment: bool,
    enable_trend_filter: bool,
    trend_min_dist_sma200: Optional[float],
    trend_min_adx: Optional[float],
    enable_pullback_filter: bool,
    pullback_max_rsi_m5: Optional[float],
    max_windows: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create sliding window sequences on M5 timeframe and apply filters.
    
    **PURPOSE**: Convert M5 candles into overlapping sliding windows
    of N consecutive candles. Apply session, trend, and pullback filters
    to reduce low-quality signals.
    
    **IMPORTANT**: Data is already on M5 timeframe (aggregated bars).
    - window_size=100 means 100 M5 candles = 500 minutes of context
    - M5 alignment filter is DISABLED (data already M5-aligned)
    
    Args:
        features: Engineered features on M5 timeframe
        targets: Binary target labels on M5 timeframe
        df_dates: M5 datetime index (used for session filtering)
        window_size: Number of M5 candles per sequence (default 100 = 500 min)
        session: Trading session filter ('london', 'ny', 'london_ny', etc.)
        custom_start_hour: Start hour for custom session (0-23)
        custom_end_hour: End hour for custom session (0-23)
        enable_m5_alignment: (Deprecated - always disabled for M5 data)
        enable_trend_filter: Enforce trend conditions
        trend_min_dist_sma200: Min distance above SMA200
        trend_min_adx: Min ADX threshold
        enable_pullback_filter: Enforce RSI_M5 pullback guard
        pullback_max_rsi_m5: Max RSI_M5 allowed
        max_windows: Maximum windows to keep
        
    Returns:
        Tuple of:
        - X: Sequence features (n_sequences, n_features*window_size)
        - y: Sequence targets (n_sequences,)
        - timestamps: M5 datetime index for each sequence (n_sequences,)
        
    Raises:
        ValueError: If window_size >= len(features) or parameters invalid
        
    Notes:
        - Sequences are flattened to 2D array for XGBoost
        - Timestamps are for the last M5 candle in each sequence
        - Filtering may reduce sequences significantly
        - M5 alignment filter is DISABLED (data already M5-aligned)
        
    Examples:
        >>> X, y, ts = build_sequences_stage(features, targets, df_dates, ...)
        >>> X.shape
        (20000, 15*100)  # 20k sequences, 15 features × 100 M5 candles
        >>> y.sum()
        7000
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    
    if window_size >= len(features):
        raise ValueError(
            f"window_size ({window_size}) must be < len(features) ({len(features)})"
        )
    
    logger.info(f"Creating sequences on M5 timeframe (window_size={window_size} M5 candles = {window_size*5} minutes)...")
    
    # CRITICAL: Disable M5 alignment filter - data is already M5-aligned
    if enable_m5_alignment:
        logger.warning("M5 alignment filter is DISABLED for M5 data (data already M5-aligned)")
        enable_m5_alignment = False
    
    filter_config = SequenceFilterConfig(
        enable_m5_alignment=False,  # Always False for M5 data
        enable_trend_filter=enable_trend_filter,
        trend_min_dist_sma200=trend_min_dist_sma200,
        trend_min_adx=trend_min_adx,
        enable_pullback_filter=enable_pullback_filter,
        pullback_max_rsi_m5=pullback_max_rsi_m5,
    )
    
    logger.info(
        f"Filter config: m5_alignment=False (already M5), trend={enable_trend_filter}, "
        f"pullback={enable_pullback_filter}, max_windows={max_windows:,}"
    )
    
    X, y, timestamps = create_sequences(
        features,
        targets,
        window_size=window_size,
        session=session,
        custom_start=custom_start_hour,
        custom_end=custom_end_hour,
        filter_config=filter_config,
        max_windows=max_windows,
    )
    
    logger.info(f"Sequences created: X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"Positive class in sequences: {y.sum():,} ({y.mean():.2%})")
    
    return X, y, timestamps


def split_and_scale_stage(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    window_size: int,
    year_filter: Optional[list[int]] = None,
    use_timeseries_cv: bool = False,
    cv_folds: int = 5,
) -> Union[
    Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex,
        RobustScaler
    ],
    List[Dict]
]:
    """Split data chronologically and scale features without leakage.
    
    **PURPOSE**: Split sequences into train/val/test sets chronologically
    (no lookahead bias). Supports both single split and Time Series CV.
    Fit RobustScaler ONLY on training data, then apply to all sets.
    
    Args:
        X: Sequence features (n_sequences, n_features)
        y: Sequence targets (n_sequences,)
        timestamps: Datetime for each sequence (n_sequences,)
        year_filter: Optional list of years (affects split strategy)
        use_timeseries_cv: If True, use CV instead of single split
        cv_folds: Number of CV folds (only used if use_timeseries_cv=True)
        
    Returns:
        If use_timeseries_cv=False:
            Single split: (X_train_scaled, X_val_scaled, X_test_scaled,
                          y_train, y_val, y_test,
                          ts_train, ts_val, ts_test, scaler)
        If use_timeseries_cv=True:
            List of folds: [{'fold': int, 'train_idx': array, 'test_idx': array,
                           'X_train_scaled': array, 'X_test_scaled': array,
                           'y_train': array, 'y_test': array,
                           'timestamps_train': DatetimeIndex, 'timestamps_test': DatetimeIndex,
                           'scaler': RobustScaler}, ...]
        
    Notes:
        - CRITICAL: Prevents data leakage by fitting scaler only on train
        - Uses percentage split (70/15/15) for year_filter
        - Uses fixed date split for full dataset
        - Features converted to float32 to save memory
        - CV provides robust metrics across multiple temporal splits
        - **VALIDATES CLASS DISTRIBUTION** to ensure temporal stability
        
    Examples:
        >>> # Single split
        >>> X_tr, X_v, X_te, y_tr, y_v, y_te, ..., scaler = split_and_scale_stage(...)
        >>> X_tr.shape
        (70000, 3420)  # 70% of sequences, 57*60 features
        >>> X_tr.dtype
        dtype('float32')
        
        >>> # Time Series CV
        >>> cv_results = split_and_scale_stage(..., use_timeseries_cv=True, cv_folds=5)
        >>> len(cv_results)  # 5 folds
        5
    """
    if use_timeseries_cv:
        logger.info(f"Using Time Series Cross-Validation with {cv_folds} folds")
        
        validator = TimeSeriesValidator(n_splits=cv_folds, gap=0)
        cv_results = []
        
        for fold, train_idx, test_val_idx in validator.split(X, y):
            # Validate boundaries
            validate_train_test_boundary(timestamps, train_idx, test_val_idx)
            
            # Split test_val_idx into validation and test for threshold optimization
            n_test_val = len(test_val_idx)
            val_end = n_test_val // 2
            val_idx = test_val_idx[:val_end]
            test_idx = test_val_idx[val_end:]
            
            # Split data
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            
            # Validate class distribution for this fold
            logger.info(f"Fold {fold}: Class distribution check")
            validate_class_distribution(y_train, y_val, y_test)
            
            # Scale only on training data
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
            X_val_scaled = scaler.transform(X_val).astype(np.float32)
            X_test_scaled = scaler.transform(X_test).astype(np.float32)
            
            cv_results.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                'X_train_scaled': X_train_scaled,
                'X_val_scaled': X_val_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'timestamps_train': timestamps[train_idx],
                'timestamps_val': timestamps[val_idx],
                'timestamps_test': timestamps[test_idx],
                'scaler': scaler,
            })
        
        return cv_results  # List of folds
    
    else:
        logger.info("Using chronological 3-way split (train/val/test)...")
        
        if year_filter is not None:
            # Use percentage split for filtered data
            n = len(X)
            train_idx = int(0.7 * n)
            val_idx = int(0.85 * n)
            X_train, y_train = X[:train_idx], y[:train_idx]
            X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
            X_test, y_test = X[val_idx:], y[val_idx:]
            ts_train = timestamps[:train_idx]
            ts_val = timestamps[train_idx:val_idx]
            ts_test = timestamps[val_idx:]
            logger.info(f"Using percentage split (70/15/15) for year_filter={year_filter}")
        else:
            # Use fixed date split for full dataset
            (X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test) = \
                split_sequences(X, y, timestamps)
            logger.info("Using fixed date split for full dataset")
        
        logger.info(f"Split sizes: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
        
        # Validate class distribution stability across time periods
        validate_class_distribution(y_train, y_val, y_test)
        
        # Validate sequence boundaries to prevent data leakage
        train_end_idx = len(X_train) - 1
        val_end_idx = len(X_train) + len(X_val) - 1
        # Use explicit window_size (in M5 candles) passed into this stage
        to_remove = validate_sequence_boundaries(timestamps, train_end_idx, val_end_idx, window_size)
        
        # Remove boundary-crossing sequences if any
        if to_remove:
            logger.info(f"Removing {len(to_remove)} boundary-crossing sequences...")
            # Create mask for sequences to keep
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[to_remove] = False
            
            # Apply mask to all arrays
            X = X[keep_mask]
            y = y[keep_mask]
            timestamps = timestamps[keep_mask]
            
            # Re-split after removing sequences
            if year_filter is not None:
                n = len(X)
                train_idx = int(0.7 * n)
                val_idx = int(0.85 * n)
                X_train, y_train = X[:train_idx], y[:train_idx]
                X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
                X_test, y_test = X[val_idx:], y[val_idx:]
                ts_train = timestamps[:train_idx]
                ts_val = timestamps[train_idx:val_idx]
                ts_test = timestamps[val_idx:]
            else:
                (X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test) = \
                    split_sequences(X, y, timestamps)
            
            logger.info(f"After removal: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
        
        # CRITICAL: Fit scaler ONLY on training data
        logger.info("Scaling features with RobustScaler (robust to outliers)...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        
        logger.info(
            f"Feature scaling complete: mean={X_train_scaled.mean():.4f}, "
            f"std={X_train_scaled.std():.4f}"
        )
        
        return X_train_scaled, X_val_scaled, X_test_scaled, \
               y_train, y_val, y_test, \
               ts_train, ts_val, ts_test, \
               scaler


def train_and_evaluate_stage(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,  # ← REQUIRED dla threshold optimization
    y_val: np.ndarray,
    X_test_scaled: np.ndarray,  # ← SEPARATE from val
    y_test: np.ndarray,
    ts_test: pd.DatetimeIndex,
    random_state: int,
    min_precision: float,
    min_recall: float,
    min_trades: Optional[int],
    max_trades_per_day: Optional[int],
    use_ev_optimization: bool = False,
    use_hybrid_optimization: bool = True,
    ev_win_coefficient: float = 1.0,
    ev_loss_coefficient: float = -1.0,
    use_cost_sensitive_learning: bool = True,
    sample_weight_positive: float = 3.0,
    sample_weight_negative: float = 1.0,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> tuple[dict[str, float], object]:
    """Train model with proper validation/test separation.

    **CRITICAL**: This function enforces proper data split usage:
    - TRAIN: fit model
    - VAL: optimize threshold
    - TEST: evaluate final metrics (no threshold tuning!)

    Args:
        X_train_scaled: Scaled training features
        y_train: Training targets
        X_val_scaled: Scaled validation features (REQUIRED)
        y_val: Validation targets (REQUIRED)
        X_test_scaled: Scaled test features
        y_test: Test targets
        ts_test: Test timestamps (for daily trade capping)
        random_state: Random seed
        min_precision: Minimum precision threshold (0-1)
        min_recall: Minimum recall threshold (0-1)
        min_trades: Minimum predicted positives (None = dynamic)
        max_trades_per_day: Cap trades per day (None = unlimited)
        use_ev_optimization: If True, optimize for Expected Value instead of F1
        use_hybrid_optimization: If True, use hybrid EV + constraints
        ev_win_coefficient: Profit multiplier for correct predictions
        ev_loss_coefficient: Loss multiplier for incorrect predictions

    Returns:
        Tuple of:
        - Dictionary with metrics (threshold, win_rate, precision, recall, f1, roc_auc, pr_auc)
        - Trained and calibrated model object

    Notes:
        - Model trained with early stopping on validation set
        - Probability calibration applied for reliable confidence scores
        - Threshold optimized on VAL set, final metrics on TEST set

    Examples:
        >>> metrics, model = train_and_evaluate_stage(X_tr, y_tr, X_v, y_v, X_te, y_te, ...)
        >>> metrics['win_rate']
        0.87
        >>> metrics['threshold']
        0.45
    """
    # ===== STAGE 1: Train model on TRAIN set =====
    logger.info("Training XGBoost model...")

    # Cost-Sensitive Learning
    sample_weight = None
    if use_cost_sensitive_learning:
        logger.info(f"\n[POINT 1 OPTIMIZATION] Applying cost-sensitive learning...")
        logger.info(f"  Sample weights: positive={sample_weight_positive:.1f}, negative={sample_weight_negative:.1f}")
        sample_weight = np.where(
            y_train == 1,
            sample_weight_positive,
            sample_weight_negative
        ).astype(np.float32)
        logger.info(f"  Weight ratio (positive/negative): {sample_weight_positive / sample_weight_negative:.2f}x")

    resolved_xgb_params = xgb_params or {}
    logger.info("XGBoost profile parameters in use (%d entries):", len(resolved_xgb_params))
    for key in sorted(resolved_xgb_params):
        logger.info("  %s = %s", key, resolved_xgb_params[key])

    model = train_xgb(
        X_train_scaled, y_train,
        X_val_scaled, y_val,  # Use val for early stopping
        random_state=random_state,
        sample_weight=sample_weight,
        xgb_params=resolved_xgb_params,
    )

    logger.info("Model training complete")

    # ===== STAGE 2: Optimize threshold on VAL set =====
    if X_val_scaled is not None and y_val is not None:
        logger.info("Optimizing decision threshold on VAL set...")

        y_pred_proba_val = model.predict_proba(X_val_scaled)[:, 1]

        best_threshold, best_score = optimize_threshold_on_val(
            y_true=y_val,
            y_pred_proba=y_pred_proba_val,
            min_precision=min_precision,
            min_recall=min_recall,
            use_ev_optimization=use_ev_optimization,
            use_hybrid_optimization=use_hybrid_optimization,
            ev_win_coefficient=ev_win_coefficient,
            ev_loss_coefficient=ev_loss_coefficient,
            min_trades=min_trades,
            timestamps=None,  # For now, no timestamps on val
            max_trades_per_day=max_trades_per_day,
        )

        logger.info(f"Optimal threshold: {best_threshold:.4f} (Score={best_score:.4f} on VAL)")
    else:
        # No validation set available (e.g., CV mode) - use default threshold
        logger.info("No validation set available - using default threshold 0.5")
        best_threshold = 0.5
        best_score = None
    # ===== STAGE 3: Evaluate on TEST set with optimized threshold =====
    # ⚠️  CRITICAL: Nie używamy X_test do szukania threshold!
    # Tylko do raportowania final metrics!

    logger.info("Evaluating on TEST set with optimized threshold...")

    metrics = evaluate_with_fixed_threshold(
        model=model,
        X_test=X_test_scaled,
        y_test=y_test,
        threshold=best_threshold,
        test_timestamps=ts_test,
        max_trades_per_day=max_trades_per_day,
    )

    logger.info(
        f"Final TEST metrics: threshold={metrics['threshold']:.4f}, "
        f"win_rate={metrics['win_rate']:.4f} ({metrics['win_rate']:.2%}), "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )

    return metrics, model


def save_model_artifacts(
    model,
    scaler: RobustScaler,
    feature_columns: list[str],
    models_dir: Path,
    threshold: float,
    win_rate: float,
    window_size: int,
    analysis_window_days: int = 7,
    *,
    max_trades_per_day: int | None = None,
    min_precision: float | None = None,
    min_recall: float | None = None,
    threshold_strategy: str | None = None,
    feature_version: str | None = None,
) -> None:
    """Save trained model and artifacts to disk.
    
    **PURPOSE**: Persist model, scaler, feature names, and metadata
    for later inference and analysis.
    
    Args:
        model: Trained CalibratedClassifierCV model
        scaler: Fitted RobustScaler
        feature_columns: List of feature column names (in order)
        models_dir: Directory to save artifacts
        threshold: Selected decision threshold
        win_rate: Expected win rate (precision on test)
        window_size: Sequence window size
        analysis_window_days: Days of historical data for indicator calculation (default 7)
        
    Raises:
        PermissionError: If models_dir not writable
        
    Notes:
        - Saves 5 files: model.pkl, scaler.pkl, feature_columns.json,
          threshold.json, metadata.json
        - All files prefixed with 'sequence_' for clarity
        - Metadata includes analysis_window_days for proper inference setup
        - 7 days = ~10,080 M1 candles for robust indicator calculation
    """
    logger.info("Saving artifacts...")
    save_artifacts(
        model,
        scaler,
        feature_columns,
        models_dir,
        threshold,
        win_rate,
        window_size,
        analysis_window_days,
        max_trades_per_day=max_trades_per_day,
        min_precision=min_precision,
        min_recall=min_recall,
        threshold_strategy=threshold_strategy,
        feature_version=feature_version,
    )
    logger.info(f"Artifacts saved to {models_dir}")
