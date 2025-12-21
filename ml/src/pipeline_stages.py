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
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ml.src.data_loading import load_all_years
from ml.src.features.engineer_m5 import aggregate_to_m5, engineer_m5_candle_features
from ml.src.pipelines.sequence_split import split_sequences
from ml.src.sequences import SequenceFilterConfig, create_sequences
from ml.src.targets import make_target
from ml.src.training import evaluate, save_artifacts, train_xgb

logger = logging.getLogger(__name__)


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
) -> pd.DataFrame:
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
        DataFrame with 15 engineered features on M5 timeframe.
        Index is M5-aligned (timestamps ending at :00, :05, :10, :15, etc.)
        
    Raises:
        ValueError: If df is empty
        
    Notes:
        - **CRITICAL**: Aggregates M1→M5 first, then calculates indicators
        - Features are on true M5 bars (proper OHLCV aggregation)
        - Multi-timeframe: M5→M15 (3:1), M5→M60 (12:1)
        - NaN values cleaned automatically (ffill + fillna(0))
        - Compression: ~5x fewer candles (10,080 M1 → 2,016 M5)
        
    Examples:
        >>> features = engineer_features_stage(df_m1, window_size=60)
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
    
    return features


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
    year_filter: Optional[list[int]] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex,
    RobustScaler
]:
    """Split data chronologically and scale features without leakage.
    
    **PURPOSE**: Split sequences into train/val/test sets chronologically
    (no lookahead bias). Fit RobustScaler ONLY on training data, then
    apply to all sets.
    
    Args:
        X: Sequence features (n_sequences, n_features)
        y: Sequence targets (n_sequences,)
        timestamps: Datetime for each sequence (n_sequences,)
        year_filter: Optional list of years (affects split strategy)
        
    Returns:
        Tuple of:
        - X_train_scaled: Training features (float32)
        - X_val_scaled: Validation features (float32)
        - X_test_scaled: Test features (float32)
        - y_train: Training targets
        - y_val: Validation targets
        - y_test: Test targets
        - ts_train: Training timestamps
        - ts_val: Validation timestamps
        - ts_test: Test timestamps
        - scaler: RobustScaler fitted on training data
        
    Notes:
        - CRITICAL: Prevents data leakage by fitting scaler only on train
        - Uses percentage split (70/15/15) for year_filter
        - Uses fixed date split for full dataset
        - Features converted to float32 to save memory
        
    Examples:
        >>> X_tr, X_v, X_te, y_tr, y_v, y_te, ..., scaler = split_and_scale_stage(...)
        >>> X_tr.shape
        (70000, 3420)  # 70% of sequences, 57*60 features
        >>> X_tr.dtype
        dtype('float32')
    """
    logger.info("Splitting data chronologically (train/val/test)...")
    
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
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    X_test_scaled: np.ndarray,
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
) -> tuple[dict[str, float], object]:
    """Train XGBoost model with calibration and evaluate on test set.
    
    **PURPOSE**: Train calibrated XGBoost classifier, select optimal
    decision threshold based on precision constraint or Expected Value,
    and evaluate on held-out test set.
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training targets
        X_val_scaled: Scaled validation features
        y_val: Validation targets
        X_test_scaled: Scaled test features
        y_test: Test targets
        ts_test: Test timestamps (for daily trade capping)
        random_state: Random seed
        min_precision: Minimum precision threshold (0-1)
        min_trades: Minimum predicted positives (None = dynamic)
        max_trades_per_day: Cap trades per day (None = unlimited)
        use_ev_optimization: If True, optimize for Expected Value instead of F1
        ev_win_coefficient: Profit multiplier for correct predictions
        ev_loss_coefficient: Loss multiplier for incorrect predictions
        
    Returns:
        Tuple of:
        - Dictionary with metrics (threshold, win_rate, precision, recall, f1, roc_auc, pr_auc)
        - Trained and calibrated model object
        
    Notes:
        - Model trained with early stopping on validation set
        - Probability calibration applied for reliable confidence scores
        - Threshold optimized on validation or test set based on strategy
        
    Examples:
        >>> metrics, model = train_and_evaluate_stage(X_tr, y_tr, X_v, y_v, ...)
        >>> metrics['win_rate']
        0.87
        >>> metrics['threshold']
        0.45
    """
    logger.info("Training XGBoost classifier...")
    model = train_xgb(X_train_scaled, y_train, X_val_scaled, y_val, random_state=random_state)
    
    logger.info("Evaluating model on test set...")
    metrics = evaluate(
        model,
        X_test_scaled,
        y_test,
        min_precision=min_precision,
        min_recall=min_recall,
        min_trades=min_trades,
        test_timestamps=ts_test,
        max_trades_per_day=max_trades_per_day,
        use_ev_optimization=use_ev_optimization,
        use_hybrid_optimization=use_hybrid_optimization,
        ev_win_coefficient=ev_win_coefficient,
        ev_loss_coefficient=ev_loss_coefficient,
    )
    
    logger.info(
        f"Evaluation metrics: threshold={metrics['threshold']:.2f}, "
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
    )
    logger.info(f"Artifacts saved to {models_dir}")
