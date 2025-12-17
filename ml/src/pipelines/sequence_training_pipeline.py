from __future__ import annotations
"""Sequence-based training pipeline for XAU/USD (1-minute data).

Purpose:
- Load historical 1m OHLCV data from `ml/src/data/`
- Create sliding windows of 60 candles (configurable) as input features
- Engineer features for each candle in the sequence (flatten to XGBoost input)
- Train XGBoost classifier with calibrated probability estimates
- Validate win rate on test set with realistic thresholds
- Provide user with expected win rate and confidence metrics

Key principles:
- Temporal context: Model sees 100 previous candles before making prediction
- No data leakage: Strict chronological split, no future information
- Win rate validation: Precision, recall, F1, and confusion matrix on test
- Reproducibility: Fixed random seeds, deterministic behavior
- Production-ready: Type hints, validation, error handling, logging

Inputs (CSV):
- `ml/src/data/XAU_1m_data_*.csv` (semicolon-separated OHLCV)

Outputs (artifacts):
- `ml/src/models/sequence_xgb_model.pkl` (calibrated classifier)
- `ml/src/models/sequence_feature_columns.json` (ordered feature names)
- `ml/src/models/sequence_threshold.json` (selected threshold + win rate)

Expected columns: [Date;Open;High;Low;Close;Volume]
Separator: `;`, Date parseable to datetime

Usage:
    # Train with default parameters
    python sequence_training_pipeline.py
    
    # Train with custom window size and horizon
    python sequence_training_pipeline.py --window-size 50 --horizon 10
    
    # Health check only
    python sequence_training_pipeline.py --health-check-dir ml/src/data
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directories to path for imports
_script_dir = Path(__file__).parent
_src_dir = _script_dir.parent
_repo_dir = _src_dir.parent.parent
sys.path.insert(0, str(_repo_dir))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Import from modularized components
from ml.src.data_loading import load_all_years, validate_schema
from ml.src.sequences import create_sequences, filter_by_session, SequenceFilterConfig
from ml.src.targets import make_target
from ml.src.utils import PipelineConfig
from ml.src.pipelines.sequence_split import split_sequences
from ml.src.features import engineer_candle_features
from ml.src.training import train_xgb, evaluate, save_artifacts


def _setup_logging(config: PipelineConfig, year_filter: Optional[List[int]] = None) -> str:
    """Setup logging to file with unique timestamp-based name.
    
    Args:
        config: PipelineConfig with output directory settings
        year_filter: List of years being trained (for log name description)
        
    Returns:
        Path to the log file as string
        
    Notes:
        - Log file name includes timestamp: sequence_xgb_train_YYYYMMDD_HHMMSS.log
        - Creates ml/outputs/logs/ directory if it doesn't exist
        - Also logs to console (INFO level)
    """
    # Create logs directory
    logs_dir = config.outputs_logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    years_suffix = f"_years_{'_'.join(map(str, year_filter))}" if year_filter else "_all_years"
    log_filename = f"sequence_xgb_train{years_suffix}_{timestamp}.log"
    log_filepath = logs_dir / log_filename
    
    # Configure root logger with both file and console handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    return str(log_filepath)


logger = logging.getLogger(__name__)


def run_pipeline(
    window_size: int = 60,
    atr_multiplier_sl: float = 1.0,
    atr_multiplier_tp: float = 2.0,
    min_hold_minutes: int = 5,
    max_horizon: int = 60,
    random_state: int = 42,
    year_filter: List[int] = None,
    session: str = "london_ny",
    custom_start_hour: int = None,
    custom_end_hour: int = None,
    max_windows: int = 200000,
    min_precision: float = 0.85,
    min_trades: int | None = None,
    max_trades_per_day: int | None = None,
    enable_m5_alignment: bool = True,
    enable_trend_filter: bool = True,
    trend_min_dist_sma200: float | None = 0.0,
    trend_min_adx: float | None = 15.0,
    enable_pullback_filter: bool = True,
    pullback_max_rsi_m5: float | None = 75.0,
) -> Dict[str, float]:
    """Execute end-to-end sequence XGBoost training pipeline for XAU/USD trading.

    **PURPOSE**: Train a calibrated XGBoost classifier on sliding window sequences
    of 1-minute candlestick data to predict profitable trading opportunities in XAU/USD.
    
    **WHAT THIS DOES**:
    1. Loads historical XAU/USD 1-minute OHLCV data from ml/src/data/
    2. Engineers 57+ technical features per candle (ATR, RSI, MACD, Bollinger Bands, etc.)
    3. Creates overlapping sliding windows of N consecutive candles as training sequences
    4. Assigns binary labels (0=loss, 1=win) based on SL/TP simulation with ATR multiples
    5. Applies session filters (London/NY hours, trend conditions, pullback guards)
    6. Splits chronologically into train/val/test sets (no data leakage)
    7. Trains XGBoost classifier with early stopping on validation set
    8. Calibrates probability estimates for reliable confidence scoring
    9. Evaluates on test set and selects optimal decision threshold
    10. Saves artifacts: model, scaler, feature columns, threshold configuration

    **DATA FLOW**:
    Raw OHLCV → Feature Engineering → Sequences → Session Filtering → 
    Train/Val/Test Split → Scaling (no leakage) → XGBoost Training → 
    Calibration → Threshold Optimization → Evaluation

    **OUTPUTS** (saved to ml/outputs/models/):
    - sequence_xgb_model.pkl: Calibrated XGBoost classifier
    - sequence_scaler.pkl: RobustScaler fitted on training features
    - sequence_feature_columns.json: Ordered list of 57+ feature names
    - sequence_metadata.json: Training configuration and hyperparameters
    - sequence_threshold.json: Optimal decision threshold and expected win rate

    **VALIDATION & SAFETY**:
    - Temporal chronological split (no lookahead bias)
    - Scaler fit ONLY on training set (prevent leakage)
    - Early stopping using independent validation set
    - Win rate (precision) reported on held-out test set
    - Session filtering removes low-quality trading hours
    - Trend/pullback guards reduce false signals

    Args:
        window_size: Number of previous candles to use as input (default: 60)
        atr_multiplier_sl: ATR multiplier for stop-loss (default: 1.0)
        atr_multiplier_tp: ATR multiplier for take-profit (default: 2.0 for 2:1 RR)
        min_hold_minutes: Minimum holding time in minutes (default: 5)
        max_horizon: Maximum forward candles to simulate (default: 60)
        random_state: Random seed for reproducibility (default: 42)
        year_filter: Optional list of years to load (e.g., [2023, 2024] for testing)
        max_windows: Maximum number of windows to keep to avoid OOM (default: 200,000)
        min_precision: Minimum precision (win rate) floor for threshold selection (default: 0.85)
        min_trades: Minimum number of predicted positives for threshold to be considered
        max_trades_per_day: Cap number of predicted trades per day after thresholding
        enable_m5_alignment: Align decisions with M5 candle closes when True (default: True)
        enable_trend_filter: Enforce SMA200/ADX trend conditions when True (default: True)
        trend_min_dist_sma200: Minimum normalized distance above SMA200; None to disable
        trend_min_adx: Minimum ADX threshold for trend filter; None to disable
        enable_pullback_filter: Enforce RSI_M5 pullback guard when True (default: True)
        pullback_max_rsi_m5: Maximum RSI_M5 allowed; None to disable

    Returns:
        Dictionary with evaluation metrics:
        - threshold: Optimal decision threshold (0-1)
        - win_rate: Precision on test set (expected win rate)
        - precision: Precision (TP / (TP + FP))
        - recall: Recall (TP / (TP + FN))
        - f1: F1 score
        - roc_auc: Area under ROC curve
        - pr_auc: Area under Precision-Recall curve
        - confusion_matrix: [TN, FP, FN, TP]
        - n_positive_predictions: Count of predicted positives at threshold

    Raises:
        FileNotFoundError: If data files not found at ml/src/data/
        ValueError: On validation failures or insufficient data
        
    Notes:
        - THIS IS CRITICAL PRODUCTION TRAINING CODE
        - All hyperparameters, feature engineering, and filters are NOT random
        - Random seeds fixed for reproducibility (seeded model, scaler, splits)
        - Results logged to ml/outputs/logs/sequence_xgb_train_*.log with timestamp
    """
    np.random.seed(random_state)
    import random
    random.seed(random_state)
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"

    logger.info("Loading data...")
    df = load_all_years(data_dir, year_filter=year_filter)
    logger.info(f"Loaded {len(df):,} rows from {data_dir}")

    logger.info(f"Engineering per-candle features (window_size={window_size})...")
    features = engineer_candle_features(df, window_size=window_size)
    logger.info(f"Features shape: {features.shape}")

    logger.info(f"Creating target (SL={atr_multiplier_sl}×ATR, TP={atr_multiplier_tp}×ATR, min_hold={min_hold_minutes}min)...")
    targets = make_target(
        df.loc[features.index],
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
        min_hold_minutes=min_hold_minutes,
        max_horizon=max_horizon,
    )
    logger.info(f"Target shape: {len(targets)}, positive class: {targets.sum()} ({targets.mean():.2%})")

    logger.info(f"Creating sequences (window_size={window_size})...")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    filter_config = SequenceFilterConfig(
        enable_m5_alignment=enable_m5_alignment,
        enable_trend_filter=enable_trend_filter,
        trend_min_dist_sma200=trend_min_dist_sma200,
        trend_min_adx=trend_min_adx,
        enable_pullback_filter=enable_pullback_filter,
        pullback_max_rsi_m5=pullback_max_rsi_m5,
    )
    logger.info(
        "Filter configuration: m5=%s, trend=%s(dist_sma200=%s, adx=%s), pullback=%s(rsi<=%s)"
        % (
            enable_m5_alignment,
            enable_trend_filter,
            trend_min_dist_sma200 if trend_min_dist_sma200 is not None else "disabled",
            trend_min_adx if trend_min_adx is not None else "disabled",
            enable_pullback_filter,
            pullback_max_rsi_m5 if pullback_max_rsi_m5 is not None else "disabled",
        )
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
    logger.info(f"Sequences: X.shape={X.shape}, y.shape={y.shape}")

    logger.info("Splitting data (chronological train/val/test)...")
    # Dynamic split based on data range
    if year_filter is not None:
        # For year filter: use percentage split to avoid empty splits
        n = len(X)
        train_idx = int(0.7 * n)
        val_idx = int(0.85 * n)
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        ts_train, ts_val, ts_test = (
            timestamps[:train_idx],
            timestamps[train_idx:val_idx],
            timestamps[val_idx:],
        )
        logger.info(f"Using percentage split (70/15/15) for year_filter={year_filter}")
    else:
        # Full date range: use fixed date splits
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            ts_train,
            ts_val,
            ts_test,
        ) = split_sequences(X, y, timestamps)
    logger.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # CRITICAL: Scale features AFTER split to prevent data leakage
    # Fit scaler ONLY on training data, then transform all sets
    logger.info("Scaling features with RobustScaler (robust to outliers)...")
    scaler = RobustScaler()
    # Ensure float32 output to save memory
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    logger.info(f"Feature scaling complete: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")

    logger.info("Training XGBoost classifier...")
    model = train_xgb(X_train_scaled, y_train, X_val_scaled, y_val, random_state=random_state)

    logger.info("Evaluating model on test set...")
    metrics = evaluate(
        model,
        X_test_scaled,
        y_test,
        min_precision=min_precision,
        min_trades=min_trades,
        test_timestamps=ts_test,
        max_trades_per_day=max_trades_per_day,
    )
    logger.info(
        "Metrics: "
        f"threshold={metrics['threshold']:.2f}, "
        f"win_rate={metrics['win_rate']:.4f} ({metrics['win_rate']:.2%}), "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )

    logger.info("Saving artifacts...")
    logger.info("Saving artifacts (model, scaler, metadata)...")
    save_artifacts(
        model,
        scaler,
        list(features.columns),
        models_dir,
        metrics["threshold"],
        metrics["win_rate"],
        window_size,
    )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sequence-based XAU/USD training pipeline")
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Number of previous candles to use as input (default: 60)",
    )
    parser.add_argument(
        "--atr-multiplier-sl",
        type=float,
        default=1.0,
        help="ATR multiplier for stop-loss level (default: 1.0 - DO NOT CHANGE)",
    )
    parser.add_argument(
        "--atr-multiplier-tp",
        type=float,
        default=2.0,
        help="ATR multiplier for take-profit level (default: 2.0 - DO NOT CHANGE)",
    )
    parser.add_argument(
        "--min-hold-minutes",
        type=int,
        default=5,
        help="Minimum hold time in minutes (default: 5)",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=60,
        help="Maximum forward candles to simulate (default: 60)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years to load (e.g., '2023,2024' for testing)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="london_ny",
        choices=["london", "ny", "asian", "london_ny", "all", "custom"],
        help="Trading session to filter data (default: london_ny)",
    )
    parser.add_argument(
        "--custom-start-hour",
        type=int,
        default=None,
        help="Start hour for custom session (0-23)",
    )
    parser.add_argument(
        "--custom-end-hour",
        type=int,
        default=None,
        help="End hour for custom session (0-23)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200000,
        help="Maximum number of windows to keep to avoid OOM (default: 200,000)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.85,
        help="Minimum precision (win rate) floor for threshold selection (default: 0.85)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        help="Minimum number of predicted positives for a threshold to be considered (default: dynamic)",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=None,
        help="Cap number of predicted trades per day after thresholding (default: unlimited)",
    )
    parser.add_argument(
        "--skip-m5-alignment",
        action="store_true",
        help="Disable M5 alignment filter (default: enabled)",
    )
    parser.add_argument(
        "--disable-trend-filter",
        action="store_true",
        help="Disable trend filter requiring price above SMA200 and ADX threshold",
    )
    parser.add_argument(
        "--trend-min-dist-sma200",
        type=float,
        default=0.0,
        help="Minimum normalized distance above SMA200 when trend filter enabled (default: 0.0)",
    )
    parser.add_argument(
        "--trend-min-adx",
        type=float,
        default=15.0,
        help="Minimum ADX when trend filter enabled (default: 15.0)",
    )
    parser.add_argument(
        "--disable-pullback-filter",
        action="store_true",
        help="Disable RSI_M5 pullback guard",
    )
    parser.add_argument(
        "--pullback-max-rsi-m5",
        type=float,
        default=75.0,
        help="Maximum RSI_M5 when pullback filter enabled (default: 75.0)",
    )
    args = parser.parse_args()

    # Parse year filter
    year_filter = None
    if args.years:
        year_filter = [int(y.strip()) for y in args.years.split(',')]

    # Setup logging to unique file in ml/outputs/logs/
    config = PipelineConfig()
    log_filepath = _setup_logging(config, year_filter)
    
    # Log training session header with full configuration
    logger.info("\n" + "=" * 80)
    logger.info("SEQUENCE XGBoost TRAINING PIPELINE - XAU/USD 1-Minute Data")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data years: {year_filter if year_filter else 'All available years'}")
    logger.info("\nPipeline Configuration:")
    logger.info(f"  Window size: {args.window_size} candles")
    logger.info(f"  ATR SL multiplier: {args.atr_multiplier_sl}x")
    logger.info(f"  ATR TP multiplier: {args.atr_multiplier_tp}x")
    logger.info(f"  Min hold time: {args.min_hold_minutes} minutes")
    logger.info(f"  Max horizon: {args.max_horizon} candles")
    logger.info(f"  Trading session: {args.session}")
    logger.info(f"  Min precision threshold: {args.min_precision}")
    logger.info(f"  M5 alignment filter: {'enabled' if not args.skip_m5_alignment else 'disabled'}")
    logger.info(f"  Trend filter: {'enabled' if not args.disable_trend_filter else 'disabled'}")
    logger.info(f"  Pullback filter: {'enabled' if not args.disable_pullback_filter else 'disabled'}")
    logger.info(f"  Random state: {args.random_state}")
    logger.info("=" * 80 + "\n")

    try:
        metrics = run_pipeline(
            window_size=args.window_size,        
            atr_multiplier_sl=args.atr_multiplier_sl,
            atr_multiplier_tp=args.atr_multiplier_tp,
            min_hold_minutes=args.min_hold_minutes,
            max_horizon=args.max_horizon,        
            random_state=args.random_state,      
            year_filter=year_filter,
            session=args.session,
            custom_start_hour=args.custom_start_hour,
            custom_end_hour=args.custom_end_hour,
            max_windows=args.max_windows,
            min_precision=args.min_precision,
            min_trades=args.min_trades,
            max_trades_per_day=args.max_trades_per_day,
            enable_m5_alignment=not args.skip_m5_alignment,
            enable_trend_filter=not args.disable_trend_filter,
            trend_min_dist_sma200=None if args.disable_trend_filter else args.trend_min_dist_sma200,
            trend_min_adx=None if args.disable_trend_filter else args.trend_min_adx,
            enable_pullback_filter=not args.disable_pullback_filter,
            pullback_max_rsi_m5=None if args.disable_pullback_filter else args.pullback_max_rsi_m5,
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - SEQUENCE PIPELINE")
        print("=" * 60)
        print(f"Window Size:       {args.window_size} candles")
        print(f"Threshold:         {metrics['threshold']:.2f}")
        print(f"WIN RATE:          {metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1 Score:          {metrics['f1']:.4f}")
        print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:            {metrics['pr_auc']:.4f}")
        print("=" * 60)
        print(f"\nWin rate is the precision: when model predicts 'BUY',")
        print(f"it will be correct {metrics['win_rate']:.2%} of the time on test data.")
        print(f"\nLog file saved to: {log_filepath}")
        print("=" * 60)
        
        # Log completion summary to file
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\nFinal Metrics:")
        logger.info(f"  Window Size:       {args.window_size} candles")
        logger.info(f"  Threshold:         {metrics['threshold']:.4f}")
        logger.info(f"  WIN RATE:          {metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})")
        logger.info(f"  Precision:         {metrics['precision']:.4f}")
        logger.info(f"  Recall:            {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:          {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
        logger.info(f"\nArtifacts saved to: {config.outputs_models_dir}")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error("Data files not found. Ensure CSVs exist at 'ml/src/data/XAU_1m_data_*.csv'.")
        logger.error(f"Error: {str(e)}")
        print("Data files not found. Ensure CSVs exist at 'ml/src/data/XAU_1m_data_*.csv'.")
        print(str(e))
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
