from __future__ import annotations
"""Sequence-based training pipeline orchestrator for XAU/USD (1-minute data).

**PURPOSE**: End-to-end training pipeline orchestrator. Coordinates execution
of modularized pipeline stages: data loading, feature engineering, target creation,
sequence building, data splitting/scaling, model training, and artifact saving.

**WHAT THIS DOES**:
1. Parse CLI arguments and validate configuration
2. Setup logging to unique timestamped file
3. Execute pipeline stages in sequence
4. Handle errors and report results

**KEY PRINCIPLES**:
- Modularity: Each stage is a separate function in pipeline_stages.py
- Clarity: Main run_pipeline() is 40 lines - coordinates stages only
- Type Safety: All functions have comprehensive type hints
- Reproducibility: Fixed random seeds, deterministic behavior
- Production-Ready: Comprehensive logging, error handling, validation

**INPUTS** (CSV):
- `ml/src/data/XAU_1m_data_*.csv` (semicolon-separated OHLCV)

**OUTPUTS** (artifacts saved to ml/src/models/):
- sequence_xgb_model.pkl: Calibrated XGBoost classifier
- sequence_scaler.pkl: RobustScaler fitted on training data
- sequence_feature_columns.json: Ordered list of feature names
- sequence_metadata.json: Training configuration
- sequence_threshold.json: Decision threshold + win rate

**USAGE**:
    # Train with defaults
    python sequence_training_pipeline.py
    
    # Custom parameters
    python sequence_training_pipeline.py --window-size 50 --min-precision 0.90
    
    # Specific years
    python sequence_training_pipeline.py --years 2023,2024

**EXPECTED COLUMNS**: [Date;Open;High;Low;Close;Volume]
**SEPARATOR**: `;` (semicolon)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add parent directories to path for imports
_script_dir = Path(__file__).parent
_src_dir = _script_dir.parent
_repo_dir = _src_dir.parent.parent
sys.path.insert(0, str(_repo_dir))

import numpy as np
from ml.src.pipeline_cli import parse_cli_arguments, parse_and_validate_years
from ml.src.pipeline_config_extended import PipelineParams
from ml.src.pipeline_stages import (
    build_sequences_stage,
    create_targets_stage,
    engineer_features_stage,
    load_and_prepare_data,
    save_model_artifacts,
    split_and_scale_stage,
    train_and_evaluate_stage,
)
from ml.src.utils import PipelineConfig


logger = logging.getLogger(__name__)


def _setup_logging(config: PipelineConfig, year_filter: Optional[list[int]] = None) -> str:
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



def run_pipeline(params: PipelineParams) -> Dict[str, float]:
    """Execute end-to-end sequence XGBoost training pipeline.

    **PURPOSE**: Orchestrate complete training pipeline from data loading
    through model evaluation. Coordinates modularized stages and handles
    intermediate data passing.
    
    **PIPELINE STAGES**:
    1. Load and prepare raw OHLCV data
    2. Engineer 57+ technical features
    3. Create binary target labels (SL/TP simulation)
    4. Build sliding window sequences
    5. Split chronologically, scale features (no leakage)
    6. Train calibrated XGBoost model
    7. Evaluate on test set with threshold optimization
    8. Save artifacts (model, scaler, metadata)

    **DATA FLOW**:
    Raw OHLCV → Features → Targets → Sequences → Split/Scale → 
    Train/Val/Test → XGBoost Training → Calibration → Evaluation

    **OUTPUTS** (saved to ml/src/models/):
    - sequence_xgb_model.pkl: Calibrated classifier
    - sequence_scaler.pkl: RobustScaler fitted on training
    - sequence_feature_columns.json: Ordered feature names
    - sequence_metadata.json: Configuration
    - sequence_threshold.json: Threshold + win_rate

    Args:
        params: PipelineParams object with all configuration

    Returns:
        Dictionary with evaluation metrics:
        - threshold: Optimal decision threshold
        - win_rate: Precision on test set (expected win rate)
        - precision: True precision
        - recall: True recall
        - f1: F1 score
        - roc_auc: ROC AUC
        - pr_auc: Precision-Recall AUC
        - confusion_matrix: [TN, FP, FN, TP]
        - n_positive_predictions: Count predicted positives

    Raises:
        FileNotFoundError: If data not found
        ValueError: On validation failures
        
    Notes:
        - CRITICAL PRODUCTION CODE: Used for live trading model training
        - All stages validated and tested independently
        - Random seeds fixed for reproducibility
        - Logs to timestamped file with full pipeline summary
    """
    # Set random seeds for reproducibility
    np.random.seed(params.random_state)
    import random
    random.seed(params.random_state)
    
    # Setup directories
    config = PipelineConfig()
    data_dir = config.data_dir
    models_dir = config.outputs_models_dir
    
    # ===== STAGE 1: Load data =====
    df = load_and_prepare_data(data_dir, year_filter=params.year_filter)
    
    # ===== STAGE 2: Engineer features =====
    features = engineer_features_stage(df, window_size=params.window_size, feature_version=params.feature_version)
    
    # ===== STAGE 3: Create targets =====
    targets = create_targets_stage(
        df,
        features,
        atr_multiplier_sl=params.atr_multiplier_sl,
        atr_multiplier_tp=params.atr_multiplier_tp,
        min_hold_minutes=params.min_hold_minutes,
        max_horizon=params.max_horizon,
    )
    
    # ===== STAGE 4: Build sequences =====
    X, y, timestamps = build_sequences_stage(
        features=features,
        targets=targets,
        df_dates=df.index,
        window_size=params.window_size,
        session=params.session,
        custom_start_hour=params.custom_start_hour,
        custom_end_hour=params.custom_end_hour,
        enable_m5_alignment=params.enable_m5_alignment,
        enable_trend_filter=params.enable_trend_filter,
        trend_min_dist_sma200=params.trend_min_dist_sma200,
        trend_min_adx=params.trend_min_adx,
        enable_pullback_filter=params.enable_pullback_filter,
        pullback_max_rsi_m5=params.pullback_max_rsi_m5,
        max_windows=params.max_windows,
    )
    
    # ===== STAGE 5: Split and scale =====
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train, y_val, y_test,
     ts_train, ts_val, ts_test,
     scaler) = split_and_scale_stage(
        X=X,
        y=y,
        timestamps=timestamps,
        year_filter=params.year_filter,
    )
    
    # ===== STAGE 6: Train and evaluate =====
    metrics, model = train_and_evaluate_stage(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_val_scaled=X_val_scaled,
        y_val=y_val,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        ts_test=ts_test,
        random_state=params.random_state,
        min_precision=params.min_precision,
        min_trades=params.min_trades,
        max_trades_per_day=params.max_trades_per_day,
    )
    
    # ===== STAGE 7: Save artifacts =====
    save_model_artifacts(
        model=model,
        scaler=scaler,
        feature_columns=list(features.columns),
        models_dir=models_dir,
        threshold=metrics["threshold"],
        win_rate=metrics["win_rate"],
        window_size=params.window_size,
    )
    
    return metrics



if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_cli_arguments()
    
    # Parse and validate year filter
    try:
        year_filter = parse_and_validate_years(args.years)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Setup logging to unique file in ml/outputs/logs/
    config = PipelineConfig()
    log_filepath = _setup_logging(config, year_filter)
    
    # Create pipeline configuration from CLI args
    params = PipelineParams.from_cli_args(args)
    params.year_filter = year_filter
    
    # Validate configuration
    try:
        params.validate()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        print(f"Configuration Error: {e}")
        sys.exit(1)
    
    # Log training session header with full configuration
    logger.info("\n" + "=" * 80)
    logger.info("SEQUENCE XGBoost TRAINING PIPELINE - XAU/USD 1-Minute Data")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data years: {year_filter if year_filter else 'All available years'}")
    logger.info("\nPipeline Configuration:")
    logger.info(f"  Feature engineering version: {params.feature_version}")
    logger.info(f"  Window size: {params.window_size} candles")
    logger.info(f"  ATR SL multiplier: {params.atr_multiplier_sl}x (OPTION B optimized)")
    logger.info(f"  ATR TP multiplier: {params.atr_multiplier_tp}x (OPTION B optimized)")
    logger.info(f"  Min hold time: {params.min_hold_minutes} minutes")
    logger.info(f"  Max horizon: {params.max_horizon} candles")
    logger.info(f"  Trading session: {params.session}")
    logger.info(f"  Min precision threshold: {params.min_precision}")
    logger.info(f"  M5 alignment filter: {'enabled' if params.enable_m5_alignment else 'disabled'}")
    logger.info(f"  Trend filter: {'enabled' if params.enable_trend_filter else 'disabled'}")
    logger.info(f"  Pullback filter: {'enabled' if params.enable_pullback_filter else 'disabled'}")
    logger.info(f"  Random state: {params.random_state}")
    logger.info("=" * 80 + "\n")

    try:
        # Execute pipeline with validated parameters
        metrics = run_pipeline(params)

        # Print results to console
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - SEQUENCE PIPELINE")
        print("=" * 60)
        print(f"Window Size:       {params.window_size} candles")
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
        logger.info(f"  Window Size:       {params.window_size} candles")
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
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        print(f"Pipeline Error: {str(e)}")
        sys.exit(1)

