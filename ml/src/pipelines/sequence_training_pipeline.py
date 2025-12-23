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
from typing import Dict, Optional, Any

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
    # Prevent duplicate handlers across repeated runs
    if root_logger.handlers:
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
    
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



def run_pipeline(params: PipelineParams) -> Dict[str, Any]:
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
    
    # ===== STAGE 1: Load M1 data =====
    df_m1 = load_and_prepare_data(data_dir, year_filter=params.year_filter)
    
    # ===== STAGE 2: Engineer features (M1→M5 aggregation + feature engineering) =====
    # Returns: (features_m5, df_m5) where df_m5 is aggregated M5 OHLCV for target creation
    # Note: engineer_features_stage now internally aggregates M1→M5 and returns M5 features
    features = engineer_features_stage(df_m1, window_size=params.window_size, feature_version=params.feature_version)
    
    # CRITICAL: Get M5 aggregated data for target creation
    # Use only the date range present in features to avoid leakage
    from ml.src.features.engineer_m5 import aggregate_to_m5
    features_start = features.index.min()
    features_end = features.index.max()
    df_m5 = aggregate_to_m5(df_m1, start_date=str(features_start), end_date=str(features_end))
    
    # ===== STAGE 3: Create targets on M5 timeframe =====
    targets = create_targets_stage(
        df_m5,  # Use M5 aggregated data
        features,
        atr_multiplier_sl=params.atr_multiplier_sl,
        atr_multiplier_tp=params.atr_multiplier_tp,
        min_hold_minutes=params.min_hold_minutes,
        max_horizon=params.max_horizon,
    )
    
    # ===== STAGE 4: Build sequences on M5 timeframe =====
    X, y, timestamps = build_sequences_stage(
        features=features,
        targets=targets,
        df_dates=df_m5.index,  # Use M5 datetime index
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
    split_result = split_and_scale_stage(
        X=X,
        y=y,
        timestamps=timestamps,
        window_size=params.window_size,
        year_filter=params.year_filter,
        use_timeseries_cv=params.use_timeseries_cv,
        cv_folds=params.cv_folds,
        drop_boundary_crossing_sequences=True,
    )
    
    if params.use_timeseries_cv:
        # ===== STAGE 6: Train and evaluate (CV mode) =====
        from ml.src.utils.timeseries_validation import TimeSeriesValidator
        
        validator = TimeSeriesValidator(n_splits=params.cv_folds)
        all_metrics = []
        thresholds: list[float] = []
        val_sizes: list[int] = []
        pooled_val_true: list[np.ndarray] = []
        pooled_val_pred: list[np.ndarray] = []
        last_model = None  # Store last fold's model for artifact saving
        
        for fold_data in split_result:
            fold_num = fold_data['fold']
            X_train_scaled = fold_data['X_train_scaled']
            X_val_scaled = fold_data['X_val_scaled']
            X_test_scaled = fold_data['X_test_scaled']
            y_train = fold_data['y_train']
            y_val = fold_data['y_val']
            y_test = fold_data['y_test']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold_num + 1} / {params.cv_folds}")
            logger.info(f"{'='*60}")
            
            # Train model on this fold
            metrics, model = train_and_evaluate_stage(
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                X_val_scaled=X_val_scaled,  # Now we have validation set for threshold optimization
                y_val=y_val,
                X_test_scaled=X_test_scaled,
                y_test=y_test,
                ts_test=fold_data['timestamps_test'],
                random_state=params.random_state,
                min_precision=params.min_precision,
                min_recall=params.min_recall,
                min_trades=params.min_trades,
                max_trades_per_day=params.max_trades_per_day,
                use_ev_optimization=params.use_ev_optimization,
                use_hybrid_optimization=params.use_hybrid_optimization,
                ev_win_coefficient=params.ev_win_coefficient,
                ev_loss_coefficient=params.ev_loss_coefficient,
                use_cost_sensitive_learning=params.use_cost_sensitive_learning,
                sample_weight_positive=params.sample_weight_positive,
                sample_weight_negative=params.sample_weight_negative,
            )
            
            # Store last fold's model for artifact saving
            last_model = model
            
            # Evaluate fold
            validator.evaluate_fold(
                y_true=y_test,
                y_pred_proba=model.predict_proba(X_test_scaled)[:, 1],
                threshold=metrics['threshold'],
                fold_num=fold_num + 1
            )
            
            all_metrics.append(metrics)
            thresholds.append(metrics['threshold'])
            val_sizes.append(len(y_val))
            pooled_val_true.append(y_val)
            pooled_val_pred.append(model.predict_proba(X_val_scaled)[:, 1])
        
        # ===== STAGE 7: Aggregate CV results & pooled threshold optimization =====
        final_metrics = validator.aggregate_metrics()
        # Pooled validation predictions across folds for robust global threshold
        try:
            from ml.src.training import optimize_threshold_on_val
            y_val_pooled = np.concatenate(pooled_val_true) if pooled_val_true else None
            y_pred_pooled = np.concatenate(pooled_val_pred) if pooled_val_pred else None
            if y_val_pooled is not None and y_pred_pooled is not None and len(y_val_pooled) == len(y_pred_pooled):
                best_thr, best_score = optimize_threshold_on_val(
                    y_true=y_val_pooled,
                    y_pred_proba=y_pred_pooled,
                    min_precision=params.min_precision,
                    min_recall=params.min_recall,
                    use_ev_optimization=params.use_ev_optimization,
                    use_hybrid_optimization=params.use_hybrid_optimization,
                    ev_win_coefficient=params.ev_win_coefficient,
                    ev_loss_coefficient=params.ev_loss_coefficient,
                    min_trades=params.min_trades,
                    timestamps=None,
                    max_trades_per_day=params.max_trades_per_day,
                )
                final_metrics['threshold'] = float(best_thr)
                logger.info(f"CV pooled threshold selected: {best_thr:.4f} (score={best_score:.4f})")
            else:
                raise RuntimeError("Pooled validation data unavailable")
        except Exception as _:
            # Fallback to weighted average of per-fold thresholds
            if len(thresholds) > 0:
                weights = np.array(val_sizes, dtype=float)
                weights = weights / weights.sum()
                final_metrics['threshold'] = float(np.sum(weights * np.array(thresholds)))
                logger.info("Fallback: using weighted average of fold thresholds")
        
        # In CV mode, do not save a single model/scaler; report aggregated metrics only
        
    else:
        # ===== STAGE 6: Train and evaluate (single split mode) =====
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train, y_val, y_test,
         ts_train, ts_val, ts_test,
         scaler) = split_result
        
        final_metrics, model = train_and_evaluate_stage(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            X_val_scaled=X_val_scaled,
            y_val=y_val,
            X_test_scaled=X_test_scaled,
            y_test=y_test,
            ts_test=ts_test,
            random_state=params.random_state,
            min_precision=params.min_precision,
            min_recall=params.min_recall,
            min_trades=params.min_trades,
            max_trades_per_day=params.max_trades_per_day,
            use_ev_optimization=params.use_ev_optimization,
            use_hybrid_optimization=params.use_hybrid_optimization,
            ev_win_coefficient=params.ev_win_coefficient,
            ev_loss_coefficient=params.ev_loss_coefficient,
            use_cost_sensitive_learning=params.use_cost_sensitive_learning,
            sample_weight_positive=params.sample_weight_positive,
            sample_weight_negative=params.sample_weight_negative,
        )
    
    # ===== STAGE 7: Save artifacts =====
    if params.use_timeseries_cv:
        # In CV mode, we don't save a single model - just log aggregated results
        logger.info("CV mode: Skipping model artifacts save (multiple models per fold)")
        logger.info(f"CV Results: precision={final_metrics.get('precision_mean', 'N/A'):.3f}±{final_metrics.get('precision_std', 'N/A'):.3f}, "
                   f"recall={final_metrics.get('recall_mean', 'N/A'):.3f}±{final_metrics.get('recall_std', 'N/A'):.3f}")
    else:
        # Single split mode - save model artifacts
        threshold_strategy = (
            "hybrid" if params.use_hybrid_optimization else "ev" if params.use_ev_optimization else "f1"
        )
        save_model_artifacts(
            model=model,
            scaler=scaler,
            feature_columns=list(features.columns),
            models_dir=models_dir,
            threshold=final_metrics["threshold"],
            win_rate=final_metrics["win_rate"],
            window_size=params.window_size,
            analysis_window_days=7,  # Recommend 7 days for robust indicator calculation
            max_trades_per_day=params.max_trades_per_day,
            min_precision=params.min_precision,
            min_recall=params.min_recall,
            threshold_strategy=threshold_strategy,
        )
    
    return final_metrics



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
    logger.info(f"  ATR SL multiplier: {params.atr_multiplier_sl}x (from risk_config.py)")
    logger.info(f"  ATR TP multiplier: {params.atr_multiplier_tp}x (from risk_config.py)")
    logger.info(f"  Min hold time: {params.min_hold_minutes} M5 candles")
    logger.info(f"  Max horizon: {params.max_horizon} M5 candles")
    logger.info(f"  Trading session: {params.session}")
    if params.use_hybrid_optimization:
        logger.info(f"  Threshold optimization: HYBRID (EV with precision AND recall floors)")
        logger.info(f"    - Min precision: {params.min_precision}")
        logger.info(f"    - Min recall: {params.min_recall}")
        logger.info(f"    - Win coefficient: {params.ev_win_coefficient}")
        logger.info(f"    - Loss coefficient: {params.ev_loss_coefficient}")
    elif params.use_ev_optimization:
        logger.info(f"  Threshold optimization: Expected Value (EV)")
        logger.info(f"    - Win coefficient: {params.ev_win_coefficient}")
        logger.info(f"    - Loss coefficient: {params.ev_loss_coefficient}")
    else:
        logger.info(f"  Threshold optimization: F1-optimized (min_precision={params.min_precision})")
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
        
        # Handle CV vs single split metrics
        if params.use_timeseries_cv:
            print(f"WIN RATE:          {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f} ({metrics['precision_mean']:.2%})")
            print(f"Precision:         {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
            print(f"Recall:            {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
            print(f"F1 Score:          {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
            print(f"ROC-AUC:           {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")
            print(f"PR-AUC:            {metrics['pr_auc_mean']:.4f} ± {metrics['pr_auc_std']:.4f}")
            win_rate_display = metrics['precision_mean']
        else:
            print(f"WIN RATE:          {metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})")
            print(f"Precision:         {metrics['precision']:.4f}")
            print(f"Recall:            {metrics['recall']:.4f}")
            print(f"F1 Score:          {metrics['f1']:.4f}")
            print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
            print(f"PR-AUC:            {metrics['pr_auc']:.4f}")
            win_rate_display = metrics['win_rate']
        
        print("=" * 60)
        print(f"\nWin rate is the precision: when model predicts 'BUY',")
        print(f"it will be correct {win_rate_display:.2%} of the time on test data.")
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
        
        if params.use_timeseries_cv:
            logger.info(f"  WIN RATE:          {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f} ({metrics['precision_mean']:.2%})")
            logger.info(f"  Precision:         {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
            logger.info(f"  Recall:            {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
            logger.info(f"  F1 Score:          {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
            logger.info(f"  ROC-AUC:           {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")
            logger.info(f"  PR-AUC:            {metrics['pr_auc_mean']:.4f} ± {metrics['pr_auc_std']:.4f}")
        else:
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

