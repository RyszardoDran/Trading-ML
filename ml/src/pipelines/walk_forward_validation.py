"""Walk-Forward Cross-Validation for time-series models.

PURPOSE:
    Implement walk-forward analysis for realistic time-series validation.
    Trains on past data, tests on future data (no lookahead bias).
    Rolls window forward and repeats to get robust performance estimates.

KEY DIFFERENCE FROM STANDARD CV:
    - Standard CV: Random shuffle, can look into future
    - Walk-Forward: Chronological, only future data in test set
    
BUSINESS VALUE:
    - Prevents overfitting by simulating real trading: train on history, predict future
    - Detects data leakage between train/test
    - More realistic performance estimates for financial models
    - Industry standard for backtesting
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ml.src.training import evaluate, train_xgb
from ml.src.filters import should_trade, get_adaptive_threshold, filter_predictions_by_regime
from ml.src.utils.risk_config import ENABLE_REGIME_FILTER

logger = logging.getLogger(__name__)


def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    train_size: int = 500,
    test_size: int = 100,
    step_size: int = 50,
    random_state: int = 42,
    min_precision: float = 0.85,
    min_recall: float = 0.15,
    use_cost_sensitive_learning: bool = True,
    sample_weight_positive: float = 3.0,
    sample_weight_negative: float = 1.0,
) -> Dict[str, float]:
    """Perform walk-forward cross-validation on time-series data.
    
    **PURPOSE**: Validate model performance using chronological train/test splits.
    Rolls forward in time, training on past data and testing on future data.
    
    **METHODOLOGY**:
    1. Split 1: Train on [0:train_size], test on [train_size:train_size+test_size]
    2. Split 2: Train on [step_size:train_size+step_size], test on [train_size+step_size:train_size+test_size+step_size]
    3. Repeat until end of data
    
    **KEY PROPERTY**: No lookahead bias - test data is always future relative to training data
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        timestamps: Datetime index (n_samples,) for validation
        train_size: Training window size (samples), default 500 (~2500 minutes = ~41 hours)
        test_size: Testing window size (samples), default 100 (~500 minutes = ~8 hours)
        step_size: Forward step for next fold (samples), default 50 (~250 minutes)
        random_state: Random seed
        min_precision: Minimum precision threshold
        min_recall: Minimum recall floor
        use_cost_sensitive_learning: Enable sample weights
        sample_weight_positive: Weight for positive class
        sample_weight_negative: Weight for negative class
        
    Returns:
        Dictionary with aggregated metrics across all folds:
        - win_rate_mean, win_rate_std: Average and std of win rates
        - precision_mean, precision_std: Average and std of precisions
        - recall_mean, recall_std: Average and std of recalls
        - f1_mean, f1_std: Average and std of F1 scores
        - n_folds: Number of folds used
        - fold_results: List of individual fold metrics
        
    Notes:
        - CRITICAL for time-series: no data contamination between train/test
        - Simulates real trading: train on history, predict future
        - More realistic than standard K-Fold CV
        - May show lower performance than single train/test (due to less training data)
        
    Examples:
        >>> results = walk_forward_validate(
        ...     X, y, timestamps,
        ...     train_size=500,
        ...     test_size=100,
        ...     step_size=50
        ... )
        >>> print(f"Win rate: {results['win_rate_mean']:.2%} ± {results['win_rate_std']:.2%}")
    """
    n_samples = len(X)
    fold_results = []
    fold_num = 0
    
    logger.info("\n" + "="*80)
    logger.info("[POINT 6] WALK-FORWARD CROSS-VALIDATION - Time-Series Validation")
    logger.info("="*80)
    logger.info(f"Data size: {n_samples:,} samples ({len(timestamps)} timestamps)")
    logger.info(f"Train window: {train_size} samples (~{train_size*5} minutes)")
    logger.info(f"Test window: {test_size} samples (~{test_size*5} minutes)")
    logger.info(f"Step size: {step_size} samples (~{step_size*5} minutes)")
    logger.info(f"No lookahead bias: test set is ALWAYS future relative to training set")
    logger.info("="*80 + "\n")
    
    # Walk forward through time
    train_start = 0
    train_end = train_size
    test_end = train_size + test_size
    
    while test_end <= n_samples:
        fold_num += 1
        test_start = train_end
        
        # Get fold data
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        ts_test = timestamps[test_start:test_end]
        
        logger.info(f"Fold {fold_num}:")
        logger.info(f"  Train: [{train_start:,}:{train_end:,}] = {len(X_train):,} samples")
        logger.info(f"         {timestamps[train_start].strftime('%Y-%m-%d %H:%M')} to {timestamps[train_end-1].strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"  Test:  [{test_start:,}:{test_end:,}] = {len(X_test):,} samples")
        logger.info(f"         {timestamps[test_start].strftime('%Y-%m-%d %H:%M')} to {timestamps[test_end-1].strftime('%Y-%m-%d %H:%M')}")
        
        # Scale features (fit ONLY on training data)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        
        # Split training set into train/val (70/30 ratio for early stopping)
        train_val_split = int(len(X_train_scaled) * 0.7)
        X_train_fold = X_train_scaled[:train_val_split]
        y_train_fold = y_train[:train_val_split]
        X_val_fold = X_train_scaled[train_val_split:]
        y_val_fold = y_train[train_val_split:]
        
        # Calculate sample weights if enabled
        sample_weight = None
        if use_cost_sensitive_learning:
            sample_weight = np.where(
                y_train_fold == 1,
                sample_weight_positive,
                sample_weight_negative
            ).astype(np.float32)
        
        # Train model (use fewer estimators for CV speed)
        model = train_xgb(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            random_state=random_state,
            sample_weight=sample_weight,
            n_estimators=20  # Minimal for quick CV testing
        )
        
        # Evaluate on test set
        # NOTE: To apply regime filter during prediction, use filter_predictions_by_regime()
        # after getting model probabilities. This gates predictions based on market conditions.
        metrics = evaluate(
            model,
            X_test_scaled,
            y_test,
            min_precision=min_precision,
            min_recall=min_recall,
            test_timestamps=ts_test,
            use_hybrid_optimization=True,
        )
        
        logger.info(f"  Results: WIN_RATE={metrics['win_rate']:.2%}, "
                   f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                   f"F1={metrics['f1']:.4f}")
        
        fold_results.append(metrics)
        
        # Move window forward
        train_start += step_size
        train_end += step_size
        test_end += step_size
    
    # Aggregate results
    logger.info("\n" + "="*80)
    logger.info("WALK-FORWARD VALIDATION RESULTS")
    logger.info("="*80)
    
    win_rates = [m['win_rate'] for m in fold_results]
    precisions = [m['precision'] for m in fold_results]
    recalls = [m['recall'] for m in fold_results]
    f1_scores = [m['f1'] for m in fold_results]
    roc_aucs = [m['roc_auc'] for m in fold_results]
    
    aggregated = {
        'win_rate_mean': np.mean(win_rates),
        'win_rate_std': np.std(win_rates),
        'win_rate_min': np.min(win_rates),
        'win_rate_max': np.max(win_rates),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'precision_min': np.min(precisions),
        'precision_max': np.max(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'recall_min': np.min(recalls),
        'recall_max': np.max(recalls),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'roc_auc_mean': np.mean(roc_aucs),
        'roc_auc_std': np.std(roc_aucs),
        'n_folds': len(fold_results),
        'fold_results': fold_results,
    }
    
    logger.info(f"\nAggregated Results ({len(fold_results)} folds):\n")
    logger.info(f"  WIN RATE:   {aggregated['win_rate_mean']:.2%} ± {aggregated['win_rate_std']:.2%} "
               f"(range: {aggregated['win_rate_min']:.2%} - {aggregated['win_rate_max']:.2%})")
    logger.info(f"  Precision:  {aggregated['precision_mean']:.4f} ± {aggregated['precision_std']:.4f} "
               f"(range: {aggregated['precision_min']:.4f} - {aggregated['precision_max']:.4f})")
    logger.info(f"  Recall:     {aggregated['recall_mean']:.4f} ± {aggregated['recall_std']:.4f} "
               f"(range: {aggregated['recall_min']:.4f} - {aggregated['recall_max']:.4f})")
    logger.info(f"  F1 Score:   {aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}")
    logger.info(f"  ROC-AUC:    {aggregated['roc_auc_mean']:.4f} ± {aggregated['roc_auc_std']:.4f}")
    logger.info("\n" + "="*80)
    
    return aggregated
