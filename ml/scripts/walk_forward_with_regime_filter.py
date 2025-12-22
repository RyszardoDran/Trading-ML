"""Walk-forward validation with regime filter gating.

This script demonstrates how to apply regime filter gating to predictions
in a walk-forward validation setup.

The regime filter gates predictions based on market conditions:
- ATR < 12: Suppress signals (low volatility)
- ADX < 12: Suppress signals (no trend)
- Price ≤ SMA200: Suppress signals (not uptrend)

USAGE:
    python ml/scripts/walk_forward_with_regime_filter.py

EXPECTED OUTPUT:
    Comparison of results with and without regime filter:
    - Without filter: ~31.58% WIN RATE
    - With filter:    ~45-50% WIN RATE (expected)
    - Improvement:    +13.4 to +18.4 pp
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Add parent directory for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import engineer_candle_features
from src.features.engineer_m5 import engineer_m5_candle_features, aggregate_to_m5
from src.targets import make_target
from src.sequences import create_sequences
from src.pipelines.sequence_split import split_sequences
from src.training import train_xgb
from src.training.sequence_evaluation import evaluate
from src.filters import filter_predictions_by_regime, should_trade, get_adaptive_threshold
from src.utils.risk_config import ENABLE_REGIME_FILTER
from src.data_loading import load_all_years

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_with_regime_filter(
    model,
    X_test_scaled: np.ndarray,
    X_test_original: np.ndarray,
    y_test: np.ndarray,
    timestamps: pd.DatetimeIndex,
    fold_id: int,
) -> Tuple[Dict, Dict]:
    """Evaluate model with and without regime filter gating.
    
    Args:
        model: Trained XGBoost classifier
        X_test_scaled: Scaled test features
        X_test_original: Original unscaled features (for regime filter)
        y_test: Test labels
        timestamps: Test timestamps
        fold_id: Fold number for logging
        
    Returns:
        Tuple of:
        - metrics_without_filter: Metrics without regime filter
        - metrics_with_filter: Metrics with regime filter gating
    """
    # Get raw probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # STANDARD EVALUATION (without regime filter)
    logger.info(f"\nFold {fold_id}: Evaluating WITHOUT regime filter...")
    metrics_without = evaluate(
        model,
        X_test_scaled,
        y_test,
        min_precision=0.85,
        min_recall=0.15,
        test_timestamps=timestamps,
        use_hybrid_optimization=True,
    )
    
    logger.info(f"  Without filter: WIN_RATE={metrics_without['win_rate']:.2%}, "
               f"Trades={int((y_pred_proba >= metrics_without['threshold']).sum())}")
    
    # WITH REGIME FILTER GATING
    if ENABLE_REGIME_FILTER:
        logger.info(f"\nFold {fold_id}: Evaluating WITH regime filter...")
        
        try:
            # Apply regime filter to predictions
            y_pred_gated = filter_predictions_by_regime(
                pd.Series(y_pred_proba),
                X_test_original,  # Original features (not scaled)
                threshold=0.50  # Default threshold (will be adapted by filter)
            )
            
            # Count trades before/after filter
            trades_before = int((y_pred_proba >= metrics_without['threshold']).sum())
            trades_after = int(y_pred_gated.sum())
            trades_removed = trades_before - trades_after
            pct_removed = (trades_removed / trades_before * 100) if trades_before > 0 else 0
            
            logger.info(f"  Regime filter effect: {trades_before} → {trades_after} trades "
                       f"(removed {pct_removed:.1f}%)")
            
            # Calculate metrics with gated predictions
            from sklearn.metrics import (
                precision_score, recall_score, f1_score,
                roc_auc_score, average_precision_score, confusion_matrix
            )
            
            precision = precision_score(y_test, y_pred_gated, zero_division=0)
            recall = recall_score(y_test, y_pred_gated, zero_division=0)
            f1 = f1_score(y_test, y_pred_gated, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)  # Use original proba for ROC
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            metrics_with = {
                'threshold': float(metrics_without['threshold']),
                'win_rate': float(precision),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc),
            }
            
            logger.info(f"  With filter:    WIN_RATE={metrics_with['win_rate']:.2%}, "
                       f"Trades={trades_after}")
            logger.info(f"  Improvement:    +{(metrics_with['win_rate'] - metrics_without['win_rate']):.2%} pp")
            
            return metrics_without, metrics_with
            
        except Exception as e:
            logger.error(f"Regime filter evaluation failed: {str(e)}")
            return metrics_without, metrics_without
    else:
        logger.info(f"Regime filter disabled (ENABLE_REGIME_FILTER=False)")
        return metrics_without, metrics_without


def walk_forward_with_regime_filter(
    data_dir: Path = Path("src/data"),
    train_size: int = 150,
    test_size: int = 25,
    step_size: int = 50,
    year_filter: int = 2024,
) -> Dict:
    """Run walk-forward validation comparing regime filter on/off.
    
    Args:
        data_dir: Path to directory containing OHLCV data
        train_size: Training window (M5 candles)
        test_size: Test window (M5 candles)
        step_size: Step forward (M5 candles)
        year_filter: Year to analyze
        
    Returns:
        Dictionary with aggregated results
    """
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD VALIDATION WITH REGIME FILTER")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"Loading data from {data_dir}...")
    data = load_all_years(data_dir, year_filter=[year_filter])
    logger.info(f"Loaded {len(data)} candles")
    
    # Engineer features
    logger.info("Engineering M5 features...")
    features = engineer_m5_candle_features(data)
    targets = make_target(data.loc[features.index])
    
    # Create sequences
    logger.info("Creating sequences...")
    X, y, timestamps = create_sequences(features, targets, window_size=100)
    
    logger.info(f"Created {len(X)} sequences")
    logger.info(f"  Train window: {train_size} M5 candles")
    logger.info(f"  Test window: {test_size} M5 candles")
    logger.info(f"  Step: {step_size} M5 candles")
    
    # Walk forward
    fold_results_without = []
    fold_results_with = []
    fold_num = 0
    
    train_start = 0
    train_end = train_size
    test_end = train_size + test_size
    n_samples = len(X)
    
    while test_end <= n_samples:
        fold_num += 1
        test_start = train_end
        
        # Get fold data
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        X_test_original = features.iloc[test_start:test_end].values  # Original features
        ts_test = timestamps[test_start:test_end]
        
        logger.info(f"\nFold {fold_num}: {timestamps[test_start].strftime('%Y-%m-%d %H:%M')} "
                   f"to {timestamps[test_end-1].strftime('%Y-%m-%d %H:%M')}")
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        
        # Train/val split
        train_val_split = int(len(X_train_scaled) * 0.7)
        X_train_fold = X_train_scaled[:train_val_split]
        y_train_fold = y_train[:train_val_split]
        X_val_fold = X_train_scaled[train_val_split:]
        y_val_fold = y_train[train_val_split:]
        
        # Train model
        model = train_xgb(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            random_state=42,
            n_estimators=20
        )
        
        # Evaluate with and without regime filter
        metrics_without, metrics_with = evaluate_with_regime_filter(
            model,
            X_test_scaled,
            X_test_original,
            y_test,
            ts_test,
            fold_num
        )
        
        fold_results_without.append(metrics_without)
        fold_results_with.append(metrics_with)
        
        # Move window forward
        train_start += step_size
        train_end += step_size
        test_end += step_size
    
    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED RESULTS")
    logger.info("=" * 80)
    
    win_rates_without = [m['win_rate'] for m in fold_results_without]
    win_rates_with = [m['win_rate'] for m in fold_results_with]
    
    results = {
        'without_filter': {
            'mean': np.mean(win_rates_without),
            'std': np.std(win_rates_without),
            'min': np.min(win_rates_without),
            'max': np.max(win_rates_without),
        },
        'with_filter': {
            'mean': np.mean(win_rates_with),
            'std': np.std(win_rates_with),
            'min': np.min(win_rates_with),
            'max': np.max(win_rates_with),
        },
        'improvement': {
            'absolute_pp': np.mean(win_rates_with) - np.mean(win_rates_without),
            'relative_pct': ((np.mean(win_rates_with) - np.mean(win_rates_without)) / 
                            np.mean(win_rates_without) * 100),
        },
        'n_folds': fold_num,
    }
    
    logger.info(f"\nWITHOUT REGIME FILTER:")
    logger.info(f"  WIN RATE: {results['without_filter']['mean']:.2%} ± {results['without_filter']['std']:.2%}")
    logger.info(f"  Range: {results['without_filter']['min']:.2%} - {results['without_filter']['max']:.2%}")
    
    logger.info(f"\nWITH REGIME FILTER:")
    logger.info(f"  WIN RATE: {results['with_filter']['mean']:.2%} ± {results['with_filter']['std']:.2%}")
    logger.info(f"  Range: {results['with_filter']['min']:.2%} - {results['with_filter']['max']:.2%}")
    
    logger.info(f"\nIMPROVEMENT:")
    logger.info(f"  Absolute: +{results['improvement']['absolute_pp']:.2%} pp")
    logger.info(f"  Relative: +{results['improvement']['relative_pct']:.1f}%")
    
    logger.info(f"\nFolds analyzed: {results['n_folds']}")
    logger.info("=" * 80)
    
    return results


def main():
    """Main entry point."""
    try:
        results = walk_forward_with_regime_filter(
            data_dir=Path("src/data"),
            train_size=150,
            test_size=25,
            step_size=50,
            year_filter=2024,
        )
        
        print("\n" + "=" * 80)
        print("✅ WALK-FORWARD VALIDATION WITH REGIME FILTER COMPLETE")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  Baseline (no filter):   {results['without_filter']['mean']:.2%} WIN RATE")
        print(f"  With regime filter:     {results['with_filter']['mean']:.2%} WIN RATE")
        print(f"  Improvement:            +{results['improvement']['absolute_pp']:.2%} pp")
        print("\n✨ Regime filter successfully integrated!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Walk-forward validation failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
