#!/usr/bin/env python3
"""Simple regime filter test on single fold."""

import sys
from pathlib import Path

# Add parent directory for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, confusion_matrix

from src.data_loading import load_all_years
from src.features.engineer_m5 import engineer_m5_candle_features, aggregate_to_m5
from src.targets import make_target
from src.sequences import create_sequences, SequenceFilterConfig
from src.pipelines.sequence_split import split_sequences
from src.training import train_xgb
from src.training.sequence_evaluation import evaluate
from src.filters import filter_predictions_by_regime
from src.utils.risk_config import ENABLE_REGIME_FILTER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_regime_filter_test():
    """Test regime filter on single fold."""
    logger.info("\n" + "=" * 80)
    logger.info("SIMPLE REGIME FILTER TEST")
    logger.info("=" * 80)
    
    # 1. Load data (only 2 months for quick test)
    logger.info("\n[1] LOADING DATA")
    data_dir = Path("src/data")
    data = load_all_years(data_dir, year_filter=[2024])
    # Take only last 2 months for speed
    data = data[data.index.month.isin([11, 12])]
    logger.info(f"✅ Loaded {len(data):,} M1 candles (2 months of 2024)")
    
    # 2. Engineer features on M5
    logger.info("\n[2] ENGINEERING M5 FEATURES")
    df_m5 = aggregate_to_m5(data)
    logger.info(f"✅ Aggregated to {len(df_m5):,} M5 candles")
    
    features_m5 = engineer_m5_candle_features(data)
    logger.info(f"✅ Engineered {features_m5.shape[1]} features")
    
    # 3. Create targets
    logger.info("\n[3] CREATING TARGETS")
    targets_m5 = make_target(df_m5)
    logger.info(f"✅ Created targets: {(targets_m5 == 1).sum():,} positive, {(targets_m5 == 0).sum():,} negative")
    
    # 4. Create sequences
    logger.info("\n[4] CREATING SEQUENCES")
    # Disable all filters for simple demo
    filter_config = SequenceFilterConfig(
        enable_m5_alignment=False,
        enable_trend_filter=False,
        enable_pullback_filter=False
    )
    X, y, timestamps = create_sequences(
        features_m5, 
        targets_m5, 
        window_size=100,
        session="all",
        filter_config=filter_config
    )
    logger.info(f"✅ Created {len(X):,} sequences")
    logger.info(f"   Positive class: {y.sum():,} ({y.mean():.1%})")
    
    # 5. Split into train/test (last 20% for test)
    logger.info("\n[5] SPLITTING DATA")
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    ts_test = timestamps[split_point:]
    
    logger.info(f"✅ Train: {len(X_train):,} sequences")
    logger.info(f"   Test: {len(X_test):,} sequences")
    
    # 6. Train model
    logger.info("\n[6] TRAINING MODEL")
    model = train_xgb(X_train, y_train, X_test, y_test)
    logger.info(f"✅ Model trained")
    
    # 7. Evaluate without filter
    logger.info("\n[7] EVALUATION WITHOUT FILTER")
    metrics_without = evaluate(model, X_test, y_test)
    win_rate_without = metrics_without.get('win_rate', metrics_without.get('precision', 0))
    logger.info(f"   WIN RATE (no filter): {win_rate_without:.2%}")
    logger.info(f"   Precision: {metrics_without.get('precision', 0):.2%}")
    logger.info(f"   Recall: {metrics_without.get('recall', 0):.2%}")
    
    # 8. Get predictions
    logger.info("\n[8] GETTING PREDICTIONS")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    logger.info(f"✅ Predictions: {y_pred.sum():,} positive ({y_pred.mean():.1%})")
    
    # 9. Apply regime filter
    logger.info("\n[9] APPLYING REGIME FILTER")
    if ENABLE_REGIME_FILTER:
        y_pred_gated = filter_predictions_by_regime(
            predictions=y_pred_proba,
            ohlcv_data=df_m5.loc[ts_test],  # OHLCV for test period
            X_test_original=X_test,
            threshold=0.50
        )
        
        trades_before = int(y_pred.sum())
        trades_after = int(y_pred_gated.sum())
        trades_removed = trades_before - trades_after
        pct_removed = (trades_removed / trades_before * 100) if trades_before > 0 else 0
        
        logger.info(f"   Trades before filter: {trades_before}")
        logger.info(f"   Trades after filter: {trades_after}")
        logger.info(f"   Trades removed: {trades_removed} ({pct_removed:.1f}%)")
        
        # Calculate metrics
        win_rate_with = precision_score(y_test, y_pred_gated, zero_division=0)
        improvement = win_rate_with - win_rate_without
        
        logger.info(f"\n   WIN RATE (with filter): {win_rate_with:.2%}")
        logger.info(f"   Improvement: +{improvement:.2%} pp")
    else:
        logger.info("   Regime filter DISABLED")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        simple_regime_filter_test()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        exit(1)
