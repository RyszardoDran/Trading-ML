#!/usr/bin/env python3
"""Run walk-forward validation on sequence training pipeline.

PURPOSE:
    Execute Point 6 optimization: Walk-Forward CV for realistic model validation.
    Tests if model generalizes or just overfits to static train/test split.

USAGE:
    python ml/scripts/walk_forward_analysis.py --years 2025 [--train-size 500] [--test-size 100]

EXPECTED OUTPUT:
    - Reality check: does WIN_RATE drop when using chronological splits?
    - If so: model may be overfitting or trading on lookahead patterns
    - If not: model generalizes well and has real edge
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.src.data_loading import load_all_years
from ml.src.features.engineer_m5 import aggregate_to_m5, engineer_m5_candle_features
from ml.src.pipelines.sequence_split import split_sequences
from ml.src.sequences import create_sequences, SequenceFilterConfig
from ml.src.targets import make_target
from ml.src.pipelines.walk_forward_validation import walk_forward_validate
from ml.src.utils.risk_config import (
    ATR_PERIOD_M5, SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
    MIN_HOLD_M5_CANDLES, MAX_HORIZON_M5_CANDLES,
    MIN_PRECISION_THRESHOLD, MIN_RECALL_FLOOR,
    USE_COST_SENSITIVE_LEARNING, SAMPLE_WEIGHT_POSITIVE, SAMPLE_WEIGHT_NEGATIVE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            project_root / "ml" / "outputs" / "logs" / "walk_forward_analysis.log",
            mode="a",
        ),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Run walk-forward validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validation for sequence XGBoost model"
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years (e.g., 2023,2024,2025)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=500,
        help="Training window size in samples (default: 500 = ~2500 minutes)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Testing window size in samples (default: 100 = ~500 minutes)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=50,
        help="Step size for rolling window (default: 50 = ~250 minutes)",
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("WALK-FORWARD VALIDATION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Years: {args.years if args.years else 'all available'}")
    logger.info(f"Train window: {args.train_size} samples (~{args.train_size*5} minutes)")
    logger.info(f"Test window: {args.test_size} samples (~{args.test_size*5} minutes)")
    logger.info(f"Step size: {args.step_size} samples (~{args.step_size*5} minutes)")
    logger.info("="*80 + "\n")
    
    try:
        # Parse years
        year_filter = None
        if args.years:
            year_filter = [int(y.strip()) for y in args.years.split(",")]
        
        # Load data
        logger.info("Loading data...")
        data_dir = project_root / "ml" / "src" / "data"
        df = load_all_years(data_dir, year_filter=year_filter)
        logger.info(f"Loaded {len(df):,} M1 candles")
        
        # Aggregate to M5
        logger.info("Aggregating to M5...")
        df_m5 = aggregate_to_m5(df)
        logger.info(f"Aggregated to {len(df_m5):,} M5 candles")
        
        # Engineer features
        logger.info("Engineering features...")
        features = engineer_m5_candle_features(df_m5)
        logger.info(f"Engineered {features.shape[1]} features")
        
        # Create targets
        logger.info("Creating targets...")
        targets = make_target(
            df_m5,
            atr_multiplier_sl=SL_ATR_MULTIPLIER,
            atr_multiplier_tp=TP_ATR_MULTIPLIER,
            min_hold_minutes=MIN_HOLD_M5_CANDLES,
            max_horizon=MAX_HORIZON_M5_CANDLES,
        )
        logger.info(f"Created targets: {(targets==1).sum():,} positive, {(targets==0).sum():,} negative")
        
        # Build sequences
        logger.info("Building sequences...")
        X, y, timestamps = create_sequences(
            features=features,
            targets=targets,
            window_size=50,
            session="london_ny",
            filter_config=SequenceFilterConfig(
                enable_trend_filter=True,
                trend_min_adx=15.0,
                trend_min_dist_sma200=0.0,
                enable_pullback_filter=False,
                enable_m5_alignment=False,
            ),
            max_windows=200000,
        )
        logger.info(f"Built {len(X):,} sequences")
        
        # Run walk-forward validation
        results = walk_forward_validate(
            X=X,
            y=y,
            timestamps=timestamps,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            min_precision=MIN_PRECISION_THRESHOLD,
            min_recall=MIN_RECALL_FLOOR,
            use_cost_sensitive_learning=USE_COST_SENSITIVE_LEARNING,
            sample_weight_positive=SAMPLE_WEIGHT_POSITIVE,
            sample_weight_negative=SAMPLE_WEIGHT_NEGATIVE,
        )
        
        # Print comparison with baseline
        logger.info("\n" + "="*80)
        logger.info("COMPARISON: Single Split vs Walk-Forward CV")
        logger.info("="*80)
        logger.info("Baseline (single 70/15/15 split):     WIN_RATE = 85.71%")
        logger.info(f"Walk-Forward CV ({results['n_folds']} folds): WIN_RATE = {results['win_rate_mean']:.2%} ± {results['win_rate_std']:.2%}")
        
        if results['win_rate_mean'] < 0.80:
            logger.warning("\n⚠️  MAJOR DROP in walk-forward performance!")
            logger.warning("     This suggests model may be overfitting or trading lookahead patterns.")
            logger.warning("     Baseline 85.71% may NOT be reliable for live trading.")
        elif results['win_rate_mean'] > 0.80:
            logger.info("\n✅ Walk-forward performance similar to baseline - good generalization!")
        
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
