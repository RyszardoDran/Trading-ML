"""CLI Script: Train sequence-based XGBoost model for XAU/USD trading.

**PURPOSE**: Production-ready CLI for training XAU/USD day trading signal generator.
Implements EARS specification requirements: P(TP)≥70%, 1:2 RR, >10 min hold time.

**CRITICAL PARAMETERS** (Production-Aligned):
    - Window size: 60 candles (~1 hour)
    - Min hold time: 10 minutes (EARS requirement)
    - Min precision: 70% (P(TP)≥70% from EARS)
    - Risk:Reward: 1:1 (0.5 ATR SL, 1.0 ATR TP - for achievability)
    - Session: London+NY (highest liquidity overlap)
    - Max trades/day: 5 (risk cap)
    - Technical filters: M5 alignment, trend (SMA200/ADX), pullback (RSI_M5)

**USAGE** (Production Scenarios):
    # Default production (recommended)
    python ml/scripts/train_sequence_model.py
    
    # Production on recent data
    python ml/scripts/train_sequence_model.py --years 2024
    
    # 2-year backtest for validation
    python ml/scripts/train_sequence_model.py --years 2023,2024
    
    # Show all available options
    python ml/scripts/train_sequence_model.py --help

**OUTPUT ARTIFACTS**:
    - ml/outputs/models/sequence_xgb_model.pkl - Trained XGBoost classifier
    - ml/outputs/models/sequence_scaler.pkl - Feature scaler (RobustScaler)
    - ml/outputs/models/sequence_feature_columns.json - Ordered feature names
    - ml/outputs/models/sequence_metadata.json - Training metadata + params
    - ml/outputs/models/sequence_threshold.json - Optimal threshold + win rate
    - ml/outputs/logs/sequence_xgb_train_*.log - Detailed training log

**QUALITY GATES** (Must satisfy before production deployment):
    ✅ Win Rate (Precision) ≥ 70%
    ✅ Recall ≥ 20%
    ✅ F1 Score ≥ 0.60
    ✅ ROC-AUC ≥ 0.75
    ✅ Min 30-50 trades on test period
    ✅ Avg hold time ≥ 10 minutes
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repo root is in path for imports
_script_dir = Path(__file__).parent
_ml_dir = _script_dir.parent
_repo_dir = _ml_dir.parent
sys.path.insert(0, str(_repo_dir))

from ml.src.pipelines.sequence_training_pipeline import run_pipeline
from ml.src.pipeline_config_extended import PipelineParams


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for training CLI.
    
    Returns:
        Configured ArgumentParser with all training options
    """
    parser = argparse.ArgumentParser(
        prog="python ml/scripts/train_sequence_model.py",
        description="Train sequence-based XGBoost model for XAU/USD day trading signals (EARS-aligned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PRODUCTION EXAMPLES:
  # Default production training (latest year, all params optimized)
  %(prog)s
  
  # Production validation (2-year backtest)
  %(prog)s --years 2023,2024
  
  # Training on recent data only (2024)
  %(prog)s --years 2024
  
  # Lighter testing (quick iteration, smaller window)
  %(prog)s --window-size 50 --session london
  
  # Disable some filters (for comparison)
  %(prog)s --disable-trend-filter --disable-pullback-filter
  
  # Show all available options
  %(prog)s --help
  
CRITICAL (DO NOT CHANGE):
  - ATR Multipliers: SL=0.5, TP=1.0 (1:1 Risk:Reward ratio for achievability)
  - These define the ground truth for model training
        """
    )
    
    # Core pipeline arguments
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        metavar="N",
        help="Number of previous candles for sequence input (default: 60)",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=120,
        metavar="N",
        help="Maximum forward candles for target simulation (default: 120)",
    )
    
    # ATR multipliers (SL/TP levels)
    parser.add_argument(
        "--atr-multiplier-sl",
        type=float,
        default=0.5,
        metavar="X",
        help="ATR multiplier for stop-loss (default: 0.5 for easier achievability)",
    )
    parser.add_argument(
        "--atr-multiplier-tp",
        type=float,
        default=1.0,
        metavar="X",
        help="ATR multiplier for take-profit (default: 1.0 for easier achievability)",
    )
    
    # Hold time
    parser.add_argument(
        "--min-hold-minutes",
        type=int,
        default=10,
        metavar="N",
        help="Minimum holding time in minutes (default: 10 - EARS requirement)",
    )
    
    # Data selection
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        metavar="YEARS",
        help="Comma-separated years to train on (e.g., '2023,2024'). "
             "Leave empty for all available years.",
    )
    
    # Trading session
    parser.add_argument(
        "--session",
        type=str,
        default="london_ny",
        choices=["london", "ny", "asian", "london_ny", "all", "custom"],
        metavar="{london,ny,asian,london_ny,all,custom}",
        help="Trading session to filter (default: london_ny)",
    )
    parser.add_argument(
        "--custom-start-hour",
        type=int,
        default=None,
        metavar="H",
        help="Start hour for custom session (0-23, only with --session custom)",
    )
    parser.add_argument(
        "--custom-end-hour",
        type=int,
        default=None,
        metavar="H",
        help="End hour for custom session (0-23, only with --session custom)",
    )
    
    # Data limits
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200000,
        metavar="N",
        help="Maximum number of training windows to avoid OOM (default: 200,000)",
    )
    
    # Threshold selection
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.70,
        metavar="P",
        help="Minimum precision (win rate) floor for threshold selection (default: 0.70 - EARS)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.15,
        metavar="R",
        help="Minimum recall floor for threshold selection (default: 0.15)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        metavar="N",
        help="Minimum number of trades for threshold to be considered (dynamic by default)",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=5,
        metavar="N",
        help="Cap number of predicted trades per day (default: 5 - risk management)",
    )
    
    # Threshold optimization strategy
    parser.add_argument(
        "--use-ev-optimization",
        action="store_true",
        help="Use Expected Value optimization instead of F1 (default: False)",
    )
    parser.add_argument(
        "--use-hybrid-optimization",
        action="store_true",
        default=True,
        help="Use hybrid EV + precision/recall constraints (default: True)",
    )
    parser.add_argument(
        "--ev-win-coefficient",
        type=float,
        default=1.0,
        metavar="W",
        help="Profit multiplier for correct predictions in EV optimization (default: 1.0)",
    )
    parser.add_argument(
        "--ev-loss-coefficient",
        type=float,
        default=-1.0,
        metavar="L",
        help="Loss multiplier for incorrect predictions in EV optimization (default: -1.0)",
    )
    
    # Cost-sensitive learning
    parser.add_argument(
        "--use-cost-sensitive-learning",
        action="store_true",
        help="Use cost-sensitive learning with sample weights (default: False)",
    )
    parser.add_argument(
        "--sample-weight-positive",
        type=float,
        default=3.0,
        metavar="W",
        help="Sample weight for positive class (default: 3.0)",
    )
    parser.add_argument(
        "--sample-weight-negative",
        type=float,
        default=1.0,
        metavar="W",
        help="Sample weight for negative class (default: 1.0)",
    )
    
    # Time Series Cross-Validation
    parser.add_argument(
        "--use-timeseries-cv",
        action="store_true",
        help="Use Time Series Cross-Validation instead of single split (default: False)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of CV folds (default: 5)",
    )
    
    # Filters - M5 alignment
    parser.add_argument(
        "--skip-m5-alignment",
        action="store_true",
        help="Disable M5 candle alignment filter (enabled by default)",
    )
    
    # Filters - Trend filter
    parser.add_argument(
        "--disable-trend-filter",
        action="store_true",
        help="Disable trend filter (SMA200 + ADX) (enabled by default)",
    )
    parser.add_argument(
        "--trend-min-dist-sma200",
        type=float,
        default=0.0,
        metavar="D",
        help="Minimum normalized distance above SMA200 in pips (default: 0.0)",
    )
    parser.add_argument(
        "--trend-min-adx",
        type=float,
        default=15.0,
        metavar="X",
        help="Minimum ADX for trend filter (default: 15.0)",
    )
    
    # Filters - Pullback filter
    parser.add_argument(
        "--disable-pullback-filter",
        action="store_true",
        help="Disable RSI_M5 pullback guard (enabled by default)",
    )
    parser.add_argument(
        "--pullback-max-rsi-m5",
        type=float,
        default=75.0,
        metavar="R",
        help="Maximum RSI_M5 for pullback filter (default: 75.0)",
    )
    
    # Reproducibility
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for reproducibility (default: 42)",
    )
    
    # Feature version
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v1",
        metavar="VERSION",
        help="Feature engineering version (default: v1)",
    )
    
    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (show all logs to console)",
    )
    
    return parser


def parse_year_filter(years_str: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated years string into list of integers.
    
    Args:
        years_str: Comma-separated string of years (e.g., "2023,2024")
        
    Returns:
        List of years as integers, or None if years_str is None
        
    Raises:
        ValueError: If years cannot be parsed
    """
    if years_str is None:
        return None
    
    try:
        years = [int(y.strip()) for y in years_str.split(',')]
        return years
    except ValueError as e:
        raise ValueError(
            f"Invalid years format: '{years_str}'. "
            f"Expected comma-separated integers like '2023,2024'. "
            f"Error: {str(e)}"
        )


def main() -> int:
    """Main entry point for training CLI.
    
    Returns:
        Exit code: 0 for success, non-zero for failure
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Log startup message
        logger.info("\n" + "=" * 80)
        logger.info("XAU/USD SEQUENCE XGBoost TRAINING - CLI ENTRY POINT")
        logger.info("=" * 80)
        
        # Parse year filter
        year_filter = parse_year_filter(args.years)
        if year_filter:
            logger.info(f"Training years: {year_filter}")
        else:
            logger.info("Training years: All available")
        
        # Log configuration summary
        logger.info(f"Configuration:")
        logger.info(f"  Window size: {args.window_size} candles")
        logger.info(f"  Max horizon: {args.max_horizon} candles")
        logger.info(f"  ATR SL multiplier: {args.atr_multiplier_sl}x")
        logger.info(f"  ATR TP multiplier: {args.atr_multiplier_tp}x")
        logger.info(f"  Min hold time: {args.min_hold_minutes} minutes")
        logger.info(f"  Trading session: {args.session}")
        logger.info(f"  M5 alignment: {'enabled' if not args.skip_m5_alignment else 'disabled'}")
        logger.info(f"  Trend filter: {'enabled' if not args.disable_trend_filter else 'disabled'}")
        logger.info(f"  Pullback filter: {'enabled' if not args.disable_pullback_filter else 'disabled'}")
        logger.info(f"  Min precision threshold: {args.min_precision}")
        logger.info(f"  Random state: {args.random_state}")
        logger.info("=" * 80 + "\n")
        
        # Call run_pipeline with all arguments
        logger.info("Starting training pipeline...")
        
        # Create PipelineParams object from CLI args
        params = PipelineParams.from_cli_args(args)
        params.year_filter = year_filter
        
        # Validate configuration
        try:
            params.validate()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            print(f"Configuration Error: {e}")
            return 1
        
        # Execute pipeline
        metrics = run_pipeline(params)
        
        # Log summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        if params.use_timeseries_cv:
            # CV mode - show aggregated metrics
            logger.info("CV Mode Results:")
            logger.info(f"Precision: {metrics.get('precision_mean', 'N/A'):.3f}±{metrics.get('precision_std', 'N/A'):.3f}")
            logger.info(f"Recall: {metrics.get('recall_mean', 'N/A'):.3f}±{metrics.get('recall_std', 'N/A'):.3f}")
            logger.info(f"F1 Score: {metrics.get('f1_mean', 'N/A'):.3f}±{metrics.get('f1_std', 'N/A'):.3f}")
            logger.info(f"ROC-AUC: {metrics.get('roc_auc_mean', 'N/A'):.3f}±{metrics.get('roc_auc_std', 'N/A'):.3f}")
            logger.info(f"PR-AUC: {metrics.get('pr_auc_mean', 'N/A'):.3f}±{metrics.get('pr_auc_std', 'N/A'):.3f}")
            
            print("\nCV Training completed successfully!")
            print(f"   CV Folds: {args.cv_folds}")
            print(f"   Avg Precision: {metrics.get('precision_mean', 0):.1%}")
            print(f"   Avg Recall: {metrics.get('recall_mean', 0):.1%}")
        else:
            # Single split mode - show individual metrics
            logger.info(f"Threshold: {metrics['threshold']:.4f}")
            logger.info(f"Win Rate (Precision): {metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
            
            print("\nTraining completed successfully!")
            print(f"   Window Size: {args.window_size} candles")
            print(f"   Win Rate: {metrics['win_rate']:.2%}")
            print(f"   Threshold: {metrics['threshold']:.4f}")
            print("\nArtifacts saved to: ml/outputs/models/")
        
        logger.info("=" * 80)
        print("Logs saved to: ml/outputs/logs/")
        
        return 0
        
    except ValueError as e:
        logger.error(f"Invalid argument: {str(e)}")
        print(f"\nError: {str(e)}\n")
        parser.print_help()
        return 1
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Make sure data files exist at ml/src/data/XAU_1m_data_*.csv")
        print(f"\nError: {str(e)}")
        print("Make sure data files exist at ml/src/data/XAU_1m_data_*.csv\n")
        return 1
        
    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        print(f"\nUnexpected error: {str(e)}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
