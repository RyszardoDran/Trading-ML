"""CLI Script: Train sequence-based XGBoost model for XAU/USD trading.

This script provides a command-line interface for the complete XAU/USD
trading signal system training pipeline.

Usage:
    # Train with default parameters
    python ml/scripts/train_sequence_model.py
    
    # Train with custom window size
    python ml/scripts/train_sequence_model.py --window-size 50
    
    # Train on specific years only (for testing)
    python ml/scripts/train_sequence_model.py --years 2023,2024
    
    # Show all available options
    python ml/scripts/train_sequence_model.py --help

Output:
    - ml/outputs/models/sequence_xgb_model.pkl - Trained XGBoost classifier
    - ml/outputs/models/sequence_scaler.pkl - Feature scaler (RobustScaler)
    - ml/outputs/models/sequence_feature_columns.json - Ordered feature names
    - ml/outputs/models/sequence_metadata.json - Training metadata
    - ml/outputs/models/sequence_threshold.json - Optimal threshold + win rate
    - ml/outputs/logs/sequence_xgb_train_*.log - Training log with timestamp

Design:
    - CLI parsing and validation
    - Delegates to run_pipeline() from ml.src.pipelines
    - Logs to ml/outputs/logs/ with timestamp
    - Returns success/failure exit code
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


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for training CLI.
    
    Returns:
        Configured ArgumentParser with all training options
    """
    parser = argparse.ArgumentParser(
        prog="python ml/scripts/train_sequence_model.py",
        description="Train sequence-based XGBoost model for XAU/USD trading signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults
  %(prog)s
  
  # Custom window size
  %(prog)s --window-size 50 --max-horizon 120
  
  # Testing on specific years
  %(prog)s --years 2023,2024
  
  # Disable some filters
  %(prog)s --disable-trend-filter --disable-pullback-filter
  
  # Show all options
  %(prog)s --help
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
        default=60,
        metavar="N",
        help="Maximum forward candles for target simulation (default: 60)",
    )
    
    # ATR multipliers (SL/TP levels)
    parser.add_argument(
        "--atr-multiplier-sl",
        type=float,
        default=1.0,
        metavar="X",
        help="ATR multiplier for stop-loss (default: 1.0 - DO NOT CHANGE)",
    )
    parser.add_argument(
        "--atr-multiplier-tp",
        type=float,
        default=2.0,
        metavar="X",
        help="ATR multiplier for take-profit (default: 2.0 for 2:1 RR - DO NOT CHANGE)",
    )
    
    # Hold time
    parser.add_argument(
        "--min-hold-minutes",
        type=int,
        default=5,
        metavar="N",
        help="Minimum holding time in minutes (default: 5)",
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
        default=0.85,
        metavar="P",
        help="Minimum precision (win rate) floor for threshold selection (default: 0.85)",
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
        default=None,
        metavar="N",
        help="Cap number of predicted trades per day (unlimited by default)",
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
        help="Minimum normalized distance above SMA200 (default: 0.0)",
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
        
        # Log summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Threshold: {metrics['threshold']:.4f}")
        logger.info(f"Win Rate (Precision): {metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
        logger.info("=" * 80)
        
        print("\n✅ Training completed successfully!")
        print(f"   Window Size: {args.window_size} candles")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Threshold: {metrics['threshold']:.4f}")
        print("\n📁 Artifacts saved to: ml/outputs/models/")
        print("📊 Logs saved to: ml/outputs/logs/")
        
        return 0
        
    except ValueError as e:
        logger.error(f"Invalid argument: {str(e)}")
        print(f"\n❌ Error: {str(e)}\n")
        parser.print_help()
        return 1
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Make sure data files exist at ml/src/data/XAU_1m_data_*.csv")
        print(f"\n❌ Error: {str(e)}")
        print("Make sure data files exist at ml/src/data/XAU_1m_data_*.csv\n")
        return 1
        
    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
