"""CLI argument parsing for sequence training pipeline.

Purpose:
    Encapsulate all command-line argument definitions and parsing logic.
    Provides a clean interface for building CLI with all pipeline parameters.
    
Usage:
    >>> from ml.src.pipeline_cli import parse_cli_arguments
    >>> args = parse_cli_arguments()
    >>> print(args.window_size)
"""

import argparse
from typing import Optional

# Import canonical parameters from risk_config
from .utils.risk_config import (
    ENABLE_M5_ALIGNMENT,
    ENABLE_PULLBACK_FILTER,
    ENABLE_TREND_FILTER,
    EV_LOSS_COEFFICIENT,
    EV_WIN_COEFFICIENT,
    MAX_HORIZON_M5_CANDLES,
    MAX_TRADES_PER_DAY,
    MIN_HOLD_M5_CANDLES,
    MIN_PRECISION_THRESHOLD,
    MIN_RECALL_FLOOR,
    MIN_TRADES_PER_TEST,
    PULLBACK_MAX_RSI_M5,
    SAMPLE_WEIGHT_NEGATIVE,
    SAMPLE_WEIGHT_POSITIVE,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
    TREND_MIN_ADX,
    TREND_MIN_DIST_SMA200,
    USE_COST_SENSITIVE_LEARNING,
    USE_EV_OPTIMIZATION,
    USE_HYBRID_OPTIMIZATION,
    WINDOW_SIZE,
)

# Compute inversion flags for filters based on risk_config defaults
# (CLI uses --disable-X flags which are "store_true", so we invert the risk_config values)
_SKIP_M5_ALIGNMENT_DEFAULT = not ENABLE_M5_ALIGNMENT
_DISABLE_TREND_FILTER_DEFAULT = not ENABLE_TREND_FILTER
_DISABLE_PULLBACK_FILTER_DEFAULT = not ENABLE_PULLBACK_FILTER


def parse_cli_arguments() -> argparse.Namespace:
    """Parse command-line arguments for sequence XGBoost training pipeline.
    
    Defines all CLI arguments for pipeline configuration including:
    - Window size and horizon settings
    - ATR multipliers for SL/TP
    - Session and time filters
    - Trend and pullback filters
    - Training data selection and constraints
    - Threshold optimization parameters
    
    Returns:
        argparse.Namespace containing parsed arguments with defaults applied
        
    Raises:
        SystemExit: If invalid arguments provided (argparse behavior)
        
    Notes:
        - All arguments have sensible defaults matching production settings
        - Boolean flags use action='store_true' (default False)
        - Year filter expects comma-separated integers
        - Session choices restricted to valid trading sessions
        
    Examples:
        >>> args = parse_cli_arguments()
        >>> # args.window_size == 60, args.atr_multiplier_sl == 1.0, etc.
    """
    parser = argparse.ArgumentParser(
        description="Sequence-based XGBoost training pipeline for XAU/USD 1-minute data"
    )
    
    # ===== Core pipeline parameters =====
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"Number of previous candles to use as input features (default: {WINDOW_SIZE} - from risk_config.py)",
    )
    parser.add_argument(
        "--atr-multiplier-sl",
        type=float,
        default=SL_ATR_MULTIPLIER,
        help=f"ATR multiplier for stop-loss level (default: {SL_ATR_MULTIPLIER} - from risk_config.py)",
    )
    parser.add_argument(
        "--atr-multiplier-tp",
        type=float,
        default=TP_ATR_MULTIPLIER,
        help=f"ATR multiplier for take-profit level (default: {TP_ATR_MULTIPLIER} - from risk_config.py)",
    )
    parser.add_argument(
        "--min-hold-minutes",
        type=int,
        default=MIN_HOLD_M5_CANDLES,
        help=f"Minimum holding time in M5 candles for target calculation (default: {MIN_HOLD_M5_CANDLES})",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=MAX_HORIZON_M5_CANDLES,
        help=f"Maximum forward M5 candles to simulate for target labels (default: {MAX_HORIZON_M5_CANDLES})",
    )
    
    # ===== Reproducibility =====
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42 - DO NOT CHANGE)",
    )
    
    # ===== Data selection =====
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years to load (e.g., '2023,2024' for testing). None=all available",
    )
    
    # ===== Session and time filters =====
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
        help="Start hour for custom session (0-23, only used if --session=custom)",
    )
    parser.add_argument(
        "--custom-end-hour",
        type=int,
        default=None,
        help="End hour for custom session (0-23, only used if --session=custom)",
    )
    
    # ===== Sequence and window constraints =====
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200000,
        help="Maximum number of training windows to keep (prevent OOM, default: 200,000)",
    )
    
    # ===== Time Series Cross-Validation =====
    parser.add_argument(
        "--use-timeseries-cv",
        action="store_true",
        default=False,
        help="Use Time Series Cross-Validation instead of single split (default: False)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for Time Series CV (default: 5)",
    )
    
    # ===== Threshold optimization =====
    parser.add_argument(
        "--min-precision",
        type=float,
        default=MIN_PRECISION_THRESHOLD,
        help=f"Minimum precision (win rate) floor for threshold selection (default: {MIN_PRECISION_THRESHOLD})",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=MIN_TRADES_PER_TEST,
        help=f"Minimum number of predicted positives for threshold (default: {MIN_TRADES_PER_TEST})",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=MAX_TRADES_PER_DAY,
        help=f"Cap number of predicted trades per day after thresholding (default: {MAX_TRADES_PER_DAY})",
    )
    
    # ===== Expected Value (EV) Optimization =====
    parser.add_argument(
        "--use-ev-optimization",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=USE_EV_OPTIMIZATION,
        help="Use Expected Value optimization for threshold selection (default: False from risk_config.py)",
    )
    parser.add_argument(
        "--use-hybrid-optimization",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=USE_HYBRID_OPTIMIZATION,
        help="Use Hybrid optimization: EV with precision AND recall floors (RECOMMENDED, default: True)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=MIN_RECALL_FLOOR,
        help=f"Minimum recall floor for hybrid optimization (default: {MIN_RECALL_FLOOR})",
    )
    parser.add_argument(
        "--ev-win-coefficient",
        type=float,
        default=EV_WIN_COEFFICIENT,
        help=f"Profit multiplier for correct predictions in EV mode (default: {EV_WIN_COEFFICIENT})",
    )
    parser.add_argument(
        "--ev-loss-coefficient",
        type=float,
        default=EV_LOSS_COEFFICIENT,
        help=f"Loss multiplier for incorrect predictions in EV mode (default: {EV_LOSS_COEFFICIENT})",
    )
    
    # ===== Cost-Sensitive Learning (POINT 1) =====
    parser.add_argument(
        "--use-cost-sensitive-learning",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=USE_COST_SENSITIVE_LEARNING,
        help=f"Enable cost-sensitive learning: penalize FP more than FN (default: {USE_COST_SENSITIVE_LEARNING} from risk_config.py)",
    )
    parser.add_argument(
        "--sample-weight-positive",
        type=float,
        default=SAMPLE_WEIGHT_POSITIVE,
        help=f"Weight for positive class samples (True Positives get more 'vote', default: {SAMPLE_WEIGHT_POSITIVE})",
    )
    parser.add_argument(
        "--sample-weight-negative",
        type=float,
        default=SAMPLE_WEIGHT_NEGATIVE,
        help=f"Weight for negative class samples (baseline, default: {SAMPLE_WEIGHT_NEGATIVE})",
    )
    
    # ===== Feature engineering version =====
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Feature engineering version: v1 (57 original features) or v2 (15 multi-timeframe features, default: v1)",
    )
    
    # ===== M5 alignment filter =====
    parser.add_argument(
        "--skip-m5-alignment",
        action="store_true",
        default=_SKIP_M5_ALIGNMENT_DEFAULT,
        help=f"Disable M5 candle close alignment filter (default: {'disabled' if _SKIP_M5_ALIGNMENT_DEFAULT else 'enabled'} from risk_config.py)",
    )
    
    # ===== Trend filter =====
    parser.add_argument(
        "--disable-trend-filter",
        action="store_true",
        default=_DISABLE_TREND_FILTER_DEFAULT,
        help=f"Disable trend filter requiring price above SMA200 and ADX threshold (default: {'disabled' if _DISABLE_TREND_FILTER_DEFAULT else 'enabled'} from risk_config.py)",
    )
    parser.add_argument(
        "--trend-min-dist-sma200",
        type=float,
        default=TREND_MIN_DIST_SMA200,
        help=f"Minimum normalized distance above SMA200 when trend filter enabled (default: {TREND_MIN_DIST_SMA200})",
    )
    parser.add_argument(
        "--trend-min-adx",
        type=float,
        default=TREND_MIN_ADX,
        help=f"Minimum ADX threshold when trend filter enabled (default: {TREND_MIN_ADX})",
    )
    
    # ===== Pullback filter =====
    parser.add_argument(
        "--disable-pullback-filter",
        action="store_true",
        default=_DISABLE_PULLBACK_FILTER_DEFAULT,
        help=f"Disable RSI_M5 pullback guard filter (default: {'disabled' if _DISABLE_PULLBACK_FILTER_DEFAULT else 'enabled'} from risk_config.py)",
    )
    parser.add_argument(
        "--pullback-max-rsi-m5",
        type=float,
        default=PULLBACK_MAX_RSI_M5,
        help=f"Maximum RSI_M5 allowed when pullback filter enabled (default: {PULLBACK_MAX_RSI_M5})",
    )
    
    args = parser.parse_args()
    return args


def parse_and_validate_years(years_str: Optional[str]) -> Optional[list[int]]:
    """Parse comma-separated year string into list of integers.
    
    Args:
        years_str: Comma-separated year string (e.g., '2023,2024') or None
        
    Returns:
        List of years as integers, or None if input is None
        
    Raises:
        ValueError: If any year cannot be converted to integer
        
    Examples:
        >>> parse_and_validate_years('2023,2024')
        [2023, 2024]
        >>> parse_and_validate_years(None)
        None
    """
    if years_str is None:
        return None
    
    try:
        return [int(y.strip()) for y in years_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid year format. Expected comma-separated integers: {years_str}") from e
