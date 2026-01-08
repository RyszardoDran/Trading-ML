"""Extended pipeline configuration with comprehensive validation.

Purpose:
    Provide PipelineParams dataclass combining all CLI arguments with validation.
    Validates parameter ranges, compatibility, and consistency.
    Acts as single source of truth for training configuration.

Usage:
    >>> from ml.src.pipeline_config_extended import PipelineParams
    >>> from ml.src.pipeline_cli import parse_cli_arguments
    >>> args = parse_cli_arguments()
    >>> params = PipelineParams.from_cli_args(args)
    >>> params.validate()  # Raises ValueError if invalid
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .training.sequence_xgb_params import available_profiles

logger = logging.getLogger(__name__)


@dataclass
class PipelineParams:
    """Complete training pipeline configuration with validation.
    
    Combines all CLI arguments and provides comprehensive parameter validation.
    Ensures that all parameters are within valid ranges and are compatible
    with each other.
    
    Attributes (Core parameters):
        window_size: Number of previous candles for sequences (must be >= 1)
        atr_multiplier_sl: ATR multiplier for stop-loss (must be > 0)
        atr_multiplier_tp: ATR multiplier for take-profit (must be > atr_multiplier_sl)
        min_hold_minutes: Minimum holding time (must be >= 1)
        max_horizon: Maximum forward candles (must be >= min_hold_minutes)
        random_state: Random seed for reproducibility
        
    Attributes (Data selection):
        year_filter: Optional list of years to load (None = all years)
        
    Attributes (Session/time filters):
        session: Trading session ('london', 'ny', 'asian', 'london_ny', 'all', 'custom')
        custom_start_hour: Start hour for custom session (0-23)
        custom_end_hour: End hour for custom session (0-23)
        
    Attributes (Sequence constraints):
        max_windows: Maximum training windows to keep
        
    Attributes (Time Series Cross-Validation):
        use_timeseries_cv: Use Time Series CV instead of single split
        cv_folds: Number of CV folds (default 5)
        
    Attributes (Threshold optimization):
        min_precision: Minimum precision threshold (0-1)
        min_trades: Minimum predicted positives (None = dynamic)
        max_trades_per_day: Cap trades per day (None = unlimited)
        use_ev_optimization: Use Expected Value optimization instead of F1
        ev_win_coefficient: Profit multiplier for correct predictions
        ev_loss_coefficient: Loss multiplier for incorrect predictions
        
    Attributes (M5 alignment filter):
        enable_m5_alignment: Enable M5 candle close alignment
        
    Attributes (Trend filter):
        enable_trend_filter: Enable trend filter
        trend_min_dist_sma200: Min distance above SMA200 (None = disabled)
        trend_min_adx: Min ADX threshold (None = disabled)
        
    Attributes (Pullback filter):
        enable_pullback_filter: Enable RSI_M5 pullback guard
        pullback_max_rsi_m5: Max RSI_M5 allowed (None = disabled)
    """
    
    # Core parameters
    window_size: int
    atr_multiplier_sl: float
    atr_multiplier_tp: float
    min_hold_minutes: int
    max_horizon: int
    random_state: int
    feature_version: str  # "v1" or "v2"
    
    # Data selection
    year_filter: Optional[list[int]]
    
    # Session/time filters
    session: str
    custom_start_hour: Optional[int]
    custom_end_hour: Optional[int]
    
    # Sequence constraints
    max_windows: int
    
    # Time Series Cross-Validation
    use_timeseries_cv: bool
    cv_folds: int
    
    # Threshold optimization
    min_precision: float
    min_recall: float
    min_trades: Optional[int]
    max_trades_per_day: Optional[int]
    use_ev_optimization: bool
    use_hybrid_optimization: bool
    ev_win_coefficient: float
    ev_loss_coefficient: float
    
    # Cost-Sensitive Learning (POINT 1)
    use_cost_sensitive_learning: bool
    sample_weight_positive: float
    sample_weight_negative: float
    
    # M5 alignment filter
    enable_m5_alignment: bool
    
    # Trend filter
    enable_trend_filter: bool
    trend_min_dist_sma200: Optional[float]
    trend_min_adx: Optional[float]
    
    # Pullback filter
    enable_pullback_filter: bool
    pullback_max_rsi_m5: Optional[float]

    # XGBoost hyperparameters
    xgb_profile: str
    xgb_learning_rate: Optional[float]
    xgb_max_depth: Optional[int]
    xgb_min_child_weight: Optional[float]
    xgb_subsample: Optional[float]
    xgb_colsample_bytree: Optional[float]
    xgb_colsample_bynode: Optional[float]
    xgb_reg_lambda: Optional[float]
    xgb_reg_alpha: Optional[float]
    xgb_gamma: Optional[float]
    xgb_n_estimators: Optional[int]
    xgb_max_delta_step: Optional[float]
    
    @classmethod
    def from_cli_args(cls, args) -> PipelineParams:
        """Create PipelineParams from parsed CLI arguments.
        
        Args:
            args: argparse.Namespace from parse_cli_arguments()
            
        Returns:
            PipelineParams instance with all values from CLI args
        """
        return cls(
            window_size=args.window_size,
            atr_multiplier_sl=args.atr_multiplier_sl,
            atr_multiplier_tp=args.atr_multiplier_tp,
            min_hold_minutes=args.min_hold_minutes,
            max_horizon=args.max_horizon,
            random_state=args.random_state,
            feature_version=args.feature_version,
            year_filter=None,  # Set after parsing
            session=args.session,
            custom_start_hour=args.custom_start_hour,
            custom_end_hour=args.custom_end_hour,
            max_windows=args.max_windows,
            use_timeseries_cv=args.use_timeseries_cv,
            cv_folds=args.cv_folds,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            min_trades=args.min_trades,
            max_trades_per_day=args.max_trades_per_day,
            use_ev_optimization=args.use_ev_optimization,
            use_hybrid_optimization=args.use_hybrid_optimization,
            ev_win_coefficient=args.ev_win_coefficient,
            ev_loss_coefficient=args.ev_loss_coefficient,
            use_cost_sensitive_learning=args.use_cost_sensitive_learning,
            sample_weight_positive=args.sample_weight_positive,
            sample_weight_negative=args.sample_weight_negative,
            enable_m5_alignment=not args.skip_m5_alignment,
            enable_trend_filter=not args.disable_trend_filter,
            trend_min_dist_sma200=None if args.disable_trend_filter else args.trend_min_dist_sma200,
            trend_min_adx=None if args.disable_trend_filter else args.trend_min_adx,
            enable_pullback_filter=not args.disable_pullback_filter,
            pullback_max_rsi_m5=None if args.disable_pullback_filter else args.pullback_max_rsi_m5,
            xgb_profile=args.xgb_profile,
            xgb_learning_rate=args.xgb_learning_rate,
            xgb_max_depth=args.xgb_max_depth,
            xgb_min_child_weight=args.xgb_min_child_weight,
            xgb_subsample=args.xgb_subsample,
            xgb_colsample_bytree=args.xgb_colsample_bytree,
            xgb_colsample_bynode=args.xgb_colsample_bynode,
            xgb_reg_lambda=args.xgb_reg_lambda,
            xgb_reg_alpha=args.xgb_reg_alpha,
            xgb_gamma=args.xgb_gamma,
            xgb_n_estimators=args.xgb_n_estimators,
            xgb_max_delta_step=args.xgb_max_delta_step,
        )
    
    def validate(self) -> None:
        """Validate all parameters for valid ranges and compatibility.
        
        Checks:
        - Core parameters within valid ranges
        - Compatibility between SL/TP multiples
        - Horizon >= min_hold_minutes
        - Precision threshold in [0, 1]
        - Session validity
        - Custom hours in [0, 23] if session is 'custom'
        - Filter parameters when filters enabled
        
        Raises:
            ValueError: If any validation fails
            
        Examples:
            >>> params = PipelineParams(...)
            >>> params.validate()  # Raises ValueError if invalid
        """
        # Core parameter validation
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        
        if self.atr_multiplier_sl <= 0:
            raise ValueError(f"atr_multiplier_sl must be > 0, got {self.atr_multiplier_sl}")
        
        if self.atr_multiplier_tp <= 0:
            raise ValueError(f"atr_multiplier_tp must be > 0, got {self.atr_multiplier_tp}")
        
        if self.atr_multiplier_tp <= self.atr_multiplier_sl:
            raise ValueError(
                f"atr_multiplier_tp ({self.atr_multiplier_tp}) must be > "
                f"atr_multiplier_sl ({self.atr_multiplier_sl}) for positive RR"
            )
        
        if self.min_hold_minutes < 1:
            raise ValueError(f"min_hold_minutes must be >= 1, got {self.min_hold_minutes}")
        
        if self.max_horizon < self.min_hold_minutes:
            raise ValueError(
                f"max_horizon ({self.max_horizon}) must be >= "
                f"min_hold_minutes ({self.min_hold_minutes})"
            )
        
        if self.max_windows < 100:
            raise ValueError(f"max_windows should be >= 100, got {self.max_windows}")
        
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        
        # Threshold validation
        if not (0 <= self.min_precision <= 1):
            raise ValueError(f"min_precision must be in [0, 1], got {self.min_precision}")

        if not (0 <= self.min_recall <= 1):
            raise ValueError(f"min_recall must be in [0, 1], got {self.min_recall}")
        
        if self.min_trades is not None and self.min_trades < 1:
            raise ValueError(f"min_trades must be >= 1 or None, got {self.min_trades}")
        
        if self.max_trades_per_day is not None and self.max_trades_per_day < 1:
            raise ValueError(f"max_trades_per_day must be >= 1 or None, got {self.max_trades_per_day}")
        
        # Session validation
        valid_sessions = ["london", "ny", "asian", "london_ny", "all", "custom"]
        if self.session not in valid_sessions:
            raise ValueError(f"session must be one of {valid_sessions}, got {self.session}")
        
        # Custom session validation
        if self.session == "custom":
            if self.custom_start_hour is None or self.custom_end_hour is None:
                raise ValueError(
                    "custom_start_hour and custom_end_hour required when session='custom'"
                )
            if not (0 <= self.custom_start_hour < 24):
                raise ValueError(f"custom_start_hour must be in [0, 23], got {self.custom_start_hour}")
            if not (0 <= self.custom_end_hour < 24):
                raise ValueError(f"custom_end_hour must be in [0, 23], got {self.custom_end_hour}")
        
        # Trend filter validation
        if self.enable_trend_filter:
            if self.trend_min_dist_sma200 is not None and self.trend_min_dist_sma200 < 0:
                raise ValueError(
                    f"trend_min_dist_sma200 must be >= 0 or None, got {self.trend_min_dist_sma200}"
                )
            if self.trend_min_adx is not None and self.trend_min_adx < 0:
                raise ValueError(
                    f"trend_min_adx must be >= 0 or None, got {self.trend_min_adx}"
                )
        
        # Pullback filter validation
        if self.enable_pullback_filter:
            if self.pullback_max_rsi_m5 is not None:
                if not (0 <= self.pullback_max_rsi_m5 <= 100):
                    raise ValueError(
                        f"pullback_max_rsi_m5 must be in [0, 100] or None, got {self.pullback_max_rsi_m5}"
                    )

        # XGBoost profile validation
        valid_profiles = set(available_profiles())
        if self.xgb_profile not in valid_profiles:
            raise ValueError(
                f"xgb_profile must be one of {sorted(valid_profiles)}, got {self.xgb_profile}"
            )

        if self.xgb_learning_rate is not None and self.xgb_learning_rate <= 0:
            raise ValueError(
                f"xgb_learning_rate must be > 0 when set, got {self.xgb_learning_rate}"
            )

        if self.xgb_max_depth is not None and self.xgb_max_depth < 1:
            raise ValueError(
                f"xgb_max_depth must be >= 1 when set, got {self.xgb_max_depth}"
            )

        if self.xgb_min_child_weight is not None and self.xgb_min_child_weight <= 0:
            raise ValueError(
                f"xgb_min_child_weight must be > 0 when set, got {self.xgb_min_child_weight}"
            )

        if self.xgb_subsample is not None:
            if not (0 < self.xgb_subsample <= 1):
                raise ValueError(
                    f"xgb_subsample must be in (0, 1], got {self.xgb_subsample}"
                )

        if self.xgb_colsample_bytree is not None:
            if not (0 < self.xgb_colsample_bytree <= 1):
                raise ValueError(
                    f"xgb_colsample_bytree must be in (0, 1], got {self.xgb_colsample_bytree}"
                )

        if self.xgb_colsample_bynode is not None:
            if not (0 < self.xgb_colsample_bynode <= 1):
                raise ValueError(
                    f"xgb_colsample_bynode must be in (0, 1], got {self.xgb_colsample_bynode}"
                )

        if self.xgb_reg_lambda is not None and self.xgb_reg_lambda < 0:
            raise ValueError(
                f"xgb_reg_lambda must be >= 0 when set, got {self.xgb_reg_lambda}"
            )

        if self.xgb_reg_alpha is not None and self.xgb_reg_alpha < 0:
            raise ValueError(
                f"xgb_reg_alpha must be >= 0 when set, got {self.xgb_reg_alpha}"
            )

        if self.xgb_gamma is not None and self.xgb_gamma < 0:
            raise ValueError(
                f"xgb_gamma must be >= 0 when set, got {self.xgb_gamma}"
            )

        if self.xgb_n_estimators is not None and self.xgb_n_estimators < 50:
            raise ValueError(
                f"xgb_n_estimators must be >= 50 when set, got {self.xgb_n_estimators}"
            )

        if self.xgb_max_delta_step is not None and self.xgb_max_delta_step < 0:
            raise ValueError(
                f"xgb_max_delta_step must be >= 0 when set, got {self.xgb_max_delta_step}"
            )
        
        logger.info("Pipeline configuration validated successfully")
