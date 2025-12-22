"""Configuration for backtesting framework.

Defines BacktestConfig dataclass with all parameters needed for backtesting.
Includes validation and sensible defaults for XAU/USD trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .position_sizer import PositionSizingMethod


@dataclass
class BacktestConfig:
    """Configuration for backtesting.
    
    Attributes:
        initial_capital: Starting capital in USD
        position_sizing: Position sizing method
        fixed_position_size: Fixed size in lots (for FIXED method)
        risk_per_trade: Risk % per trade (for RISK_BASED)
        spread_pips: Bid-ask spread in pips (XAU/USD typical: 0.5 pips = $0.50)
        slippage_pips: Slippage per trade in pips
        commission: Commission per trade in USD
        min_probability: Minimum model probability to take trade
        use_stop_loss: Whether to use stop loss from model
        use_take_profit: Whether to use take profit from model
        max_trades_per_day: Maximum trades per day (None = unlimited)
        max_drawdown_pct: Maximum drawdown before stopping (e.g., 0.20 = 20%)
        periods_per_year: Trading periods per year (for annualization)
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        walk_forward_enabled: Enable walk-forward analysis
        walk_forward_train_days: Days for training window
        walk_forward_test_days: Days for testing window
        walk_forward_step_days: Days to step forward
        
    Notes:
        For 1-minute XAU/USD data:
        - periods_per_year = 252 * 390 (trading days * minutes per day)
        - Typical spread: $0.50 per trade
        - Typical slippage: $0.30 per trade
    """
    
    # Capital and position sizing
    initial_capital: float = 100000.0
    position_sizing: PositionSizingMethod = PositionSizingMethod.FIXED
    fixed_position_size: float = 0.01  # 0.01 lots = 1 oz gold
    risk_per_trade: float = 0.01  # 1% risk per trade
    kelly_fraction: float = 0.25  # Fractional Kelly for safety
    
    # Transaction costs (XAU/USD)
    spread_pips: float = 0.5  # $0.50 spread
    slippage_pips: float = 0.3  # $0.30 slippage
    commission: float = 0.0  # Most Forex brokers don't charge commission
    
    # Trading rules
    min_probability: float = 0.5  # Minimum model probability
    use_stop_loss: bool = True
    use_take_profit: bool = True
    atr_sl_multiplier: float = 1.0  # ATR multiplier for stop loss
    atr_tp_multiplier: float = 1.8  # ATR multiplier for take profit
    max_horizon_minutes: int = 60  # Maximum trade duration in minutes
    max_trades_per_day: Optional[int] = None  # None = unlimited
    max_position_size: float = 1.0  # Max 1 lot
    min_position_size: float = 0.01  # Min 0.01 lots
    
    # Risk management
    max_drawdown_pct: float = 0.20  # Stop at 20% drawdown
    
    # Performance calculation
    periods_per_year: int = 252 * 390  # 1-minute data
    risk_free_rate: float = 0.045  # 4.5% annual (Dec 2025)
    
    # Walk-forward analysis
    walk_forward_enabled: bool = False
    walk_forward_train_days: int = 180  # 6 months training
    walk_forward_test_days: int = 30  # 1 month testing
    walk_forward_step_days: int = 30  # Step forward 1 month
    
    # Outputs
    save_trades: bool = True
    save_equity_curve: bool = True
    output_dir: Path = field(default_factory=lambda: Path("ml/outputs/backtests"))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError(f"risk_per_trade must be in (0, 1], got {self.risk_per_trade}")
        
        if self.spread_pips < 0:
            raise ValueError(f"spread_pips must be non-negative, got {self.spread_pips}")
        
        if self.slippage_pips < 0:
            raise ValueError(f"slippage_pips must be non-negative, got {self.slippage_pips}")
        
        if self.commission < 0:
            raise ValueError(f"commission must be non-negative, got {self.commission}")
        
        if not 0 <= self.min_probability <= 1:
            raise ValueError(f"min_probability must be in [0, 1], got {self.min_probability}")
        
        if not 0 < self.max_drawdown_pct <= 1:
            raise ValueError(f"max_drawdown_pct must be in (0, 1], got {self.max_drawdown_pct}")
        
        if self.max_position_size < self.min_position_size:
            raise ValueError(
                f"max_position_size ({self.max_position_size}) must be >= "
                f"min_position_size ({self.min_position_size})"
            )
        
        if self.walk_forward_enabled:
            if self.walk_forward_train_days <= 0:
                raise ValueError("walk_forward_train_days must be positive")
            if self.walk_forward_test_days <= 0:
                raise ValueError("walk_forward_test_days must be positive")
            if self.walk_forward_step_days <= 0:
                raise ValueError("walk_forward_step_days must be positive")
    
    def total_transaction_cost(self) -> float:
        """Calculate total transaction cost per trade.
        
        Returns:
            Total cost in USD (spread + slippage + commission)
        """
        spread_cost = self.spread_pips
        slippage_cost = self.slippage_pips
        return spread_cost + slippage_cost + self.commission
    
    def to_dict(self) -> dict:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            'initial_capital': self.initial_capital,
            'position_sizing': self.position_sizing.value,
            'fixed_position_size': self.fixed_position_size,
            'risk_per_trade': self.risk_per_trade,
            'spread_pips': self.spread_pips,
            'slippage_pips': self.slippage_pips,
            'commission': self.commission,
            'min_probability': self.min_probability,
            'max_drawdown_pct': self.max_drawdown_pct,
            'walk_forward_enabled': self.walk_forward_enabled,
        }
