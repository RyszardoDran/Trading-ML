"""Position sizing strategies for trading.

Implements multiple position sizing methods:
- Fixed: Constant position size (e.g., 0.01 lots)
- Risk-based: Size based on % of capital at risk
- Kelly Criterion: Optimal f based on win rate and win/loss ratio
- Volatility-based: Adjust size based on market volatility

All methods validate inputs and handle edge cases (zero capital, extreme volatility).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Available position sizing methods."""
    FIXED = "fixed"
    RISK_BASED = "risk_based"
    KELLY = "kelly"
    VOLATILITY_BASED = "volatility_based"


class PositionSizer:
    """Calculate position sizes for trades.
    
    Attributes:
        method: Position sizing method to use
        fixed_size: Fixed position size (for FIXED method)
        risk_per_trade: Risk percentage per trade (for RISK_BASED)
        kelly_fraction: Fraction of Kelly to use (0-1, default 0.25)
        min_size: Minimum position size allowed
        max_size: Maximum position size allowed
        
    Examples:
        >>> sizer = PositionSizer(method=PositionSizingMethod.FIXED, fixed_size=0.01)
        >>> size = sizer.calculate_size(capital=10000, signal_prob=0.7)
        >>> size
        0.01
    """
    
    def __init__(
        self,
        method: PositionSizingMethod = PositionSizingMethod.FIXED,
        fixed_size: float = 0.01,
        risk_per_trade: float = 0.01,
        kelly_fraction: float = 0.25,
        min_size: float = 0.01,
        max_size: float = 1.0,
    ):
        """Initialize position sizer.
        
        Args:
            method: Position sizing method
            fixed_size: Fixed size for FIXED method (in lots)
            risk_per_trade: Risk % per trade for RISK_BASED (e.g., 0.01 = 1%)
            kelly_fraction: Fraction of Kelly criterion (default 0.25 for safety)
            min_size: Minimum position size (lots)
            max_size: Maximum position size (lots)
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.method = method
        self.fixed_size = fixed_size
        self.risk_per_trade = risk_per_trade
        self.kelly_fraction = kelly_fraction
        self.min_size = min_size
        self.max_size = max_size
        
        # Validate
        if fixed_size <= 0:
            raise ValueError(f"fixed_size must be positive, got {fixed_size}")
        
        if not 0 < risk_per_trade <= 1:
            raise ValueError(f"risk_per_trade must be in (0, 1], got {risk_per_trade}")
        
        if not 0 < kelly_fraction <= 1:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        
        if min_size <= 0:
            raise ValueError(f"min_size must be positive, got {min_size}")
        
        if max_size < min_size:
            raise ValueError(f"max_size ({max_size}) must be >= min_size ({min_size})")
        
        logger.info(f"Initialized PositionSizer: method={method.value}, "
                   f"risk_per_trade={risk_per_trade:.2%}")
    
    def calculate_size(
        self,
        capital: float,
        signal_prob: Optional[float] = None,
        stop_loss_pips: Optional[float] = None,
        pip_value: float = 1.0,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """Calculate position size for a trade.
        
        Args:
            capital: Current account capital
            signal_prob: Model's predicted probability (0-1)
            stop_loss_pips: Distance to stop loss in pips
            pip_value: Value of 1 pip (default 1.0 for XAU/USD)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average winning trade size (for Kelly)
            avg_loss: Average losing trade size (for Kelly)
            volatility: Market volatility (for volatility-based)
            
        Returns:
            Position size in lots (bounded by min_size and max_size)
            
        Raises:
            ValueError: If required parameters are missing or invalid
            
        Examples:
            >>> sizer = PositionSizer(method=PositionSizingMethod.RISK_BASED)
            >>> size = sizer.calculate_size(10000, stop_loss_pips=50)
            >>> 0.01 <= size <= 1.0
            True
        """
        if capital <= 0:
            raise ValueError(f"capital must be positive, got {capital}")
        
        if self.method == PositionSizingMethod.FIXED:
            size = self._fixed_size()
        
        elif self.method == PositionSizingMethod.RISK_BASED:
            if stop_loss_pips is None or stop_loss_pips <= 0:
                raise ValueError("stop_loss_pips required for risk-based sizing")
            size = self._risk_based_size(capital, stop_loss_pips, pip_value)
        
        elif self.method == PositionSizingMethod.KELLY:
            if win_rate is None or avg_win is None or avg_loss is None:
                raise ValueError("win_rate, avg_win, avg_loss required for Kelly sizing")
            size = self._kelly_size(capital, win_rate, avg_win, avg_loss)
        
        elif self.method == PositionSizingMethod.VOLATILITY_BASED:
            if volatility is None or volatility <= 0:
                raise ValueError("volatility required for volatility-based sizing")
            size = self._volatility_based_size(capital, volatility)
        
        else:
            raise ValueError(f"Unknown position sizing method: {self.method}")
        
        # Apply bounds
        size = max(self.min_size, min(size, self.max_size))
        
        return size
    
    def _fixed_size(self) -> float:
        """Return fixed position size."""
        return self.fixed_size
    
    def _risk_based_size(
        self,
        capital: float,
        stop_loss_pips: float,
        pip_value: float,
    ) -> float:
        """Calculate risk-based position size.
        
        Formula: size = (capital * risk_per_trade) / (stop_loss_pips * pip_value)
        
        Args:
            capital: Current capital
            stop_loss_pips: Stop loss distance in pips
            pip_value: Value per pip
            
        Returns:
            Position size in lots
        """
        risk_amount = capital * self.risk_per_trade
        size = risk_amount / (stop_loss_pips * pip_value)
        return size
    
    def _kelly_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Calculate Kelly Criterion position size.
        
        Kelly % = W - (1-W) / R
        where W = win rate, R = avg_win / avg_loss
        
        Args:
            capital: Current capital
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive value)
            
        Returns:
            Position size in lots
            
        Notes:
            - Uses fractional Kelly (default 0.25) for safety
            - Returns min_size if Kelly is negative
        """
        if not 0 <= win_rate <= 1:
            raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")
        
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply fractional Kelly for safety
        kelly_pct *= self.kelly_fraction
        
        if kelly_pct <= 0:
            logger.warning(f"Negative Kelly ({kelly_pct:.4f}), using min_size")
            return self.min_size
        
        # Convert to position size (assuming size proportional to capital)
        # This is a simplified model; real implementation needs contract specs
        size = kelly_pct * capital / 10000  # Scale factor
        return size
    
    def _volatility_based_size(
        self,
        capital: float,
        volatility: float,
    ) -> float:
        """Calculate volatility-adjusted position size.
        
        Inverse relationship: higher volatility = smaller size.
        
        Args:
            capital: Current capital
            volatility: Market volatility (e.g., ATR or standard deviation)
            
        Returns:
            Position size in lots
        """
        # Base size adjusted by inverse volatility
        base_vol = 0.01  # Reference volatility
        size = self.fixed_size * (base_vol / volatility)
        return size


def calculate_pip_value(
    instrument: str = "XAUUSD",
    lot_size: float = 0.01,
) -> float:
    """Calculate pip value for instrument.
    
    Args:
        instrument: Trading instrument (default XAU/USD)
        lot_size: Position size in lots
        
    Returns:
        Value of 1 pip movement
        
    Notes:
        For XAU/USD: 1 lot = 100 oz, 1 pip = $0.01
        For 0.01 lots: pip value = $1.00
        
    Examples:
        >>> calculate_pip_value("XAUUSD", 0.01)
        1.0
    """
    if instrument.upper() == "XAUUSD":
        # XAU/USD: 1 standard lot = 100 oz
        # 1 pip = $0.01 per oz
        # For 1 lot: pip value = 100 * 0.01 = $1.00
        # For 0.01 lots: pip value = $0.01
        return lot_size * 100.0
    else:
        raise ValueError(f"Unsupported instrument: {instrument}")


def lots_to_units(
    lots: float,
    instrument: str = "XAUUSD",
) -> float:
    """Convert lots to units (ounces for XAU/USD).
    
    Args:
        lots: Position size in lots
        instrument: Trading instrument
        
    Returns:
        Position size in units
        
    Examples:
        >>> lots_to_units(0.01, "XAUUSD")
        1.0
    """
    if instrument.upper() == "XAUUSD":
        return lots * 100.0  # 1 lot = 100 oz
    else:
        raise ValueError(f"Unsupported instrument: {instrument}")
