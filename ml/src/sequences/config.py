"""Sequence filtering configuration.

Purpose:
    Define configuration dataclass for session-level and feature-based filters.

Example:
    >>> config = SequenceFilterConfig(
    ...     enable_trend_filter=True,
    ...     trend_min_adx=20.0,
    ... )
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SequenceFilterConfig:
    """Configuration for session-level sequence filters.

    Attributes:
        enable_m5_alignment: If True, keep only windows that close one minute before
            a new M5 candle to synchronize decisions with bar openings.
        enable_trend_filter: If True, require price to be above a long-term trend
            proxy and volatility regime to exceed a minimum threshold.
        trend_min_dist_sma200: Minimum normalized distance above SMA200 accepted when
            trend filter is enabled.
        trend_min_adx: Minimum ADX value required when trend filter is enabled.
        enable_pullback_filter: If True, constrain RSI-based pullback conditions.
        pullback_max_rsi_m5: Maximum allowed RSI_M5 reading when pullback filter is
            enabled.
    """

    enable_m5_alignment: bool = True
    enable_trend_filter: bool = True
    trend_min_dist_sma200: Optional[float] = 0.0
    trend_min_adx: Optional[float] = 15.0
    enable_pullback_filter: bool = True
    pullback_max_rsi_m5: Optional[float] = 75.0
