"""Trading filters module.

Purpose:
    Apply market condition filters to trading decisions.

Exports:
    - MarketRegime: Regime classification enum
    - should_trade(): Check if current regime is suitable for trading
    - classify_regime(): Classify market regime (TIER 1-4)
    - filter_sequences_by_regime(): Filter sequences to favorable regimes
    - filter_predictions_by_regime(): Gate model predictions by regime
    - get_adaptive_threshold(): Get probability threshold based on volatility
"""

from .regime_filter import (
    MarketRegime,
    classify_regime,
    filter_predictions_by_regime,
    filter_sequences_by_regime,
    get_adaptive_threshold,
    should_trade,
)

__all__ = [
    "MarketRegime",
    "classify_regime",
    "should_trade",
    "filter_sequences_by_regime",
    "filter_predictions_by_regime",
    "get_adaptive_threshold",
]
