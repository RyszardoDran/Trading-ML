"""Risk management configuration shared across ML Python code.

This module centralizes the canonical ATR-based SL/TP parameters
and trading filter settings. Keep these consistent with ml/SEQUENCE_PIPELINE_README.md.
"""

from __future__ import annotations

# ===== ATR-Based Risk Parameters =====
ATR_PERIOD_M5: int = 14
SL_ATR_MULTIPLIER: float = 1
TP_ATR_MULTIPLIER: float = 2

# ===== Model Threshold Parameters =====
MIN_PRECISION_THRESHOLD: float = 0.7

# ===== Trading Filters =====
ENABLE_M5_ALIGNMENT: bool = False      # M5 candle close alignment
ENABLE_TREND_FILTER: bool = True       # Price above SMA200 and ADX threshold
ENABLE_PULLBACK_FILTER: bool = True   # RSI_M5 pullback guard

# ===== Trend Filter Parameters (when enabled) =====
TREND_MIN_DIST_SMA200: float = 0.0
TREND_MIN_ADX: float = 15.0

# ===== Pullback Filter Parameters (when enabled) =====
PULLBACK_MAX_RSI_M5: float = 75.0


def risk_reward_ratio() -> float:
    return TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER
