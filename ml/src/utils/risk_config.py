"""Risk management configuration shared across ML Python code.

This module centralizes the canonical ATR-based SL/TP parameters.
Keep these consistent with ml/SEQUENCE_PIPELINE_README.md.
"""

from __future__ import annotations

ATR_PERIOD_M5: int = 14
SL_ATR_MULTIPLIER: float = 1.0
TP_ATR_MULTIPLIER: float = 2.0


def risk_reward_ratio() -> float:
    return TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER
