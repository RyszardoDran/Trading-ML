from __future__ import annotations

from pathlib import Path

import sys

# Make ml/src importable for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_risk_config_constants_exist_and_are_stable() -> None:
    from utils.risk_config import ATR_PERIOD_M5, SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER

    assert ATR_PERIOD_M5 == 14
    assert SL_ATR_MULTIPLIER == 1.0
    assert TP_ATR_MULTIPLIER == 2.0
    assert TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER == 2.0
