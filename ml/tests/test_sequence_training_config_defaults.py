from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_pipeline_config_defaults_match_risk_config() -> None:
    from utils.risk_config import SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER
    from utils.sequence_training_config import PipelineConfig

    cfg = PipelineConfig()

    assert cfg.atr_multiplier_sl == SL_ATR_MULTIPLIER
    assert cfg.atr_multiplier_tp == TP_ATR_MULTIPLIER
