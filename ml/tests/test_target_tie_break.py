import pandas as pd
import numpy as np
from ml.src.targets.target_maker import make_target


def test_tp_sl_tie_is_loss():
    # Construct a small M5-like OHLC DataFrame
    # Index: 0..4
    # At idx 0, close=100 -> SL=99 (atr=1, sl_mult=1), TP=102 (tp_mult=2)
    # At idx 1 (first future candle), set High >= 102 and Low <= 99 in the same bar (tie)

    data = {
        'Open': [100, 100, 101, 102, 103],
        'High': [100, 102, 103, 104, 105],  # idx1 high reaches TP
        'Low': [100, 99, 100, 101, 102],    # idx1 low reaches SL
        'Close': [100, 100, 102, 103, 104],
        'Volume': [100, 100, 100, 100, 100],
    }
    idx = pd.date_range('2025-12-01', periods=5, freq='5min')
    df = pd.DataFrame(data, index=idx)

    # Provide an explicit small ATR so thresholds are predictable
    # Add atr_m5 column so make_target picks it up
    df['atr_m5'] = 1.0

    # Use parameters so that tie would occur on the first future candle
    targets = make_target(df, atr_multiplier_sl=1.0, atr_multiplier_tp=2.0, min_hold_minutes=1, max_horizon=2)

    # The first available target corresponds to idx 0; for that sample TP and SL both hit on idx1
    # Conservative policy: tie -> LOSS (0)
    first_idx = targets.index[0]
    assert targets.iloc[0] == 0, f"Expected tie to be treated as loss, got {targets.iloc[0]} at {first_idx}"
