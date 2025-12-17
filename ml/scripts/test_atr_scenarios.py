#!/usr/bin/env python3
"""Test different ATR multiplier scenarios."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from ml.src.data_loading.loaders import load_all_years
from ml.src.features.engineer import engineer_candle_features
from ml.src.targets.target_maker import make_target

data_dir = Path("ml/src/data")
df = load_all_years(data_dir, year_filter=[2024])
print(f"Loaded {len(df):,} rows")

# Test different ATR multipliers
scenarios = [
    ("Current", 0.5, 1.0),
    ("Option A", 0.2, 0.4),
    ("Option B", 0.1, 0.2),
    ("Ultra Aggressive", 0.05, 0.1),
]

print("\n" + "="*80)
print("ATR MULTIPLIER SCENARIOS - TARGET DISTRIBUTION")
print("="*80)

for name, sl, tp in scenarios:
    targets = make_target(df, atr_multiplier_sl=sl, atr_multiplier_tp=tp, max_horizon=120)
    pos = (targets == 1).sum()
    neg = (targets == 0).sum()
    pos_pct = 100 * pos / len(targets)
    rr = tp / sl
    
    print(f"\n{name:20s} (SL={sl}, TP={tp}, RR=1:{rr:.1f})")
    print(f"  Positive: {pos:7,} ({pos_pct:5.1f}%)")
    print(f"  Negative: {neg:7,} ({100-pos_pct:5.1f}%)")
    print(f"  Ratio: {neg/max(pos,1):.2f}:1")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("Use OPTION A or OPTION B to increase positive class %")
print("More 'wins' = more signal for model to learn from")
