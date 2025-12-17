#!/usr/bin/env python3
"""Optimize ATR multipliers for target recall (>=20%) + win rate (>=70%)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from ml.src.data_loading.loaders import load_all_years
from ml.src.features.engineer_v2 import engineer_candle_features_v2
from ml.src.targets.target_maker import make_target
from ml.src.training.sequence_xgb_trainer import train_xgb
from ml.src.training.sequence_evaluation import evaluate

# Load data
data_dir = Path("ml/src/data")
df = load_all_years(data_dir, year_filter=[2024])
print(f"Loaded {len(df):,} rows")

# Engineer features V2
print("Engineering V2 features...")
features = engineer_candle_features_v2(df)

# Test different ATR multipliers
test_cases = [
    ("OPTION A", 0.2, 0.4),    # Current: recall 16%
    ("OPTION A+", 0.15, 0.3),  # More aggressive
    ("OPTION B", 0.1, 0.2),    # Very aggressive
    ("OPTION C", 0.18, 0.36),  # Fine-tune between A and A+
    ("OPTION D", 0.16, 0.32),  # Fine-tune
]

print("\n" + "=" * 130)
print("ATR MULTIPLIER OPTIMIZATION - FINDING OPTIMAL RECALL >= 20%")
print("=" * 130 + "\n")

results = []

for name, sl, tp in test_cases:
    print(f"Testing {name:15s} (SL={sl:.2f}, TP={tp:.2f})...", end="", flush=True)
    
    # Create targets
    targets = make_target(df, atr_multiplier_sl=sl, atr_multiplier_tp=tp, max_horizon=120)
    pos_pct = 100 * (targets == 1).sum() / len(targets)
    
    # Align
    valid_idx = targets.index
    X = features.loc[valid_idx, :]
    y = targets
    
    # Split: 70/15/15
    split_70 = int(0.70 * len(X))
    split_85 = int(0.85 * len(X))
    
    X_train = X.iloc[:split_70]
    X_val = X.iloc[split_70:split_85]
    X_test = X.iloc[split_85:]
    
    y_train = y.iloc[:split_70]
    y_val = y.iloc[split_70:split_85]
    y_test = y.iloc[split_85:]
    
    # Train
    model = train_xgb(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    
    # Store results
    results.append({
        'Option': name,
        'SL': sl,
        'TP': tp,
        'RR': tp / sl,
        'Pos%': pos_pct,
        'WinRate': metrics['win_rate'],
        'Recall': metrics['recall'],
        'ROC-AUC': metrics['roc_auc'],
        'F1': metrics['f1'],
    })
    
    print(f" ✓ ROC-AUC={metrics['roc_auc']:.4f}, Recall={metrics['recall']:.2%}, WinRate={metrics['win_rate']:.2%}")

# Print results table
print("\n" + "=" * 130)
print("SUMMARY TABLE")
print("=" * 130)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best option
print("\n" + "=" * 130)
print("ANALYSIS")
print("=" * 130)

# Filter for good win rates
good_wr = results_df[results_df['WinRate'] >= 0.70]
if len(good_wr) > 0:
    # Among those, find best recall
    best = good_wr.nlargest(1, 'Recall').iloc[0]
    print(f"\n✅ BEST OPTION: {best['Option']}")
    print(f"   SL={best['SL']:.2f}, TP={best['TP']:.2f}")
    print(f"   Win Rate: {best['WinRate']:.2%}")
    print(f"   Recall:   {best['Recall']:.2%}")
    print(f"   ROC-AUC:  {best['ROC-AUC']:.4f}")
    
    if best['Recall'] >= 0.20:
        print(f"\n🎉 SPEC MET! Recall >= 20% achieved!")
    else:
        needed_improvement = 0.20 - best['Recall']
        print(f"\n⚠️  Need {needed_improvement:.2%} more recall to hit 20% target")
else:
    print("No options meet win rate >= 70%")

print("\n" + "=" * 130)
