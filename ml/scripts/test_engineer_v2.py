#!/usr/bin/env python3
"""Test improved feature engineering V2."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from ml.src.data_loading.loaders import load_all_years
from ml.src.features.engineer_v2 import engineer_candle_features_v2
from ml.src.targets.target_maker import make_target

# Load data
data_dir = Path("ml/src/data")
df = load_all_years(data_dir, year_filter=[2024])
print(f"Loaded {len(df):,} rows")

# Engineer features V2
print("\nEngineering features V2 (multi-timeframe focus)...")
df_feat = engineer_candle_features_v2(df)
print(f"Created {df_feat.shape[1]} features (vs 57 before)\n")

# Create targets with OPTION A (0.2/0.4)
targets = make_target(df, atr_multiplier_sl=0.2, atr_multiplier_tp=0.4, max_horizon=120)

# Align
valid_idx = targets.index
X_feat = df_feat.loc[valid_idx, :]
y = targets

print("="*100)
print("FEATURE CORRELATIONS - V2 (IMPROVED)")
print("="*100)
print()

# Calculate correlations
results = []
for col in X_feat.columns:
    if X_feat[col].notna().sum() > 0:
        corr = X_feat[col].corr(y)
        if not np.isnan(corr):
            results.append((col, corr, abs(corr)))

results.sort(key=lambda x: x[2], reverse=True)

print(f"{'Feature':<25} {'Correlation':>12} {'Abs_Corr':>12}")
print("-" * 100)
for feat, corr, abs_corr in results:
    strength = "STRONG" if abs_corr > 0.05 else "MEDIUM" if abs_corr > 0.02 else "WEAK"
    print(f"{feat:<25} {corr:>+12.5f} {abs_corr:>12.5f}  [{strength}]")

# Summary
print("\n" + "="*100)
print("SUMMARY")
print("="*100)

mean_corr = np.mean([r[2] for r in results])
max_corr = max([r[2] for r in results])
strong_count = len([r for r in results if r[2] > 0.05])
medium_count = len([r for r in results if 0.02 < r[2] <= 0.05])
weak_count = len([r for r in results if r[2] <= 0.02])

print(f"Total features: {len(results)}")
print(f"Mean correlation: {mean_corr:.5f}")
print(f"Max correlation:  {max_corr:.5f}")
print(f"Strong (>0.05):   {strong_count}")
print(f"Medium (0.02-0.05): {medium_count}")
print(f"Weak (<=0.02):    {weak_count}")

print("\n✅ Comparison:")
print(f"  V1 (57 features):  max_corr=0.0747, mean=0.0148, features_kept=all")
print(f"  V2 ({len(results)} features): max_corr={max_corr:.5f}, mean={mean_corr:.5f}, features_kept={strong_count+medium_count} better ones")

if max_corr > 0.0747:
    print(f"\n🎉 IMPROVEMENT! V2 max correlation ({max_corr:.5f}) > V1 ({0.0747:.5f})")
else:
    print(f"\n⚠️  V2 max correlation ({max_corr:.5f}) <= V1 ({0.0747:.5f})")
    print("   But fewer features = simpler model, less overfitting risk")
