#!/usr/bin/env python3
"""Compare feature correlations for different ATR multipliers."""

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
print(f"Loaded {len(df):,} rows, engineering features...")

# Engineer features
df_feat = engineer_candle_features(df)
features = [col for col in df_feat.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"Engineered {len(features)} features")

# Test different ATR multipliers
scenarios = [
    ("Current (0.5/1.0)", 0.5, 1.0),
    ("Option A (0.2/0.4)", 0.2, 0.4),
    ("Option B (0.1/0.2)", 0.1, 0.2),
]

print("\n" + "="*100)
print("FEATURE CORRELATIONS BY ATR MULTIPLIER")
print("="*100)

results = []

for name, sl, tp in scenarios:
    targets = make_target(df, atr_multiplier_sl=sl, atr_multiplier_tp=tp, max_horizon=120)
    
    # Align
    valid_idx = targets.index
    X_feat = df_feat.loc[valid_idx, features]
    y = targets
    
    # Calculate correlations
    corrs = []
    for col in features:
        if X_feat[col].notna().sum() > 0:
            corr = X_feat[col].corr(y)
            if not np.isnan(corr):
                corrs.append((col, abs(corr)))
    
    corrs.sort(key=lambda x: x[1], reverse=True)
    max_corr = corrs[0][1] if corrs else 0
    mean_corr = np.mean([c[1] for c in corrs])
    
    pos = (y == 1).sum()
    pos_pct = 100 * pos / len(y)
    
    results.append({
        'Scenario': name,
        'SL/TP': f"{sl}/{tp}",
        'Pos %': f"{pos_pct:.1f}%",
        'Max Corr': max_corr,
        'Mean Corr': mean_corr,
        'Top Feature': corrs[0][0],
    })
    
    print(f"\n{name:30s} (Positive: {pos_pct:.1f}%)")
    print(f"  Max correlation:  {max_corr:.6f} ({corrs[0][0]})")
    print(f"  Mean correlation: {mean_corr:.6f}")
    print(f"  Top 5 features:")
    for feat, corr in corrs[:5]:
        print(f"    {feat:25s} {corr:.6f}")

results_df = pd.DataFrame(results)
print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print(results_df.to_string(index=False))

print("\n📊 INSIGHT: Weaker ATR targets (higher % wins) should show STRONGER feature correlations")
print("   If correlations stay weak with Option A/B, problem is feature engineering, not ATR multipliers")
