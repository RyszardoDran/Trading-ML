#!/usr/bin/env python3
"""Ultra-fast debug: Feature-target correlation on SAMPLE DATA."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

data_file = Path("ml/src/data/XAU_1m_data_2024.csv")
print(f"Loading {data_file}...")

df = pd.read_csv(data_file, sep=';', index_col='Date', parse_dates=True)
print(f"[OK] Loaded {len(df):,} rows")

# SAMPLE ONLY FIRST 10k rows for speed
df = df.iloc[:10000].copy()
print(f"Using first {len(df):,} rows for fast analysis")

print("\n[1] SIMPLE TARGETS")
# Does next close go higher?
df['target_up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
up_pct = 100 * df['target_up'].sum() / len(df)
print(f"  Up: {df['target_up'].sum():,} ({up_pct:.1f}%)")

# Does price move significantly (>0.5% ATR)?
df['atr'] = (df['High'].rolling(14).max() - df['Low'].rolling(14).min()).fillna(0)
df['target_sig'] = (np.abs(df['Close'].shift(-1) - df['Close']) > 0.005 * df['Close']).astype(int)
sig_pct = 100 * df['target_sig'].sum() / len(df)
print(f"  Significant move: {df['target_sig'].sum():,} ({sig_pct:.1f}%)")

print("\n[2] BASIC FEATURES (5 features for speed)")
features = pd.DataFrame({
    'close_pct_change': df['Close'].pct_change().fillna(0),
    'high_low_ratio': (df['High'] - df['Low']) / df['Close'],
    'open_close_ratio': (df['Open'] - df['Close']) / df['Close'],
    'volume_change': df['Volume'].pct_change().fillna(0),
    'price_position': (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-6),
})

print("Feature statistics:")
for col in features.columns:
    print(f"  {col:20s}: mean={features[col].mean():+.6f}, std={features[col].std():.6f}")

print("\n[3] CORRELATIONS WITH TARGETS")
print("Target: Simple UP (next_close > current_close)")
corrs_up = [features[col].corr(df['target_up']) for col in features.columns]
for col, corr in zip(features.columns, corrs_up):
    print(f"  {col:20s}: {corr:+.6f}")
max_corr_up = max(abs(c) for c in corrs_up if not np.isnan(c))
print(f"  Max abs correlation: {max_corr_up:.6f}")

print("\nTarget: Significant move")
corrs_sig = [features[col].corr(df['target_sig']) for col in features.columns]
for col, corr in zip(features.columns, corrs_sig):
    print(f"  {col:20s}: {corr:+.6f}")
max_corr_sig = max(abs(c) for c in corrs_sig if not np.isnan(c))
print(f"  Max abs correlation: {max_corr_sig:.6f}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
if max_corr_up < 0.01 and max_corr_sig < 0.01:
    print("[PROBLEM] Features have ZERO correlation with targets")
    print("Possible causes:")
    print("  1. Features are not engineered correctly")
    print("  2. Features are too simple (just OHLC)")
    print("  3. Gold price is fundamentally unpredictable with these features")
    print("  4. Need to add more complex features (indicators, volume patterns, etc)")
elif max_corr_up < 0.05 or max_corr_sig < 0.05:
    print("[WARNING] Feature correlation is very weak (<0.05)")
    print("Model will have difficulty learning")
else:
    print("[OK] Features show reasonable correlation")
    print("Model should be able to learn patterns")
