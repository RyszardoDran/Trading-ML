#!/usr/bin/env python3
"""Proper correlation analysis with REAL 57 features on FULL dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from ml.src.data_loading.loaders import load_all_years
from ml.src.features.engineer import engineer_candle_features
from ml.src.targets.target_maker import make_target

print("="*80)
print("FEATURE CORRELATION ANALYSIS - FULL 2024 DATA WITH 57 ENGINEERED FEATURES")
print("="*80)

# Load FULL data
print("\n[1] LOADING DATA")
data_dir = Path("ml/src/data")
df = load_all_years(data_dir, year_filter=[2024])
print(f"Loaded {len(df):,} rows (FULL dataset)")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Engineer ACTUAL 57 features
print("\n[2] ENGINEERING 57 FEATURES")
window_size = 60
features_df = engineer_candle_features(df, window_size=window_size)
print(f"Shape: {features_df.shape}")
print(f"Columns: {list(features_df.columns)[:10]}... (showing first 10 of {len(features_df.columns)})")

# Check feature quality
print("\n[3] FEATURE QUALITY CHECK")
nan_count = features_df.isnull().sum().sum()
inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
print(f"Total NaN values: {nan_count}")
print(f"Total Inf values: {inf_count}")

# Variance check
print("\n[4] FEATURE VARIANCE")
variances = features_df.var(numeric_only=True).sort_values(ascending=False)
print(f"Top 10 features by variance:")
for i, (feat, var) in enumerate(variances.head(10).items(), 1):
    print(f"  {i:2d}. {feat:30s}: {var:.6f}")

zero_var = (variances < 1e-10).sum()
low_var = ((variances >= 1e-10) & (variances < 0.001)).sum()
print(f"\nFeatures with zero variance: {zero_var}")
print(f"Features with very low variance (<0.001): {low_var}")
print(f"Features with reasonable variance: {len(variances) - zero_var - low_var}")

# Create REAL targets using actual function
print("\n[5] CREATE TARGETS (SL=0.5, TP=1.0)")
targets = make_target(
    df,
    atr_multiplier_sl=0.5,
    atr_multiplier_tp=1.0,
    min_hold_minutes=10,
    max_horizon=120,
)
print(f"Target shape: {targets.shape}")
pos_count = int((targets == 1).sum())
neg_count = int((targets == 0).sum())
pos_pct = 100 * pos_count / len(targets)
print(f"Positive: {pos_count:,} ({pos_pct:.1f}%)")
print(f"Negative: {neg_count:,} ({100-pos_pct:.1f}%)")

# Align targets with features
min_len = min(len(features_df), len(targets))
features_aligned = features_df.iloc[:min_len]
targets_aligned = targets[:min_len]

print(f"\nAligned for correlation: {len(features_aligned):,} samples")

# Calculate correlations
print("\n[6] FEATURE-TARGET CORRELATIONS")
correlations = features_aligned.corrwith(pd.Series(targets_aligned, index=features_aligned.index))
correlations_abs = correlations.abs().sort_values(ascending=False)

print("\nTop 20 features by absolute correlation with target:")
for i, (feat, corr) in enumerate(correlations_abs.head(20).items(), 1):
    orig_corr = correlations[feat]
    print(f"  {i:2d}. {feat:30s}: {orig_corr:+.6f} (abs: {abs(orig_corr):.6f})")

# Statistics
print("\n[7] CORRELATION STATISTICS")
valid_corrs = correlations[~np.isnan(correlations) & ~np.isinf(correlations)]
print(f"Features with valid correlations: {len(valid_corrs)} / {len(correlations)}")
print(f"Mean absolute correlation: {valid_corrs.abs().mean():.6f}")
print(f"Max absolute correlation: {valid_corrs.abs().max():.6f}")
print(f"Median absolute correlation: {valid_corrs.abs().median():.6f}")

corrs_strong = (valid_corrs.abs() > 0.1).sum()
corrs_medium = ((valid_corrs.abs() > 0.05) & (valid_corrs.abs() <= 0.1)).sum()
corrs_weak = ((valid_corrs.abs() > 0.01) & (valid_corrs.abs() <= 0.05)).sum()
corrs_none = (valid_corrs.abs() <= 0.01).sum()

print(f"\nCorrelation distribution:")
print(f"  Strong (>0.1): {corrs_strong} features")
print(f"  Medium (0.05-0.1): {corrs_medium} features")
print(f"  Weak (0.01-0.05): {corrs_weak} features")
print(f"  None (<0.01): {corrs_none} features")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if valid_corrs.abs().max() < 0.05:
    print("[CRITICAL] Max correlation < 0.05 - features have NO predictive power")
    print("This explains why ROC-AUC = 0.49 (near random)")
elif valid_corrs.abs().max() < 0.1:
    print("[WARNING] Max correlation < 0.1 - very weak predictive power")
else:
    print("[OK] Max correlation > 0.1 - features should work")

if zero_var > 5:
    print(f"\n[PROBLEM] {zero_var} features have ZERO variance")
if low_var > 20:
    print(f"\n[WARNING] {low_var} features have very low variance")
