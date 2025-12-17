#!/usr/bin/env python3
"""Debug script for data quality analysis."""

import sys
import os
from pathlib import Path

# Add repo root to path FIRST
repo_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_dir))
os.chdir(repo_dir)

print(f"Working directory: {os.getcwd()}")
print(f"Python path includes: {sys.path[0]}")

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

try:
    from ml.src.data_loading.loaders import load_all_years
    from ml.src.features.engineer import engineer_candle_features
    from ml.src.targets.target_maker import make_target
    from ml.src.sequences.sequencer import create_sequences
    print("[OK] All imports successful!")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def debug_data_quality():
    """Run comprehensive data quality checks."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("DATA QUALITY DEBUG - XAU/USD 2024")
    logger.info("="*80)
    
    # 1. Load data
    logger.info("\n[1] LOADING DATA")
    data_dir = Path("ml/src/data")
    df = load_all_years(data_dir, year_filter=[2024])
    logger.info(f"✅ Loaded {len(df):,} rows from 2024")
    logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"   Columns: {list(df.columns)}")
    
    # 2. Check for NaN/Inf in raw data
    logger.info("\n[2] RAW DATA INTEGRITY")
    nan_count = df.isnull().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    logger.info(f"   NaN values: {nan_count}")
    logger.info(f"   Inf values: {inf_count}")
    for col in df.columns:
        min_val, max_val = df[col].min(), df[col].max()
        logger.info(f"   {col:8s}: [{min_val:10.2f}, {max_val:10.2f}]")
    
    # 3. Engineer features
    logger.info("\n[3] FEATURE ENGINEERING")
    window_size = 60
    features_df = engineer_candle_features(df, window_size=window_size)
    logger.info(f"✅ Engineered {features_df.shape[1]} features for {features_df.shape[0]:,} candles")
    
    # Check feature quality
    logger.info("\n   Feature Statistics:")
    nan_by_feature = features_df.isnull().sum()
    inf_by_feature = np.isinf(features_df.select_dtypes(include=[np.number])).sum()
    
    features_with_nan = nan_by_feature[nan_by_feature > 0]
    features_with_inf = inf_by_feature[inf_by_feature > 0]
    
    if len(features_with_nan) > 0:
        logger.warning(f"   ⚠️  {len(features_with_nan)} features have NaN values:")
        for feat, cnt in features_with_nan.head(10).items():
            logger.warning(f"      {feat}: {cnt} NaN")
    else:
        logger.info(f"   ✅ No NaN values in features")
    
    if len(features_with_inf) > 0:
        logger.warning(f"   ⚠️  {len(features_with_inf)} features have Inf values:")
        for feat, cnt in features_with_inf.head(10).items():
            logger.warning(f"      {feat}: {cnt} Inf")
    else:
        logger.info(f"   ✅ No Inf values in features")
    
    # Check variance by feature
    logger.info("\n   Feature Variance:")
    variances = features_df.var(numeric_only=True)
    zero_variance = variances[variances < 1e-10]
    low_variance = variances[(variances >= 1e-10) & (variances < 0.01)]
    
    if len(zero_variance) > 0:
        logger.warning(f"   ❌ {len(zero_variance)} features have ZERO variance (constant value):")
        for feat in zero_variance.head(10).index:
            logger.warning(f"      {feat}")
    
    if len(low_variance) > 0:
        logger.warning(f"   ⚠️  {len(low_variance)} features have VERY LOW variance (<0.01):")
        for feat in low_variance.head(10).index:
            logger.warning(f"      {feat}: var={variances[feat]:.2e}")
    
    if len(zero_variance) == 0 and len(low_variance) < 5:
        logger.info(f"   ✅ Most features have reasonable variance")
    
    # 4. Create targets
    logger.info("\n[4] TARGET CREATION (SL/TP Simulation)")
    targets = make_target(
        df,
        atr_multiplier_sl=0.5,
        atr_multiplier_tp=1.0,
        min_hold_minutes=10,
        max_horizon_minutes=120,
    )
    logger.info(f"✅ Created targets for {len(targets):,} candles")
    
    # Target statistics
    pos_count = int((targets == 1).sum())
    neg_count = int((targets == 0).sum())
    pos_pct = 100 * pos_count / len(targets)
    neg_pct = 100 * neg_count / len(targets)
    
    logger.info(f"   Positive class (win):  {pos_count:7,} ({pos_pct:5.2f}%)")
    logger.info(f"   Negative class (loss): {neg_count:7,} ({neg_pct:5.2f}%)")
    logger.info(f"   Class ratio: {neg_count/max(pos_count,1):.2f}:1")
    
    # Check target distribution over time
    logger.info("\n   Target Distribution by Month:")
    targets_series = pd.Series(targets, index=df.index[:len(targets)])
    monthly_dist = targets_series.resample('M').agg([
        ('total', 'count'),
        ('positive', 'sum'),
        ('pos_pct', lambda x: 100*x.sum()/len(x) if len(x) > 0 else 0),
    ])
    for month, row in monthly_dist.iterrows():
        logger.info(f"   {month.strftime('%Y-%m')}: {int(row['total']):6,} total, "
                   f"{int(row['positive']):5,} pos ({row['pos_pct']:5.2f}%)")
    
    # 5. Build sequences
    logger.info("\n[5] SEQUENCE BUILDING")
    X, y, timestamps = create_sequences(
        features_df,
        targets,
        window_size=window_size,
    )
    logger.info(f"✅ Created {len(X):,} sequences")
    logger.info(f"   X shape: {X.shape} (sequences, features)")
    logger.info(f"   y shape: {y.shape} (binary targets)")
    
    pos_in_seq = int((y == 1).sum())
    neg_in_seq = int((y == 0).sum())
    pos_pct_seq = 100 * pos_in_seq / len(y)
    
    logger.info(f"   In sequences - Positive: {pos_in_seq:6,} ({pos_pct_seq:5.2f}%)")
    logger.info(f"   In sequences - Negative: {neg_in_seq:6,} ({100-pos_pct_seq:5.2f}%)")
    
    # 6. Check feature correlation with target
    logger.info("\n[6] FEATURE-TARGET CORRELATION")
    X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
    correlations = X_df.corrwith(pd.Series(y))
    top_corr = correlations.abs().nlargest(15)
    
    logger.info(f"   Top 15 features by correlation with target:")
    for i, (feat, corr) in enumerate(top_corr.items(), 1):
        logger.info(f"   {i:2d}. {feat:10s}: {corr:+.6f}")
    
    # Check if all correlations are near zero
    max_abs_corr = correlations.abs().max()
    logger.info(f"\n   Max absolute correlation: {max_abs_corr:.6f}")
    if max_abs_corr < 0.01:
        logger.warning(f"   ⚠️  PROBLEM: All correlations are near ZERO!")
        logger.warning(f"      This means features have ZERO predictive power")
    elif max_abs_corr < 0.05:
        logger.warning(f"   ⚠️  WARNING: Correlations are very weak (<0.05)")
    else:
        logger.info(f"   ✅ Reasonable correlation detected")
    
    # 7. Apply scaling and check distribution
    logger.info("\n[7] FEATURE SCALING")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"   Scaled feature statistics:")
    logger.info(f"   Mean: {X_scaled.mean():.6f}")
    logger.info(f"   Std:  {X_scaled.std():.6f}")
    logger.info(f"   Min:  {X_scaled.min():.2f}")
    logger.info(f"   Max:  {X_scaled.max():.2f}")
    
    # 8. Time series properties
    logger.info("\n[8] TIME SERIES PROPERTIES")
    logger.info(f"   Total sequences: {len(X):,}")
    logger.info(f"   Time span: {timestamps[0]} to {timestamps[-1]}")
    logger.info(f"   ~{len(X) / 252:.1f} years of trading data (252 trading days/year)")
    
    logger.info("\n" + "="*80)
    logger.info("DEBUG SUMMARY")
    logger.info("="*80)
    
    issues = []
    if len(zero_variance) > 0:
        issues.append(f"❌ {len(zero_variance)} features have zero variance")
    if max_abs_corr < 0.01:
        issues.append(f"❌ Features have ZERO correlation with target (max={max_abs_corr:.6f})")
    if pos_pct < 20 or pos_pct > 80:
        issues.append(f"⚠️  Severe class imbalance: {pos_pct:.1f}% positive")
    if nan_count > 0:
        issues.append(f"⚠️  {nan_count} NaN values in raw data")
    
    if issues:
        logger.warning("\nPOTENTIAL ISSUES FOUND:")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("\n✅ NO MAJOR ISSUES DETECTED")
    
    logger.info("\nRECOMMENDATIONS:")
    if max_abs_corr < 0.01:
        logger.info("  1. Features are NOT predictive - investigate feature engineering")
        logger.info("  2. Check if targets are correctly generated")
        logger.info("  3. Consider simpler targets (e.g., next candle close > current close)")
    elif pos_pct < 10:
        logger.info("  1. Class imbalance is too severe - consider oversampling or different SL/TP")
        logger.info("  2. Try different ATR multipliers (e.g., 0.2 SL, 0.4 TP)")
    else:
        logger.info("  1. Data looks reasonable - problem likely in model training")
        logger.info("  2. Try different XGBoost hyperparameters")
        logger.info("  3. Try different window sizes (50, 100)")

if __name__ == "__main__":
    debug_data_quality()
