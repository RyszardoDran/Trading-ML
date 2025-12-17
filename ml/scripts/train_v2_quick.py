#!/usr/bin/env python3
"""Quick training with V2 features and OPTION A ATR multipliers (0.2/0.4)."""

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
print("=" * 100)
print("TRAINING WITH V2 FEATURES (MULTI-TIMEFRAME CONTEXT)")
print("=" * 100)

df = load_all_years(data_dir, year_filter=[2024])
print(f"\n✓ Loaded {len(df):,} rows of OHLCV data")

# Engineer features V2
print("\n[1] Engineering V2 features (multi-timeframe: M5, M15, M60)...")
features = engineer_candle_features_v2(df)
print(f"  → {features.shape[1]} features engineered")

# Create targets with OPTION A (0.2/0.4 ATR)
print("\n[2] Creating targets (OPTION A: SL=0.2 ATR, TP=0.4 ATR)...")
targets = make_target(df, atr_multiplier_sl=0.2, atr_multiplier_tp=0.4, max_horizon=120)
pos_pct = 100 * (targets == 1).sum() / len(targets)
print(f"  → {(targets==1).sum():,} positive, {(targets==0).sum():,} negative ({pos_pct:.1f}% positive)")

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

print(f"\n[3] Split data (70/15/15):")
print(f"  Train: {len(X_train):,} samples")
print(f"  Val:   {len(X_val):,} samples")
print(f"  Test:  {len(X_test):,} samples")

# Train model
print(f"\n[4] Training XGBoost model...")
model = train_xgb(X_train, y_train, X_val, y_val)
print(f"  → Model trained successfully")

# Evaluate
print(f"\n[5] Evaluating on test set...")
metrics = evaluate(model, X_test, y_test)

print("\n" + "=" * 100)
print("RESULTS - V2 FEATURES WITH OPTION A ATR")
print("=" * 100)
print(f"Threshold:         {metrics['threshold']:.4f}")
print(f"Win Rate:          {metrics['win_rate']:.2%}")
print(f"Recall:            {metrics['recall']:.2%}")
print(f"F1 Score:          {metrics['f1']:.4f}")
print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
print(f"PR-AUC:            {metrics['pr_auc']:.4f}")

# Compare to baseline
print("\n" + "=" * 100)
print("COMPARISON TO BASELINE (V1 + OLD ATR)")
print("=" * 100)
print(f"V1 (57 feat, old ATR): ROC-AUC=0.4910, Recall=7.26%, WinRate=38.54%")
print(f"V2 (15 feat, V2 ATR):  ROC-AUC={metrics['roc_auc']:.4f}, Recall={metrics['recall']:.2%}, WinRate={metrics['win_rate']:.2%}")

if metrics['roc_auc'] > 0.50:
    print(f"\n✅ IMPROVEMENT! Model performs better than random!")
else:
    print(f"\n⚠️  Model still near random. May need further feature engineering.")

print("\n" + "=" * 100)
