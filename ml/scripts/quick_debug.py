#!/usr/bin/env python3
"""Quick debug: Check feature-target correlation."""

import sys
import os
from pathlib import Path

repo_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_dir))
os.chdir(repo_dir)

import numpy as np
import pandas as pd

# Simple loading without fancy imports
data_file = Path("ml/src/data/XAU_1m_data_2024.csv")
print(f"Loading {data_file}...")

df = pd.read_csv(data_file, sep=';', index_col='Date', parse_dates=True)
print(f"[OK] Loaded {len(df):,} rows")

# Quick feature creation - just use OHLC
print("\n[1] RAW DATA STATS")
print(f"Columns: {list(df.columns)}")
print(f"Close range: [{df['Close'].min():.2f}, {df['Close'].max():.2f}]")

# Simple target: does next candle close higher?
print("\n[2] SIMPLE TARGET (next_close > current_close)")
df['next_close'] = df['Close'].shift(-1)
df['simple_target'] = (df['next_close'] > df['Close']).astype(int)
simple_pos = df['simple_target'].sum()
simple_pct = 100 * simple_pos / len(df)
print(f"Positive: {simple_pos:,} ({simple_pct:.1f}%)")
print(f"Negative: {len(df)-simple_pos:,} ({100-simple_pct:.1f}%)")

# ATR-based target (like training does)
print("\n[3] ATR-BASED TARGET (SL=0.5, TP=1.0)")
df['atr_14'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
atr_sl = 0.5 * df['atr_14']
atr_tp = 1.0 * df['atr_14']

# Simulate: did price hit TP or SL first?
atr_targets = []
for i in range(len(df)-120):  # Look ahead 120 candles
    current_close = df['Close'].iloc[i]
    
    # Check next 120 candles
    future_prices = df['High'].iloc[i+1:i+121]
    future_lows = df['Low'].iloc[i+1:i+121]
    
    if len(future_prices) < 5:
        atr_targets.append(0)
        continue
    
    # Did we hit TP first?
    tp_price = current_close + atr_tp.iloc[i]
    if (future_prices >= tp_price).any():
        atr_targets.append(1)
    # Or hit SL first?
    elif (future_lows <= current_close - atr_sl.iloc[i]).any():
        atr_targets.append(0)
    else:
        atr_targets.append(0)

atr_targets = np.array(atr_targets + [0]*(len(df)-len(atr_targets)))
atr_pos = atr_targets.sum()
atr_pct = 100 * atr_pos / len(atr_targets)
print(f"Positive: {atr_pos:,} ({atr_pct:.1f}%)")
print(f"Negative: {len(atr_targets)-atr_pos:,} ({100-atr_pct:.1f}%)")

# Basic features
print("\n[4] FEATURE VARIANCE")
features = pd.DataFrame()
features['open_close'] = (df['Open'] - df['Close']) / df['Close']
features['high_low'] = (df['High'] - df['Low']) / df['Close']
features['close_prev'] = df['Close'].pct_change()
features['volume'] = df['Volume'].pct_change().fillna(0)
features['atr_ratio'] = df['atr_14'] / df['Close']

for col in features.columns:
    var = features[col].var()
    mean = features[col].mean()
    print(f"  {col:15s}: variance={var:10.4f}, mean={mean:+.6f}")

# Correlation
print("\n[5] FEATURE-TARGET CORRELATION")
print("Simple target (next_close > current):")
for col in features.columns:
    corr = features[col].corr(df['simple_target'].fillna(0))
    print(f"  {col:15s}: {corr:+.6f}")

print("\nATR target (TP/SL simulation):")
for col in features.columns:
    corr = features[col].corr(pd.Series(atr_targets, index=df.index))
    print(f"  {col:15s}: {corr:+.6f}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("If all correlations are near zero (<0.05), features are not predictive")
print("If simple target has similar distribution to ATR target, labels OK")
print("If ATR target is heavily skewed (>70% pos or <30% pos), need different ATR")
