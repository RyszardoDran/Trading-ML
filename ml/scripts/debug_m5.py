#!/usr/bin/env python3
"""Check M5 data for this trade."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from ml.src.features.engineer_m5 import aggregate_to_m5

# Load M1 data
df = pd.read_csv('ml/src/data/XAU_1m_data_2023.csv', sep=';', parse_dates=['Date'])
df = df.rename(columns={'Date': 'Datetime'})
df = df.set_index('Datetime')

# Aggregate to M5
print("Aggregating M1 → M5...")
m5 = aggregate_to_m5(df)
print(f"M5 data: {len(m5)} candles\n")

# Find entry in M5
entry_time = pd.Timestamp('2023-03-16 15:45:00')
sl_price = 1922.6654
tp_price = 1928.4391

# Get M5 candles around entry
entry_idx = m5.index.get_loc(entry_time)
future_m5 = m5.iloc[entry_idx:entry_idx+200]  # Next 200 M5 candles (1000 min)

print(f"Entry: {entry_time}")
print(f"Entry M5 Close: {m5.loc[entry_time]['Close']:.2f}")
print(f"SL: {sl_price:.2f}, TP: {tp_price:.2f}\n")

print(f"Checking M5 candles for TP/SL hit...\n")

for i, (ts, row) in enumerate(future_m5.iloc[1:20].iterrows()):  # Check first 20 M5
    if row['Low'] <= sl_price or row['High'] >= tp_price:
        print(f"[{i+1}] {ts} - Low={row['Low']:.2f}, High={row['High']:.2f}, Close={row['Close']:.2f}")
        if row['Low'] <= sl_price:
            print(f"     ❌ SL HIT")
        if row['High'] >= tp_price:
            print(f"     ✅ TP HIT")
        
        duration = (ts - entry_time).total_seconds() / 60
        print(f"     Duration: {duration:.0f} minutes\n")
        
        if row['High'] >= tp_price:
            break
