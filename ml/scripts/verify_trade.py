#!/usr/bin/env python3
"""Verify specific backtest trade against real data."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

# Load 2023 data
print("Loading 2023 M1 data...")
df = pd.read_csv('ml/src/data/XAU_1m_data_2023.csv', sep=';', parse_dates=['Date'])
df = df.rename(columns={'Date': 'Datetime'})
df = df.set_index('Datetime')

# Trade details from backtest
entry_time = pd.Timestamp('2023-03-16 15:45:00')
entry_price = 1924.59
sl_price = 1922.6654
tp_price = 1928.4391
expected_exit_time = entry_time + pd.Timedelta(minutes=915)  # ~2023-03-17 06:00

print(f"\n{'='*80}")
print("BACKTEST TRADE VERIFICATION")
print(f"{'='*80}")

# Check entry candle
print(f"\n[1] ENTRY CANDLE: {entry_time}")
if entry_time in df.index:
    entry_candle = df.loc[entry_time]
    print(f"    Open:  {entry_candle['Open']:.2f}")
    print(f"    High:  {entry_candle['High']:.2f}")
    print(f"    Low:   {entry_candle['Low']:.2f}")
    print(f"    Close: {entry_candle['Close']:.2f}")
    print(f"    ✅ Entry price match: {abs(entry_candle['Close'] - entry_price) < 0.01}")
else:
    print(f"    ❌ Entry time not found in data!")
    sys.exit(1)

# Get future candles
print(f"\n[2] CHECKING FUTURE CANDLES (next 915 minutes)")
print(f"    SL level: {sl_price:.2f}")
print(f"    TP level: {tp_price:.2f}")

future_df = df.loc[entry_time:expected_exit_time + pd.Timedelta(hours=1)]
print(f"    Found {len(future_df)} candles from {future_df.index[0]} to {future_df.index[-1]}")

# Check when SL/TP hit
sl_hit = None
tp_hit = None

for idx, row in future_df.iloc[1:].iterrows():  # Skip first candle (entry)
    if sl_hit is None and row['Low'] <= sl_price:
        sl_hit = idx
    if tp_hit is None and row['High'] >= tp_price:
        tp_hit = idx
    
    # Stop if both hit
    if sl_hit and tp_hit:
        break

print(f"\n[3] EXIT ANALYSIS")
if tp_hit and (sl_hit is None or tp_hit <= sl_hit):
    print(f"    ✅ TP HIT FIRST at {tp_hit}")
    print(f"    Duration: {(tp_hit - entry_time).total_seconds() / 60:.0f} minutes")
    tp_candle = df.loc[tp_hit]
    print(f"    Exit candle: High={tp_candle['High']:.2f}, Low={tp_candle['Low']:.2f}")
    print(f"    Exit price: {tp_price:.2f} (TP level)")
elif sl_hit:
    print(f"    ❌ SL HIT FIRST at {sl_hit}")
    print(f"    Duration: {(sl_hit - entry_time).total_seconds() / 60:.0f} minutes")
    sl_candle = df.loc[sl_hit]
    print(f"    Exit candle: High={sl_candle['High']:.2f}, Low={sl_candle['Low']:.2f}")
    print(f"    Exit price: {sl_price:.2f} (SL level)")
else:
    print(f"    ⚠️  NEITHER SL NOR TP HIT in available data")

# Check backtest result
print(f"\n[4] BACKTEST RESULT COMPARISON")
print(f"    Backtest exit time: ~{expected_exit_time}")
print(f"    Backtest says: TP hit, WIN")
if tp_hit:
    duration_diff = abs((tp_hit - entry_time).total_seconds() - 915*60)
    print(f"    ✅ MATCHES: TP hit at {tp_hit}")
    print(f"    Duration difference: {duration_diff/60:.0f} minutes")
else:
    print(f"    ❌ MISMATCH: Real data shows different result!")

print(f"\n{'='*80}")
