#!/usr/bin/env python3
"""Debug backtest exit logic."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

# Load 2023 data
df = pd.read_csv('ml/src/data/XAU_1m_data_2023.csv', sep=';', parse_dates=['Date'])
df = df.rename(columns={'Date': 'Datetime'})
df = df.set_index('Datetime')

# Simulate exact backtest logic
entry_time = pd.Timestamp('2023-03-16 15:45:00')
entry_price = 1924.59
sl_price = 1922.6654
tp_price = 1928.4391

# Get future prices like backtest does
prices = df[['Open', 'High', 'Low', 'Close', 'Volume']]
entry_idx = prices.index.get_loc(entry_time)

print(f"Entry index: {entry_idx}")
print(f"Total prices: {len(prices)}")

# Get ALL future prices (like backtest)
future_prices = prices.iloc[entry_idx:]
print(f"Future prices shape: {len(future_prices)}")
print(f"From: {future_prices.index[0]} to {future_prices.index[-1]}")

# Simulate backtest loop
exit_time = None
exit_price = None
is_win = None

print(f"\n{'='*80}")
print("SIMULATING BACKTEST LOOP")
print(f"{'='*80}")
print(f"SL: {sl_price:.2f}, TP: {tp_price:.2f}\n")

for idx in range(1, min(len(future_prices), 50)):  # Check first 50 candles
    candle = future_prices.iloc[idx]
    candle_time = future_prices.index[idx]
    
    sl_hit = candle['Low'] <= sl_price
    tp_hit = candle['High'] >= tp_price
    
    if sl_hit or tp_hit:
        print(f"[{idx}] {candle_time} - Low={candle['Low']:.2f}, High={candle['High']:.2f}")
        print(f"     SL hit: {sl_hit}, TP hit: {tp_hit}")
        
        if sl_hit and tp_hit:
            print(f"     BOTH HIT - checking Open distance")
            open_price = candle['Open']
            dist_to_tp = abs(open_price - tp_price)
            dist_to_sl = abs(open_price - sl_price)
            print(f"     Open: {open_price:.2f}, dist to TP: {dist_to_tp:.2f}, dist to SL: {dist_to_sl:.2f}")
            
            if dist_to_tp < dist_to_sl:
                exit_time = candle_time
                exit_price = tp_price
                is_win = True
                print(f"     ✅ TP HIT FIRST")
            else:
                exit_time = candle_time
                exit_price = sl_price
                is_win = False
                print(f"     ❌ SL HIT FIRST")
            break
        
        elif sl_hit:
            exit_time = candle_time
            exit_price = sl_price
            is_win = False
            print(f"     ❌ SL HIT")
            break
        
        elif tp_hit:
            exit_time = candle_time
            exit_price = tp_price
            is_win = True
            print(f"     ✅ TP HIT")
            break

if exit_time:
    duration = (exit_time - entry_time).total_seconds() / 60
    print(f"\n{'='*80}")
    print(f"EXIT FOUND:")
    print(f"  Time: {exit_time}")
    print(f"  Duration: {duration:.0f} minutes")
    print(f"  Price: {exit_price:.2f}")
    print(f"  Result: {'WIN' if is_win else 'LOSS'}")
    print(f"{'='*80}")
else:
    print(f"\n❌ NO EXIT FOUND in first 50 candles")
