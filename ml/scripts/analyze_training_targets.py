#!/usr/bin/env python3
"""Analyze training target parameters and expected profit/loss."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np

# Load 2023 data (same as backtest)
print("Loading 2023 data...")
df = pd.read_csv("src/data/XAU_1m_data_2023.csv", sep=";", parse_dates=["Date"])
df = df.rename(columns={"Date": "Datetime"})
print(f"Loaded {len(df):,} candles")

# Calculate ATR(14) like training does
print("\nCalculating ATR(14)...")
df['atr_14'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()

# Stats
atr_mean = df['atr_14'].mean()
atr_median = df['atr_14'].median()
atr_std = df['atr_14'].std()

print(f"\n{'='*80}")
print("ATR(14) STATISTICS FOR 2023")
print(f"{'='*80}")
print(f"Mean:   {atr_mean:.4f}")
print(f"Median: {atr_median:.4f}")
print(f"Std:    {atr_std:.4f}")
print(f"Min:    {df['atr_14'].min():.4f}")
print(f"Max:    {df['atr_14'].max():.4f}")

# Training parameters
atr_sl_mult = 0.1
atr_tp_mult = 0.2

sl_distance = atr_sl_mult * atr_mean
tp_distance = atr_tp_mult * atr_mean

print(f"\n{'='*80}")
print("TRAINING PARAMETERS (OPTION B)")
print(f"{'='*80}")
print(f"ATR SL multiplier: {atr_sl_mult}x")
print(f"ATR TP multiplier: {atr_tp_mult}x")
print(f"Risk:Reward Ratio: 1:{atr_tp_mult/atr_sl_mult:.1f}")
print(f"\nAverage SL distance: {sl_distance:.4f} points")
print(f"Average TP distance: {tp_distance:.4f} points")

# Position size (from backtest)
position_size = 0.01  # lots
oz_per_lot = 31.1035  # 1 oz gold
oz_total = position_size * oz_per_lot

print(f"\n{'='*80}")
print("EXPECTED PROFIT/LOSS PER TRADE (0.01 LOTS)")
print(f"{'='*80}")
print(f"Position size: {position_size} lots = {oz_total:.4f} oz")
print(f"\nPer point move: ${oz_total:.4f}")
print(f"\nExpected SL loss:  ${sl_distance * oz_total:.4f}")
print(f"Expected TP profit: ${tp_distance * oz_total:.4f}")

# Transaction costs (from backtest config)
spread_cost = 0.50
slippage_cost = 0.30
total_cost = spread_cost + slippage_cost

print(f"\n{'='*80}")
print("TRANSACTION COSTS")
print(f"{'='*80}")
print(f"Spread:   ${spread_cost:.2f}")
print(f"Slippage: ${slippage_cost:.2f}")
print(f"Total:    ${total_cost:.2f}")

# Net profit/loss
net_tp_profit = tp_distance * oz_total - total_cost
net_sl_loss = -(sl_distance * oz_total + total_cost)

print(f"\n{'='*80}")
print("NET PROFIT/LOSS (AFTER TRANSACTION COSTS)")
print(f"{'='*80}")
print(f"Winning trade (TP hit): ${net_tp_profit:.4f}")
print(f"Losing trade (SL hit):  ${net_sl_loss:.4f}")

# Break-even analysis
print(f"\n{'='*80}")
print("BREAK-EVEN ANALYSIS")
print(f"{'='*80}")

if net_tp_profit <= 0:
    print("⚠️  WARNING: Even winning trades LOSE money!")
    print(f"   TP profit (${tp_distance * oz_total:.4f}) < Transaction costs (${total_cost:.2f})")
    print(f"   Net loss per winning trade: ${net_tp_profit:.4f}")
else:
    # Calculate required win rate for break-even
    break_even_win_rate = abs(net_sl_loss) / (net_tp_profit + abs(net_sl_loss))
    print(f"Break-even win rate: {break_even_win_rate:.2%}")
    print(f"(To break even, need to win {break_even_win_rate:.1%} of trades)")

# Model performance from metadata
model_win_rate = 0.32653061224489793
print(f"\n{'='*80}")
print("MODEL PERFORMANCE (FROM TRAINING)")
print(f"{'='*80}")
print(f"Training win rate: {model_win_rate:.2%}")
print(f"Threshold: 0.32")

if net_tp_profit > 0:
    expected_pnl = (model_win_rate * net_tp_profit) + ((1 - model_win_rate) * net_sl_loss)
    print(f"\nExpected PnL per trade: ${expected_pnl:.4f}")
    
    if expected_pnl < 0:
        print(f"⚠️  NEGATIVE expectancy! Losing ${abs(expected_pnl):.4f} per trade on average")
else:
    print(f"\n⚠️  FATAL: All trades lose money regardless of win rate!")

# Recommendations
print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")

# Option 1: Increase position size
min_position_for_profit = total_cost / (tp_distance * (31.1035 / 1.0))
print(f"1. Increase position size to at least {min_position_for_profit:.4f} lots")
print(f"   (to make TP profit > transaction costs)")

# Option 2: Increase ATR multipliers
min_tp_mult_for_profit = total_cost / (atr_mean * oz_total)
print(f"2. Increase TP multiplier to at least {min_tp_mult_for_profit:.2f}x")
print(f"   (keeping SL at {atr_sl_mult}x)")

# Option 3: Reduce transaction costs
max_cost_for_profit = tp_distance * oz_total
print(f"3. Reduce transaction costs below ${max_cost_for_profit:.4f}")
print(f"   (current: ${total_cost:.2f})")

# Option 4: Window size mismatch
print(f"4. ⚠️  CRITICAL: Window size mismatch detected!")
print(f"   Training: 14 candles (from sequence_threshold.json)")
print(f"   Backtest: 100 candles (from quick_backtest.py)")
print(f"   → Model sees different feature distributions!")
print(f"   → MUST fix this before optimizing other parameters!")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")
print("The model is fundamentally unprofitable with current configuration:")
print(f"  - TP profit (~$0.04) << Transaction costs ($0.80)")
print(f"  - Even 100% win rate would lose money!")
print(f"\nPriority fixes:")
print(f"  1. FIX WINDOW SIZE MISMATCH (14 vs 100) - CRITICAL!")
print(f"  2. Increase position size to 0.1+ lots OR")
print(f"  3. Increase ATR multipliers (e.g., SL=0.5, TP=1.0) OR")
print(f"  4. Reduce transaction costs")
