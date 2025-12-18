#!/usr/bin/env python3
"""Quick backtest on one week of data for testing.

This is a simplified version for rapid testing.
"""

import sys
from pathlib import Path

# Add project root to path
_script_dir = Path(__file__).parent
_ml_dir = _script_dir.parent
sys.path.insert(0, str(_ml_dir.parent))

# Suppress sklearn version warnings
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings('ignore', message='.*unpickle estimator.*')

# Reduce logging verbosity
import logging
logging.getLogger('ml.src.features').setLevel(logging.WARNING)
logging.getLogger('ml.src.scripts').setLevel(logging.WARNING)
logging.getLogger('ml.src.features.engineer').setLevel(logging.ERROR)  # Suppress INFO logs from engineer

import pandas as pd
import numpy as np
import argparse
from typing import Iterable, List
import datetime as _dt

from ml.src.backtesting import BacktestEngine, BacktestConfig
from ml.src.backtesting.position_sizer import PositionSizingMethod
from ml.src.data_loading import load_all_years
from ml.src.scripts.predict_sequence import load_model_artifacts
from ml.src.features.engineer_m5 import aggregate_to_m5, engineer_m5_candle_features
from ml.src.utils import PipelineConfig

print("=" * 80)
print("QUICK BACKTEST - ONE DAY OF DATA")
print("=" * 80)
# Parse CLI args for session/day selection
parser = argparse.ArgumentParser(description='Quick backtest with optional session/day filters')
parser.add_argument('--session', choices=['all', 'london', 'ny', 'asia'], default='all',
                    help='Restrict data to a trading session (hours in UTC).')
parser.add_argument('--days', type=str, default=None,
                    help='Comma-separated weekdays (Mon,Tue,...) or ISO dates (YYYY-MM-DD) to include. Example: Mon,Tue or 2023-06-01,2023-06-05')
args = parser.parse_args()

def _parse_days_arg(days_arg: str) -> List[str]:
    return [d.strip() for d in days_arg.split(',') if d.strip()]

def _is_weekday_token(tok: str) -> bool:
    return tok[:3].lower() in {'mon','tue','wed','thu','fri','sat','sun'}

def _parse_date_tokens(tokens: Iterable[str]) -> List[_dt.date]:
    out: List[_dt.date] = []
    for t in tokens:
        try:
            out.append(_dt.date.fromisoformat(t))
        except Exception:
            continue
    return out

def filter_by_session(df: pd.DataFrame, session: str) -> pd.DataFrame:
    if session == 'all' or df.empty:
        return df

    # Assumes index is UTC or timezone-naive representing UTC
    idx = df.index
    if getattr(idx, 'tz', None) is None:
        hours = idx.hour
    else:
        hours = idx.tz_convert('UTC').hour

    sessions = {
        'london': (7, 16),   # 07:00-16:00 UTC (approx London session)
        'ny': (12, 21),      # 12:00-21:00 UTC (approx New York session)
        'asia': (23, 8),     # 23:00-08:00 UTC (wrap-around Asia session)
    }

    start_h, end_h = sessions.get(session, (0, 24))
    if start_h < end_h:
        mask = (hours >= start_h) & (hours < end_h)
    else:
        # wrap-around (e.g., 23 -> 08)
        mask = (hours >= start_h) | (hours < end_h)

    return df.iloc[mask]

# Setup paths
config_obj = PipelineConfig()
data_dir = config_obj.data_dir
models_dir = config_obj.outputs_models_dir

# Load 2023 data
print("\n[1/4] Loading 2023 data...")
prices_all = load_all_years(data_dir, year_filter=[2023])
print(f"Loaded {len(prices_all):,} candles from 2023")
print(f"Period: {prices_all.index[0]} to {prices_all.index[-1]}")

# Extract target period for backtesting
print("\n[2/4] Extracting target period and preparing context...")
if args.days:
    tokens = _parse_days_arg(args.days)
    date_tokens = [t for t in tokens if not _is_weekday_token(t)]
    weekday_tokens = [t for t in tokens if _is_weekday_token(t)]

    selected = prices_all
    dates_list = _parse_date_tokens(date_tokens)
    if dates_list:
        masks = [((prices_all.index.date) == d) for d in dates_list]
        if masks:
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m
            selected = selected[combined]

    if weekday_tokens:
        wk_map = {'mon':0,'tue':1,'wed':2,'thu':3,'fri':4,'sat':5,'sun':6}
        wk_nums = {wk_map[t[:3].lower()] for t in weekday_tokens}
        selected = selected[selected.index.weekday.isin(wk_nums)]

    target_period = selected
    if target_period.empty:
        print("No data matched --days filter; exiting.")
        raise SystemExit(1)
else:
    # Default: first calendar day
    start_date = prices_all.index[0]
    end_date = start_date + pd.Timedelta(days=1)
    target_period = prices_all[(prices_all.index >= start_date) & (prices_all.index < end_date)]

print(f"Target period: {target_period.index[0]} to {target_period.index[-1]}")
print(f"Target samples: {len(target_period):,} M1 candles")

# Apply session filter to target period
if args.session and args.session != 'all':
    before = len(target_period)
    target_period = filter_by_session(target_period, args.session)
    after = len(target_period)
    print(f"Applied session='{args.session}': {before:,} -> {after:,} candles")

# Calculate required lookback for M5 context (need 260 M5 = ~1300 M1 = ~1 day)
# Add 2 days of M1 data before target period for safety
lookback_days = 2
target_start = target_period.index[0]
context_start = target_start - pd.Timedelta(days=lookback_days)

print(f"\nAdding {lookback_days} days of M1 context for M5 warmup...")
print(f"  Context starts: {context_start}")
print(f"  Target starts: {target_start}")

# Extract M1 data with context (context + target)
prices_with_context = prices_all[(prices_all.index >= context_start) & (prices_all.index <= target_period.index[-1])]
print(f"Total M1 candles with context: {len(prices_with_context):,}")

# Load model ONCE
print("\n[3/4] Loading model and preparing M5 data...")
artifacts = load_model_artifacts(models_dir)
model = artifacts['model']
scaler = artifacts['scaler']
window_size_m5 = artifacts['window_size']
threshold = artifacts['threshold']
print(f"Model window size: {window_size_m5} M5 candles")
print(f"Decision threshold: {threshold:.4f}")

# Aggregate M1→M5 ONCE (with context)
print(f"\nAggregating M1 → M5 (with context)...")
m5_data = aggregate_to_m5(prices_with_context)
print(f"✅ Aggregated to {len(m5_data):,} M5 candles ({len(prices_with_context)/len(m5_data):.1f}x compression)")

# Engineer M5 features ONCE
print(f"Engineering M5 features...")
m5_features = engineer_m5_candle_features(m5_data)
print(f"✅ Engineered {len(m5_features):,} M5 rows × {m5_features.shape[1]} features")

# Calculate minimum M5 candles needed
min_m5_candles = window_size_m5 + 200
print(f"\nMinimum M5 candles: {min_m5_candles} ({window_size_m5} window + 200 warmup)")

if len(m5_features) < min_m5_candles:
    print(f"\n❌ ERROR: Insufficient M5 data even with context!")
    print(f"  Need: {min_m5_candles} M5 candles")
    print(f"  Got: {len(m5_features)} M5 candles")
    print(f"  Try selecting more days or removing session filter")
    sys.exit(1)

# Generate predictions for TARGET PERIOD ONLY
print(f"\n[4/4] Generating predictions for target period...")
print(f"  Target: {target_period.index[0]} to {target_period.index[-1]}")

predictions_list = []

# Find M5 indices corresponding to target period
target_start_ts = target_period.index[0]
target_end_ts = target_period.index[-1]

# Get M5 timestamps that fall within target period
m5_target_mask = (m5_data.index >= target_start_ts) & (m5_data.index <= target_end_ts)
m5_target_indices = np.where(m5_target_mask)[0]

print(f"  Target M5 candles: {len(m5_target_indices):,}")
print(f"  Predictions to generate: {len(m5_target_indices):,}")

total = len(m5_target_indices)
for idx, i in enumerate(m5_target_indices):
    # Progress indicator (every 10%)
    if idx % max(total // 10, 1) == 0 or idx == 0:
        progress = int(100 * idx / total)
        print(f"  Progress: {progress}% ({idx}/{total})")
    
    # Need full context before this M5 candle
    if i < min_m5_candles:
        continue  # Skip if not enough warmup
    
    # Extract window of M5 features
    feature_window = m5_features.iloc[i-window_size_m5:i]
    
    # Flatten to 1D (model expects flattened sequence: 60 M5 × 15 features = 900)
    X = feature_window.values.flatten().reshape(1, -1)
    
    # Scale features
    X = scaler.transform(X)
    
    # Predict
    proba = model.predict_proba(X)[0, 1]
    prediction = 1 if proba >= threshold else 0
    
    # Map back to M1 timestamp (last M1 candle of this M5 candle)
    m5_timestamp = m5_data.index[i]
    
    # Find corresponding M1 timestamp in target_period
    m1_matches = target_period[target_period.index <= m5_timestamp]
    if len(m1_matches) == 0:
        continue
    m1_timestamp = m1_matches.index[-1]
    
    predictions_list.append({
        'timestamp': m1_timestamp,
        'probability': proba,
        'prediction': prediction,
        'threshold': threshold,
        'atr': None,
    })

if len(predictions_list) == 0:
    print(f"\n❌ ERROR: No predictions generated!")
    sys.exit(1)

predictions_df = pd.DataFrame(predictions_list)
predictions_df.set_index('timestamp', inplace=True)
print(f"\n✅ Generated {len(predictions_df):,} predictions")

print(f"\nBUY signals: {(predictions_df['prediction']==1).sum():,}")
print(f"HOLD signals: {(predictions_df['prediction']==0).sum():,}")

# Keep FULL M1 data for accurate SL/TP detection
print(f"\nPreparing data for backtest...")
print(f"  Predictions: {len(predictions_df):,} M5 timestamps")
print(f"  Full M1 data: {len(target_period):,} candles (for SL/TP detection)")

# Run backtest with M5 predictions but M1 price resolution
print("\nRunning backtest engine...")
config = BacktestConfig(
    initial_capital=100000,
    position_sizing=PositionSizingMethod.FIXED,
    fixed_position_size=0.3,
    spread_pips=1.03,  # Only spread cost
    slippage_pips=0.0,  # No slippage
    commission=0.0,  # No commission
    min_probability=threshold,  # Use same threshold as model
    atr_sl_multiplier=0.2,  # Same as training!
    atr_tp_multiplier=0.4,  # Same as training!
    # No max_horizon - check all available data like real trading
    save_trades=True,
    save_equity_curve=True,
)

print(f"Backtest config:")
print(f"  Threshold: {threshold:.4f}")
print(f"  Position size: {config.fixed_position_size} lots")
print(f"  SL: {config.atr_sl_multiplier} ATR")
print(f"  TP: {config.atr_tp_multiplier} ATR")
print(f"  Max horizon: {config.max_horizon_minutes} minutes")

engine = BacktestEngine(config)
# Pass predictions (M5) and full M1 prices for accurate exit detection
results = engine.run(predictions_df, target_period)

# Display results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

metrics = results['metrics']

print(f"\nReturns:")
print(f"  Total Return:      {metrics['total_return']:>10.2%}")
print(f"  Annualized Return: {metrics['annualized_return']:>10.2%}")

print(f"\nRisk-Adjusted:")
print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}")

print(f"\nDrawdown:")
print(f"  Max Drawdown:      {metrics['max_drawdown']:>10.2%}")

print(f"\nTrades:")
print(f"  Total Trades:      {metrics['total_trades']:>10,}")
print(f"  Win Rate:          {metrics['win_rate']:>10.2%}")
print(f"  Profit Factor:     {metrics['profit_factor']:>10.2f}")

print(f"\nFinal Capital:       ${results['equity_curve'].iloc[-1]:,.2f}")

print("=" * 80)
print("✅ Backtest completed!")
print("=" * 80)
