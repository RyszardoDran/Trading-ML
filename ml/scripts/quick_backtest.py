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
from ml.src.scripts.predict_sequence import load_model_artifacts, predict
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
prices = load_all_years(data_dir, year_filter=[2023])
print(f"Loaded {len(prices):,} candles from 2023")
print(f"Period: {prices.index[0]} to {prices.index[-1]}")

# Take first DAY only (for speed)

# Take first DAY only (for speed) OR apply user day/session filters
print("\n[2/4] Extracting day(s) and applying filters...")
if args.days:
    tokens = _parse_days_arg(args.days)
    # Separate date tokens from weekday tokens
    date_tokens = [t for t in tokens if not _is_weekday_token(t)]
    weekday_tokens = [t for t in tokens if _is_weekday_token(t)]

    selected = prices
    # Filter by explicit dates if provided
    dates_list = _parse_date_tokens(date_tokens)
    if dates_list:
        masks = [((prices.index.date) == d) for d in dates_list]
        if masks:
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m
            selected = selected[combined]

    # Filter by weekdays if provided (Mon,Tue,...)
    if weekday_tokens:
        wk_map = {'mon':0,'tue':1,'wed':2,'thu':3,'fri':4,'sat':5,'sun':6}
        wk_nums = {wk_map[t[:3].lower()] for t in weekday_tokens}
        selected = selected[selected.index.weekday.isin(wk_nums)]

    week_prices = selected
    if week_prices.empty:
        print("No data matched --days filter; exiting.")
        raise SystemExit(1)
else:
    # Default behaviour: use the very first calendar day in the dataset
    start_date = prices.index[0]
    end_date = start_date + pd.Timedelta(days=1)
    week_prices = prices[(prices.index >= start_date) & (prices.index < end_date)]

print(f"Day(s) span: {week_prices.index[0]} to {week_prices.index[-1]}")
print(f"Samples: {len(week_prices):,} candles")

# Apply session filter (if any)
if args.session and args.session != 'all':
    before = len(week_prices)
    week_prices = filter_by_session(week_prices, args.session)
    after = len(week_prices)
    print(f"Applied session='{args.session}': {before:,} -> {after:,} candles")

# Load model ONCE (not in loop!)
print("\n[3/4] Loading model...")
artifacts = load_model_artifacts(models_dir)
window_size = artifacts['window_size']
print(f"Model window size: {window_size}")

# Generate predictions
print(f"\n[4/4] Generating predictions for {len(week_prices)-window_size:,} windows...")
predictions_list = []
error_count = 0
last_error = None

total = len(week_prices) - window_size
for i in range(window_size, len(week_prices)):
    # Progress indicator (every 10%)
    if i % max(total // 10, 1) == 0 or i == window_size:
        progress = int(100 * (i - window_size) / total)
        print(f"  Progress: {progress}% ({i-window_size}/{total})")
    
    window_data = week_prices.iloc[i-window_size:i]
    
    try:
        # Pass pre-loaded artifacts to avoid reloading model each time
        result = predict(window_data, models_dir, artifacts=artifacts)
        
        # Calculate ATR from last candle for SL/TP
        atr_value = window_data['atr_14'].iloc[-1] if 'atr_14' in window_data.columns else None
        
        predictions_list.append({
            'timestamp': week_prices.index[i],
            'probability': result['probability'],
            'prediction': result['prediction'],
            'threshold': result['threshold'],
            'atr': atr_value,
        })
    except Exception as e:
        error_count += 1
        last_error = str(e)
        if error_count == 1:
            # Print first error for debugging
            print(f"\n⚠️  First prediction error at index {i}: {e}")
            print(f"Window data shape: {window_data.shape}")
            print(f"Window columns: {window_data.columns.tolist()}")
        continue

if len(predictions_list) == 0:
    print(f"\n❌ ERROR: No predictions generated!")
    print(f"Total errors: {error_count}")
    print(f"Last error: {last_error}")
    sys.exit(1)

predictions_df = pd.DataFrame(predictions_list)
predictions_df.set_index('timestamp', inplace=True)
print(f"\n✅ Generated {len(predictions_df):,} predictions")

print(f"\nBUY signals: {(predictions_df['prediction']==1).sum():,}")
print(f"HOLD signals: {(predictions_df['prediction']==0).sum():,}")

# Run backtest
print("\nRunning backtest engine...")
config = BacktestConfig(
    initial_capital=100000,
    position_sizing=PositionSizingMethod.FIXED,
    fixed_position_size=0.3,  # Increased from 0.1 to 0.3 lots (need ~3x profit to beat costs)
    spread_pips=0.5,
    slippage_pips=0.3,
    min_probability=0.5,
    save_trades=True,
    save_equity_curve=True,
)

engine = BacktestEngine(config)
results = engine.run(predictions_df, week_prices)

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
