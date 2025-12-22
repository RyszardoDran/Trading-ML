#!/usr/bin/env python3
"""Simple backtest for the newly trained model."""

import sys
import os
from pathlib import Path

# Setup path properly
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Now import ML modules
from ml.src.backtesting import BacktestEngine, BacktestConfig
from ml.src.data_loading import load_all_years
from ml.src.scripts.predict_sequence import load_model_artifacts
from ml.src.features.engineer_m5 import aggregate_to_m5, engineer_m5_candle_features
from ml.src.utils import PipelineConfig
import pandas as pd
import numpy as np

logger.info("=" * 80)
logger.info("BACKTEST WITH NEWLY TRAINED MODEL (TP=2.0×ATR)")
logger.info("=" * 80)

# Load model artifacts
logger.info("\n[1] Loading trained model...")
model, scaler, feature_columns, metadata, threshold = load_model_artifacts()
logger.info(f"✅ Model loaded. Threshold: {threshold:.4f}")

# Load data
logger.info("\n[2] Loading 2024 data...")
config = PipelineConfig()
df_m1 = load_all_years(config.data_dir, year_filter=[2024])
logger.info(f"✅ Loaded {len(df_m1):,} M1 candles")

# Aggregate to M5
logger.info("\n[3] Aggregating to M5...")
df_m5 = aggregate_to_m5(df_m1)
logger.info(f"✅ Aggregated to {len(df_m5):,} M5 candles")

# Engineer features
logger.info("\n[4] Engineering features...")
features = engineer_m5_candle_features(df_m5)
logger.info(f"✅ Engineered {features.shape[1]} features for {len(features):,} candles")

# Scale features
logger.info("\n[5] Scaling features...")
X_scaled = scaler.transform(features[feature_columns])
logger.info(f"✅ Scaled to shape {X_scaled.shape}")

# Make predictions
logger.info("\n[6] Generating predictions...")
y_pred_proba = model.predict_proba(X_scaled)[:, 1]
y_pred = (y_pred_proba >= threshold).astype(int)
logger.info(f"✅ Generated predictions")
logger.info(f"   Positive predictions: {y_pred.sum():,} ({100*y_pred.mean():.2f}%)")
logger.info(f"   Avg probability: {y_pred_proba.mean():.4f}")

# Setup backtest
logger.info("\n[7] Running backtest...")
backtest_config = BacktestConfig(
    initial_capital=10000,
    position_size_pct=0.1,
    position_sizing_method='fixed_pct'
)
engine = BacktestEngine(backtest_config)

# Align data
valid_idx = features.index
prices = df_m5.loc[valid_idx, 'Close'].values
timestamps = df_m5.loc[valid_idx].index

# Run backtest
results = engine.run(
    predictions=y_pred,
    prices=prices,
    timestamps=timestamps
)

# Print results
logger.info("\n" + "=" * 80)
logger.info("BACKTEST RESULTS")
logger.info("=" * 80)
logger.info(f"Initial Capital:       ${backtest_config.initial_capital:,.2f}")
logger.info(f"Final Equity:          ${results['final_equity']:,.2f}")
logger.info(f"Total Return:          {results['total_return']:.2%}")
logger.info(f"Total Trades:          {results['total_trades']}")
logger.info(f"Winning Trades:        {results['winning_trades']}")
logger.info(f"Win Rate:              {results['win_rate']:.2%}")
logger.info(f"Sharpe Ratio:          {results['sharpe_ratio']:.2f}")
logger.info(f"Max Drawdown:          {results['max_drawdown']:.2%}")
logger.info(f"Profit Factor:         {results['profit_factor']:.2f}")

logger.info("\n✅ Backtest complete!")
