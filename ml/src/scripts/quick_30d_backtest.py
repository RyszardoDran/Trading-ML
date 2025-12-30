#!/usr/bin/env python3
"""Quick 30-day vectorized backtest using saved features + model.

Outputs concise metrics: sequences, signals, percentiles, trades, win rate, return.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def main(days: int = 30):
    models_dir = Path("ml/outputs/models")
    feats_path = Path("ml/outputs/backtest_features.pkl")

    with open(models_dir / "sequence_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(models_dir / "sequence_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "sequence_threshold.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(feats_path, "rb") as f:
        payload = pickle.load(f)

    features = np.asarray(payload["features"])
    prices = np.asarray(payload["prices"])  # aligned to features index in generator
    timestamps = np.asarray(payload["timestamps"], dtype="datetime64[ns]")

    # Determine date window
    last_ts = pd.to_datetime(timestamps[-1])
    start_ts = last_ts - pd.Timedelta(days=days)

    # Find feature rows in the requested window (timestamps are for M5 bars)
    ts_index = pd.to_datetime(timestamps)
    mask = ts_index >= start_ts
    if mask.sum() < int(meta.get("window_size", 60)):
        print("Not enough data for requested window")
        return

    # Extract slice and rebuild sequences
    features_slice = features[mask]
    prices_slice = prices[mask]
    timestamps_slice = ts_index[mask].values

    window = int(meta.get("window_size", 60))
    nf = features_slice.shape[1]
    if features_slice.shape[0] < window:
        print("Not enough rows for window")
        return

    X = np.lib.stride_tricks.sliding_window_view(features_slice, window, axis=0)
    X = X.reshape((X.shape[0], window * nf))

    prices_seq = prices_slice[window - 1 :]
    timestamps_seq = timestamps_slice[window - 1 :]

    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]
    thr = float(meta.get("threshold", 0.623))

    total_seq = len(Xs)
    n_thr = int((proba >= thr).sum())
    n_05 = int((proba >= 0.5).sum())
    pct = np.percentile(proba, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])

    # Fast vectorized trade simulation
    signals = (proba >= thr).astype(int)
    entry_idxs = np.where(signals[:-1] == 1)[0]
    entry_dates = pd.to_datetime(timestamps_seq[entry_idxs]).date
    df = pd.DataFrame({"idx": entry_idxs, "date": entry_dates})
    max_per_day = int(meta.get("max_trades_per_day", 30))
    kept = df.groupby("date").head(max_per_day)["idx"].values

    if len(kept) > 0:
        entry_prices = prices_seq[kept] * (1 + 0.0001 / 2)
        exit_prices = prices_seq[kept + 1] * (1 - 0.0001 / 2)
        entry_cost = entry_prices * (1 + 0.0005)
        exit_value = exit_prices * (1 - 0.0005)
        pnl = (exit_value - entry_cost) / entry_cost
        wins = (pnl > 0).sum()
        num_trades = len(pnl)
        win_rate = wins / num_trades * 100
        equity = 100000.0
        for r in pnl:
            equity = equity * (1 + r)
        total_return = (equity / 100000.0 - 1) * 100
    else:
        num_trades = 0
        win_rate = 0.0
        total_return = 0.0

    print(f"30d window: {start_ts.date()} -> {last_ts.date()}")
    print(f"total_sequences: {total_seq}")
    print(f"proba>=threshold ({thr:.4f}): {n_thr}")
    print(f"proba>=0.5: {n_05}")
    print("proba_percentiles:", list(map(float, pct)))
    print(f"trades_after_cap: {num_trades}")
    print(f"win_rate_pct: {win_rate:.2f}")
    print(f"total_return_pct: {total_return:.4f}")


if __name__ == "__main__":
    main(30)
