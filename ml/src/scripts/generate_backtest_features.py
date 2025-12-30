#!/usr/bin/env python3
"""Generate M5 features from raw M1 CSVs and save for backtesting.

Saves a pickle with keys: 'features' (ndarray), 'prices' (list), 'timestamps' (list of ISO strings), 'columns' (list)

Usage:
    python ml/src/scripts/generate_backtest_features.py --years 2021 2022 2023 2024 2025
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from ml.src.data_loading import load_all_years
from ml.src.features.engineer_m5 import engineer_m5_candle_features, aggregate_to_m5


def main():
    parser = argparse.ArgumentParser(description="Generate M5 features from M1 CSVs")
    parser.add_argument("--data-dir", type=Path, default=Path("ml/src/data"))
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--output", type=Path, default=Path("ml/outputs/backtest_features.pkl"))
    args = parser.parse_args()

    data_dir = args.data_dir
    years = args.years
    out_path = args.output

    print(f"Loading M1 CSVs from {data_dir} for years {years}...")
    df_m1 = load_all_years(data_dir, year_filter=years)
    print(f"Loaded {len(df_m1)} M1 candles")

    # Engineer M5 features
    print("Generating M5 features (this may take a while)...")
    features_m5 = engineer_m5_candle_features(df_m1)

    # Also build M5 price series from aggregation and align to features index
    df_m5 = aggregate_to_m5(df_m1)
    # Align prices/timestamps to the final features index to avoid length mismatch
    aligned_index = features_m5.index.intersection(df_m5.index)
    df_m5_aligned = df_m5.reindex(aligned_index)
    prices = df_m5_aligned["Close"].astype(float).values
    timestamps = list(df_m5_aligned.index.astype(str))

    payload = {
        "features": features_m5.values,
        "prices": prices.tolist(),
        "timestamps": timestamps,
        "columns": list(features_m5.columns),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved features to {out_path} (rows={features_m5.shape[0]}, cols={features_m5.shape[1]})")


if __name__ == "__main__":
    main()
