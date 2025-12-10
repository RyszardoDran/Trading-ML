from __future__ import annotations

"""Predict on the latest candle using saved model artifacts.

Usage:
  python ml/scripts/predict_latest.py

Requires:
- Artifacts: ml/models/xgb_model.pkl, ml/models/feature_columns.json
- Data: at least one CSV in ml/src/data/XAU_1m_data_*.csv
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def engineer_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    ret1 = np.log(close).diff(1)
    ret5 = np.log(close).diff(5)
    ret15 = np.log(close).diff(15)
    vol20 = ret1.rolling(20).std()
    vol60 = ret1.rolling(60).std()
    range_n = (high - low) / close
    mom10 = ret1.rolling(10).mean()
    mom30 = ret1.rolling(30).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema_spread_n = (ema12 - ema26) / close
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi14 = 100 - (100 / (1 + rs))
    minutes = df.index.hour * 60 + df.index.minute
    day_minutes = 24 * 60
    tod_sin = np.sin(2 * np.pi * minutes / day_minutes)
    tod_cos = np.cos(2 * np.pi * minutes / day_minutes)

    X = pd.DataFrame(
        {
            "ret_1": ret1,
            "ret_5": ret5,
            "ret_15": ret15,
            "vol_20": vol20,
            "vol_60": vol60,
            "range_n": range_n,
            "mom_10": mom10,
            "mom_30": mom30,
            "ema_spread_n": ema_spread_n,
            "rsi_14": rsi14,
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
        },
        index=df.index,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    return X


def main() -> None:
    repo_root = Path(__file__).parents[2]
    models_dir = repo_root / "models"
    data_dir = repo_root / "data"

    # Load artifacts
    model_path = models_dir / "xgb_model.pkl"
    features_path = models_dir / "feature_columns.json"
    threshold_path = models_dir / "threshold.json"
    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError("Model artifacts not found. Train first via training_pipeline.py")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    # Threshold: load if exists, else default to 0.5
    threshold = 0.5
    if threshold_path.exists():
        try:
            with open(threshold_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "threshold" in data:
                    threshold = float(data["threshold"])
        except Exception:
            # Fallback silently to default, prediction still works
            pass

    # Load latest CSV
    csvs = sorted(data_dir.glob("XAU_1m_data_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    df = (
        pd.read_csv(csvs[-1], sep=";", parse_dates=["Date"])
        .rename(columns=lambda c: c.strip())
        .set_index("Date")
        .sort_index()
    )

    # Engineer features and align
    X = engineer_minimal_features(df)
    # Ensure all required columns present
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")
    X = X[feature_cols]

    latest_row = X.iloc[-1].values.reshape(1, -1)
    proba = float(model.predict_proba(latest_row)[0, 1])
    pred = int(proba >= threshold)

    print("=== Inference Result (Latest Candle) ===")
    print(f"Probability: {proba:.4f}")
    print(f"Threshold:   {threshold:.2f}")
    print(f"Predicted class (>=threshold): {pred}")


if __name__ == "__main__":
    main()