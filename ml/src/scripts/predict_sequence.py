from __future__ import annotations
"""Prediction script for sequence-based XAU/USD model.

Purpose:
- Load trained sequence model from artifacts
- Accept 100 previous candles as input
- Engineer features matching training pipeline
- Return probability of successful trade + expected win rate

Usage:
    # Predict from CSV file with 100 candles
    python predict_sequence.py --input-csv latest_100_candles.csv
    
    # Predict from live data directory (takes last 100 candles)
    python predict_sequence.py --data-dir ml/src/data
    
Output:
    {
        "probability": 0.73,
        "prediction": 1,
        "threshold": 0.45,
        "expected_win_rate": 0.68,
        "confidence": "high"
    }
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_model_artifacts(models_dir: Path) -> Dict:
    """Load trained model and metadata.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Dictionary with model, feature_columns, threshold, win_rate, window_size

    Raises:
        FileNotFoundError: If artifacts are missing
    """
    model_path = models_dir / "sequence_xgb_model.pkl"
    features_path = models_dir / "sequence_feature_columns.json"
    metadata_path = models_dir / "sequence_threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {features_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": metadata["threshold"],
        "win_rate": metadata["win_rate"],
        "window_size": metadata["window_size"],
    }


def validate_input_candles(df: pd.DataFrame, required_size: int) -> None:
    """Validate input candles have correct schema and size.

    Args:
        df: DataFrame with OHLCV data
        required_size: Expected number of candles

    Raises:
        ValueError: On validation failures
    """
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(df) != required_size:
        raise ValueError(f"Expected {required_size} candles, got {len(df)}")

    # Check for NaN/inf
    if df[list(required_cols)].isnull().any().any():
        raise ValueError("Input contains NaN values")
    if np.isinf(df[list(required_cols)].values).any():
        raise ValueError("Input contains infinite values")

    # Check price constraints
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC contains non-positive values")
    if (df["High"] < df["Low"]).any():
        raise ValueError("Price inconsistency: High < Low detected")


def engineer_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer per-candle features matching training pipeline.

    This MUST match the feature engineering in sequence_training_pipeline.py

    Args:
        df: OHLCV DataFrame with datetime index

    Returns:
        DataFrame with per-candle features

    Raises:
        ValueError: If feature engineering fails
    """
    df = df.copy()
    close = df["Close"].astype(float).clip(lower=1e-9)
    open_ = df["Open"].astype(float).clip(lower=1e-9)
    high = df["High"].astype(float).clip(lower=1e-9)
    low = df["Low"].astype(float).clip(lower=1e-9)
    volume = df["Volume"].astype(float).clip(lower=1e-9)

    # Log returns
    logc = np.log(close)
    ret1 = logc.diff(1)

    # Candle structure
    range_ = high - low
    range_safe = range_.replace(0, np.nan)
    
    range_n = range_ / close
    body_ratio = np.abs(close - open_) / (range_safe + 1e-9)
    upper_shadow = (high - np.maximum(open_, close)) / (range_safe + 1e-9)
    lower_shadow = (np.minimum(open_, close) - low) / (range_safe + 1e-9)

    # Volume features
    vol_change = np.log(volume / volume.shift(1).replace(0, np.nan))

    # Rolling features
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema_spread_n = (ema12 - ema26) / close

    # Simplified RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi14 = 100 - (100 / (1 + rs))

    # Volatility
    vol20 = ret1.rolling(20, min_periods=1).std()
    
    # ATR (Average True Range)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = true_range.rolling(14, min_periods=1).mean()
    atr_n = atr_14 / close  # Normalized ATR

    # Time features
    hour = df.index.hour
    minute = df.index.minute
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    features = pd.DataFrame(
        {
            "ret_1": ret1,
            "range_n": range_n,
            "body_ratio": body_ratio,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "vol_change": vol_change,
            "ema_spread_n": ema_spread_n,
            "rsi_14": rsi14,
            "vol_20": vol20,
            "atr_14": atr_14,
            "atr_n": atr_n,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "minute_sin": minute_sin,
            "minute_cos": minute_cos,
        },
        index=df.index,
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN with forward fill for prediction (no future data)
    features = features.ffill().fillna(0)
    
    if features.empty:
        raise ValueError("Feature matrix is empty after engineering")
    
    return features


def predict(
    candles: pd.DataFrame,
    models_dir: Path,
) -> Dict[str, float]:
    """Predict probability and win rate from 100 candles.

    Args:
        candles: DataFrame with exactly window_size candles (OHLCV)
        models_dir: Directory containing model artifacts

    Returns:
        Dictionary with prediction results:
        {
            "probability": float,        # Model's probability of success
            "prediction": int,           # Binary prediction (0 or 1)
            "threshold": float,          # Classification threshold used
            "expected_win_rate": float,  # Expected win rate from training
            "confidence": str,           # "low", "medium", "high"
        }

    Raises:
        ValueError: On input validation failures
        FileNotFoundError: If model artifacts not found
    """
    # Load model and metadata
    artifacts = load_model_artifacts(models_dir)
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]
    threshold = artifacts["threshold"]
    win_rate = artifacts["win_rate"]
    window_size = artifacts["window_size"]

    # Validate input
    validate_input_candles(candles, window_size)

    # Engineer features
    logger.info("Engineering features...")
    features = engineer_candle_features(candles)
    
    if len(features) != window_size:
        raise ValueError(f"Feature engineering produced {len(features)} rows, expected {window_size}")

    # Verify feature alignment
    if list(features.columns) != feature_columns:
        logger.warning("Feature columns mismatch; attempting to reorder")
        try:
            features = features[feature_columns]
        except KeyError as e:
            raise ValueError(f"Feature mismatch: {e}")

    # Flatten to model input format
    X = features.values.flatten().reshape(1, -1).astype(np.float32)

    # Predict
    logger.info("Running prediction...")
    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba >= threshold)

    # Confidence level
    if proba >= 0.7 or proba <= 0.3:
        confidence = "high"
    elif proba >= 0.55 or proba <= 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    result = {
        "probability": float(proba),
        "prediction": prediction,
        "threshold": float(threshold),
        "expected_win_rate": float(win_rate),
        "confidence": confidence,
    }

    return result


def load_candles_from_csv(csv_path: Path, n_candles: int = 100) -> pd.DataFrame:
    """Load last n_candles from CSV file.

    Args:
        csv_path: Path to CSV file with OHLCV data
        n_candles: Number of candles to load (from end)

    Returns:
        DataFrame with last n_candles, datetime indexed

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If insufficient data
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        sep=";",
        parse_dates=["Date"],
        encoding="utf-8",
    )
    
    if "Date" not in df.columns:
        raise ValueError(f"File {csv_path} missing 'Date' column")
    
    df = df.set_index("Date")
    df.sort_index(inplace=True)
    
    if len(df) < n_candles:
        raise ValueError(f"Insufficient data: need {n_candles} candles, got {len(df)}")
    
    return df.tail(n_candles)


def load_latest_candles_from_dir(data_dir: Path, n_candles: int = 100) -> pd.DataFrame:
    """Load last n_candles from all CSV files in directory.

    Args:
        data_dir: Directory containing XAU_1m_data_*.csv files
        n_candles: Number of candles to load (from end)

    Returns:
        DataFrame with last n_candles, datetime indexed

    Raises:
        FileNotFoundError: If no data files found
        ValueError: If insufficient data
    """
    files = sorted(data_dir.glob("XAU_1m_data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp, sep=";", parse_dates=["Date"], encoding="utf-8")
        df = df.set_index("Date")
        dfs.append(df)

    combined = pd.concat(dfs, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    if len(combined) < n_candles:
        raise ValueError(f"Insufficient data: need {n_candles} candles, got {len(combined)}")

    return combined.tail(n_candles)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict from 100 candles using sequence model")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to CSV file with candles (will use last 100)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with data files (will use last 100 candles from all files)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing model artifacts (default: ml/src/models)",
    )
    args = parser.parse_args()

    try:
        # Determine models directory
        if args.models_dir:
            models_dir = Path(args.models_dir)
        else:
            models_dir = Path(__file__).parent.parent / "models"

        # Load candles
        if args.input_csv:
            logger.info(f"Loading candles from {args.input_csv}...")
            candles = load_candles_from_csv(Path(args.input_csv))
        elif args.data_dir:
            logger.info(f"Loading latest candles from {args.data_dir}...")
            candles = load_latest_candles_from_dir(Path(args.data_dir))
        else:
            # Default: use data directory
            data_dir = Path(__file__).parent.parent / "data"
            logger.info(f"Loading latest candles from {data_dir}...")
            candles = load_latest_candles_from_dir(data_dir)

        logger.info(f"Loaded {len(candles)} candles from {candles.index.min()} to {candles.index.max()}")

        # Predict
        result = predict(candles, models_dir)

        # Output
        print("\n" + "=" * 60)
        print("PREDICTION RESULT - SEQUENCE MODEL")
        print("=" * 60)
        print(f"Probability:       {result['probability']:.4f} ({result['probability']:.2%})")
        print(f"Prediction:        {'BUY (1)' if result['prediction'] == 1 else 'NO TRADE (0)'}")
        print(f"Threshold:         {result['threshold']:.2f}")
        print(f"Expected Win Rate: {result['expected_win_rate']:.4f} ({result['expected_win_rate']:.2%})")
        print(f"Confidence:        {result['confidence'].upper()}")
        print("=" * 60)
        
        if result['prediction'] == 1:
            print(f"\n✅ MODEL RECOMMENDS: BUY")
            print(f"   Based on the last 100 candles, the model predicts a {result['probability']:.2%} chance")
            print(f"   of achieving the target return. Historical win rate on test data: {result['expected_win_rate']:.2%}")
        else:
            print(f"\n❌ MODEL RECOMMENDS: NO TRADE")
            print(f"   Probability {result['probability']:.2%} is below threshold {result['threshold']:.2f}")
        
        print("=" * 60)

        # JSON output for programmatic use
        print("\nJSON Output:")
        print(json.dumps(result, indent=2))

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nError: {e}")
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"\nError: {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print(f"\nError: {e}")
        raise
