from __future__ import annotations
"""Prediction script for sequence-based XAU/USD model (M5 timeframe).

Purpose:
- Load trained sequence model from artifacts
- Accept historical M1 candles (last 7 days recommended)
- Aggregate M1 → M5 timeframe (proper OHLCV aggregation)
- Engineer features on M5 bars with M15/M60 context
- Use only last sequence_length M5 candles for prediction
- Return probability of successful trade + expected win rate

**CRITICAL: M5 TIMEFRAME ARCHITECTURE**
Model operates on M5 timeframe (5-minute bars):

**WHY 7 DAYS OF M1 DATA?**
1. ~10,080 M1 candles → ~2,016 M5 candles after aggregation
2. SMA200 needs 200 M5 candles minimum (~16.7 hours of M5 data)
3. Multi-timeframe indicators (M5→M15→M60) require historical context
4. Session features need data spanning multiple trading sessions
5. 7 days provides sufficient M5 bars for all indicators

**PROCESS:**
1. Load last 7 days of M1 candles (~10,080 candles)
2. Aggregate M1 → M5 using proper OHLCV (Open=first, High=max, Low=min, Close=last, Volume=sum)
3. Calculate features on M5 timeframe with M15/M60 context
4. Extract LAST 100 M5 candles for prediction (500 minutes = 8.3 hours context)
5. Model predicts on properly-calculated M5 features

Usage:
    # Predict from CSV file (automatically uses last 7 days of M1 data)
    python predict_sequence.py --input-csv data.csv
    
    # Predict from live data directory (takes last 7 days of M1)
    python predict_sequence.py --data-dir ml/src/data
    
    # Custom analysis window (e.g., 3 days of M1)
    python predict_sequence.py --input-csv data.csv --analysis-days 3
    
Output:
    {
        "probability": 0.73,
        "prediction": 1,
        "threshold": 0.45,
        "expected_win_rate": 0.68,
        "confidence": "high",
        "m1_candles_analyzed": 10080,
        "m5_candles_generated": 2016,
        "m5_candles_used_for_prediction": 100,
        "analysis_window_days": 7
    }
"""

import json
import logging
import pickle
import sys
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Setup imports for both contexts:
# 1. When called directly from ml/src/scripts: add ml to path
# 2. When called from pipeline: use ml.src imports
script_dir = Path(__file__).resolve()
src_dir = script_dir.parent.parent
ml_dir = src_dir.parent

# Add ml to path for package imports
sys.path.insert(0, str(ml_dir))

# Now import using ml.src package path
from src.features.engineer_m5 import aggregate_to_m5, engineer_m5_candle_features
from src.features.indicators import compute_atr
from src.filters.regime_filter import should_trade
from src.utils.risk_config import ATR_PERIOD_M5, SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER, risk_reward_ratio

# Suppress sklearn version warnings
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _compute_atr_m5(candles_m5: pd.DataFrame, period: int = ATR_PERIOD_M5) -> Optional[float]:
    if candles_m5.empty:
        return None

    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(candles_m5.columns):
        return None

    atr_series = compute_atr(
        candles_m5["High"].astype(np.float32),
        candles_m5["Low"].astype(np.float32),
        candles_m5["Close"].astype(np.float32),
        period=period,
    )

    atr_last = float(atr_series.iloc[-1])
    if not np.isfinite(atr_last) or atr_last <= 0:
        return None

    return atr_last


def load_model_artifacts(models_dir: Path) -> Dict:
    """Load trained model, scaler, and metadata.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Dictionary with model, feature_columns, threshold, win_rate, window_size, scaler

    Raises:
        FileNotFoundError: If artifacts are missing
    """
    model_path = models_dir / "sequence_xgb_model.pkl"
    features_path = models_dir / "sequence_feature_columns.json"
    metadata_path = models_dir / "sequence_threshold.json"
    scaler_path = models_dir / "sequence_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {features_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Extract analysis_window_days (default 7 days for robust indicator calculation)
    # For backwards compatibility, also check for warmup_period
    analysis_window_days = metadata.get("analysis_window_days", 7)
    
    return {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": metadata["threshold"],
        "win_rate": metadata["win_rate"],
        "window_size": metadata["window_size"],
        "analysis_window_days": analysis_window_days,
        "scaler": scaler,
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
        raise ValueError("Price inconsistency: High < Low detected")raise ValueError("Price inconsistency: High < Low detected")


def predict(
    candles: pd.DataFrame,
    models_dir: Path,
    artifacts: Optional[Dict] = None,  # NEW: optional pre-loaded artifacts
) -> Dict[str, float]:
    """Predict probability and win rate from M1 candles (aggregates to M5).

    **CRITICAL: M5 TIMEFRAME ARCHITECTURE**
    This function requires M1 candles, then aggregates to M5 timeframe:
    - 7 days M1 (~10,080 candles) → ~2,016 M5 candles
    - SMA200 needs 200 M5 candles (16.7 hours of M5 data)
    - Multi-timeframe indicators (M5→M15→M60) require historical context
    - Model operates on 100 M5 candles = 500 minutes (8.3 hours) context
    
    **RECOMMENDED**: Provide last 7 days of M1 data (~10,080 M1 candles)
    **MINIMUM**: Provide at least (window_size * 5 + 1000) M1 candles for M5 aggregation
    
    Process:
    1. Load historical M1 candles (recommend: last 7 days)
    2. Aggregate M1 → M5 using proper OHLCV aggregation
    3. Engineer features on M5 timeframe with M15/M60 context
    4. Extract LAST window_size M5 candles for prediction
    5. Model predicts on properly-calculated M5 indicators
    
    Args:
        candles: DataFrame with historical M1 OHLCV data (recommend 7 days)
        models_dir: Directory containing model artifacts
        artifacts: Optional pre-loaded artifacts (for performance)

    Returns:
        Dictionary with prediction results:
        {
            "probability": float,                      # Model's probability of success
            "prediction": int,                         # Binary prediction (0 or 1)
            "threshold": float,                        # Classification threshold used
            "expected_win_rate": float,                # Expected win rate from training
            "confidence": str,                         # "low", "medium", "high"
            "m1_candles_analyzed": int,                # Total M1 candles analyzed
            "m5_candles_generated": int,               # M5 candles after aggregation
            "m5_candles_used_for_prediction": int,     # M5 candles in model input
            "analysis_window_days": int,               # Recommended analysis window
        }

    Raises:
        ValueError: On input validation failures (insufficient candles, missing columns)
        FileNotFoundError: If model artifacts not found
        
    Examples:
        >>> # Load last 7 days of M1 data (recommended)
        >>> candles_m1 = load_candles_last_n_days(7)
        >>> result = predict(candles_m1, models_dir)
        >>> print(f"M1 analyzed: {result['m1_candles_analyzed']}")
        >>> print(f"M5 generated: {result['m5_candles_generated']}")
        >>> print(f"Probability: {result['probability']:.2%}")
    """
    # Load model and metadata (or use pre-loaded)
    if artifacts is None:
        artifacts = load_model_artifacts(models_dir)
    
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]
    saved_threshold = float(artifacts["threshold"])
    win_rate = artifacts["win_rate"]
    window_size = artifacts["window_size"]
    analysis_window_days = artifacts["analysis_window_days"]
    scaler = artifacts.get("scaler")

    # Apply optional production min threshold override via MIN_PROD_THRESHOLD env var
    min_prod_env = os.getenv("MIN_PROD_THRESHOLD")
    threshold_used = float(saved_threshold)
    threshold_source = "saved"
    if min_prod_env is not None:
        try:
            min_prod_val = float(min_prod_env)
            if min_prod_val > threshold_used:
                threshold_used = min_prod_val
                threshold_source = "env_min_override"
        except ValueError:
            logger.warning(f"Invalid MIN_PROD_THRESHOLD='{min_prod_env}'; ignoring")

    logger.info(
        f"Using threshold={threshold_used:.4f} (saved={saved_threshold:.4f}, source={threshold_source})"
    )

    if scaler is None:
        raise ValueError("Scaler artifact missing; retrain pipeline to regenerate artifacts")

    # Calculate recommended minimum candles in M1 units (use metadata if available)
    recommended_min_m1 = artifacts.get("recommended_min_candles_m1", (window_size + 200) * 5)

    # Validate input (check M1 candles provided)
    if len(candles) < window_size * 5:
        raise ValueError(
            f"Insufficient M1 candles: need at least {window_size*5} (window_size={window_size} M5 candles), got {len(candles)}"
        )

    if len(candles) < recommended_min_m1:
        logger.warning(
            f"Low candle count: {len(candles)} M1 candles provided. "
            f"Recommend at least {recommended_min_m1} M1 candles ({(recommended_min_m1//5)} M5 candles) "
            f"or {analysis_window_days} days (~{analysis_window_days * 24 * 60} M1 candles) for optimal indicator accuracy."
        )
    
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(candles.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Store original M1 candle count for reporting
    m1_candles_count = len(candles)
    
    # STEP 1: Aggregate M1 → M5 (proper OHLCV aggregation)
    logger.debug(f"Aggregating {m1_candles_count} M1 candles to M5 timeframe...")
    candles_m5 = aggregate_to_m5(candles)
    logger.debug(f"Aggregated to {len(candles_m5)} M5 candles ({len(candles_m5)/m1_candles_count*100:.1f}% compression)")

    if len(candles_m5) < window_size:
        raise ValueError(
            f"Insufficient M5 candles after aggregation: need at least {window_size}, got {len(candles_m5)}"
        )

    entry_price = float(candles_m5["Close"].iloc[-1])
    atr_m5 = _compute_atr_m5(candles_m5)
    
    # STEP 2: Engineer features on M5 timeframe with M15/M60 context
    # This ensures indicators have proper multi-timeframe context
    logger.debug(f"Engineering features on {len(candles_m5)} M5 candles (with M15/M60 context)...")
    features_full = engineer_m5_candle_features(candles_m5)
    
    # Ensure we have enough engineered rows to build the final sequence
    if len(features_full) < window_size:
        raise ValueError(
            f"Insufficient engineered feature rows: need at least {window_size}, got {len(features_full)}"
        )
    
    # STEP 3: Extract LAST window_size M5 candles for prediction
    # This gives us properly calculated indicators with full historical context
    features = features_full.tail(window_size)
    
    if len(features) != window_size:
        raise ValueError(
            f"After extracting sequence, got {len(features)} rows, expected {window_size}"
        )

    # Keep full features (for trend/regime checks) and create a separate
    # DataFrame for model input (may be reordered to match saved columns)
    features_for_model = features.copy()
    if list(features_for_model.columns) != feature_columns:
        logger.warning("Feature columns mismatch; attempting to reorder for model input")
        try:
            features_for_model = features_for_model[feature_columns]
        except KeyError as e:
            raise ValueError(f"Feature mismatch: {e}")

    rr = risk_reward_ratio()
    common_result = {
        "threshold": float(threshold_used),
        "threshold_source": threshold_source,
        "m1_candles_analyzed": m1_candles_count,
        "m5_candles_generated": len(candles_m5),
        "m5_candles_used_for_prediction": window_size,
        "analysis_window_days": analysis_window_days,
        "entry_price": float(entry_price),
        "atr_m5": float(atr_m5) if atr_m5 is not None else None,
        "sl_atr_multiplier": float(SL_ATR_MULTIPLIER),
        "tp_atr_multiplier": float(TP_ATR_MULTIPLIER),
        "rr": float(rr),
    }

    # --- TREND FILTER CHECK ---
    # Check if the latest candle is in an uptrend (Close > SMA200) AND ADX > 15 AND RSI_M5 < 75
    skip_regime_env = os.getenv("SKIP_REGIME_FILTER") is not None and os.getenv("SKIP_REGIME_FILTER").strip().lower() in ("1","true","yes","y")
    if skip_regime_env:
        logger.info("Skipping trend/pullback filters because SKIP_REGIME_FILTER is set")
    else:
        if "dist_sma_200" in features.columns and "adx" in features.columns:
            last_dist = features["dist_sma_200"].iloc[-1]
            last_adx = features["adx"].iloc[-1]
            
            if last_dist <= 0:
                logger.debug(f"Trend Filter: dist_sma_200={last_dist:.4f} (filtered)")
                return {
                    **common_result,
                    "probability": 0.0,
                    "prediction": 0,
                    "expected_win_rate": 0.0,
                    "confidence": "low (trend filter)",
                    "sl": None,
                    "tp": None,
                }
            if last_adx <= 15:
                logger.debug(f"Trend Filter: ADX={last_adx:.2f} (filtered)")
                return {
                    **common_result,
                    "probability": 0.0,
                    "prediction": 0,
                    "expected_win_rate": 0.0,
                    "confidence": "low (weak trend)",
                    "sl": None,
                    "tp": None,
                }
                
        if "rsi_m5" in features.columns:
            last_rsi = features["rsi_m5"].iloc[-1]
            if last_rsi >= 75:
                logger.debug(f"Pullback Filter: RSI_M5={last_rsi:.2f} (filtered)")
                return {
                    **common_result,
                    "probability": 0.0,
                    "prediction": 0,
                    "expected_win_rate": 0.0,
                    "confidence": "low (overbought)",
                    "sl": None,
                    "tp": None,
                }
    # --------------------------

    # Flatten to model input format using the reordered model input features
    X_raw = features_for_model.values.flatten().reshape(1, -1)
    X_scaled = scaler.transform(X_raw)
    X = X_scaled.astype(np.float32)

    # Predict
    logger.info("Running prediction...")
    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba >= threshold_used)

    # Confidence level
    if proba >= 0.7 or proba <= 0.3:
        confidence = "high"
    elif proba >= 0.55 or proba <= 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    # Regime filter: suppress trades in poor regimes (quick production safeguard)
    try:
        last_feat = features.iloc[-1]
        adx = float(last_feat.get("adx", np.nan))
        dist_sma = float(last_feat.get("dist_sma_200", np.nan))
        sma200 = entry_price - dist_sma if not pd.isna(dist_sma) else float("nan")

        allowed, regime, reason = should_trade(
            atr_m5 if atr_m5 is not None else 0.0,
            adx,
            entry_price,
            sma200,
            threshold=threshold_used,
        )
    except Exception as e:
        logger.exception("Regime check failed; proceeding without regime gating")
        allowed, regime, reason = True, "UNKNOWN", "failed_check"

    if not allowed:
        logger.info(f"Regime gating suppressed trade: regime={regime}, reason={reason}")
        prediction = 0


    sl = None
    tp = None
    if prediction == 1 and atr_m5 is not None:
        sl = float(entry_price - (SL_ATR_MULTIPLIER * atr_m5))
        tp = float(entry_price + (TP_ATR_MULTIPLIER * atr_m5))

    result = {
        **common_result,
        "probability": float(proba),
        "prediction": prediction,
        "expected_win_rate": float(win_rate),
        "confidence": confidence,
        "sl": sl,
        "tp": tp,
        "regime": regime,
        "regime_allowed": bool(allowed),
        "regime_reason": reason,
    }

    return result


def load_candles_from_csv(
    csv_path: Path, 
    n_days: Optional[int] = 7,
    n_candles: Optional[int] = None
) -> pd.DataFrame:
    """Load last n_days or n_candles from CSV file.
    
    **RECOMMENDED**: Use n_days=7 (default) for robust indicator calculation.
    7 days of M1 data = ~10,080 candles, enough for SMA200, sessions, MTF indicators.

    Args:
        csv_path: Path to CSV file with OHLCV data
        n_days: Number of days to load from end (default 7). If provided, overrides n_candles.
        n_candles: Number of candles to load (alternative to n_days)

    Returns:
        DataFrame with last n_days/n_candles, datetime indexed

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If insufficient data
        
    Examples:
        >>> # Load last 7 days (recommended, ~10,080 M1 candles)
        >>> candles = load_candles_from_csv('data.csv')
        >>> len(candles)
        10080
        
        >>> # Load specific number of candles
        >>> candles = load_candles_from_csv('data.csv', n_days=None, n_candles=500)
        >>> len(candles)
        500
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
    
    # Determine how many candles to load
    if n_days is not None:
        # Load last n_days of data
        cutoff_date = df.index.max() - pd.Timedelta(days=n_days)
        df_filtered = df[df.index >= cutoff_date]
        
        if len(df_filtered) == 0:
            raise ValueError(f"No data found in last {n_days} days")
        
        logger.info(f"Loaded {len(df_filtered)} candles from last {n_days} days")
        return df_filtered
    elif n_candles is not None:
        # Load last n_candles
        if len(df) < n_candles:
            raise ValueError(f"Insufficient data: need {n_candles} candles, got {len(df)}")
        return df.tail(n_candles)
    else:
        raise ValueError("Must specify either n_days or n_candles")raise ValueError("Must specify either n_days or n_candles")


def load_latest_candles_from_dir(
    data_dir: Path,
    n_days: Optional[int] = 7,
    n_candles: Optional[int] = None
) -> pd.DataFrame:
    """Load last n_days or n_candles from all CSV files in directory.
    
    **RECOMMENDED**: Use n_days=7 (default) for robust indicator calculation.
    7 days of M1 data = ~10,080 candles, enough for SMA200, sessions, MTF indicators.

    Args:
        data_dir: Directory containing XAU_1m_data_*.csv files
        n_days: Number of days to load from end (default 7)
        n_candles: Number of candles to load (alternative to n_days)

    Returns:
        DataFrame with last n_days/n_candles, datetime indexed

    Raises:
        FileNotFoundError: If no data files found
        ValueError: If insufficient data
        
    Examples:
        >>> # Load last 7 days (recommended)
        >>> candles = load_latest_candles_from_dir(Path('ml/src/data'))
        >>> len(candles)  # ~10,080 M1 candles
        10080
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

    # Determine how many candles to return
    if n_days is not None:
        # Load last n_days of data
        cutoff_date = combined.index.max() - pd.Timedelta(days=n_days)
        df_filtered = combined[combined.index >= cutoff_date]
        
        if len(df_filtered) == 0:
            raise ValueError(f"No data found in last {n_days} days")
        
        logger.info(f"Loaded {len(df_filtered)} candles from last {n_days} days")
        return df_filtered
    elif n_candles is not None:
        if len(combined) < n_candles:
            raise ValueError(f"Insufficient data: need {n_candles} candles, got {len(combined)}")
        return combined.tail(n_candles)
    else:
        raise ValueError("Must specify either n_days or n_candles")raise ValueError("Must specify either n_days or n_candles")


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
    parser.add_argument(
        "--skip-regime",
        action="store_true",
        help="Skip trend/regime filters for this run (temporary; overrides config/env).",
    )
    args = parser.parse_args()

    try:
        # Determine models directory
        if args.models_dir:
            models_dir = Path(args.models_dir)
        else:
            models_dir = Path(__file__).parent.parent.parent / "outputs" / "models"

        # CLI override: skip regime/trend gating if requested
        if getattr(args, "skip_regime", False):
            os.environ["SKIP_REGIME_FILTER"] = "1"
            logger.info("SKIP_REGIME_FILTER set via --skip-regime CLI flag (regime filters will be skipped)")

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
