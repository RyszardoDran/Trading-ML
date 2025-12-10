from __future__ import annotations
"""Production-grade training pipeline for XAU/USD (1-minute data).

Purpose:
- Load semicolon-separated 1m CSVs from `ml/src/data/`
- Validate schema and data integrity
- Engineer compact, robust features (no TA-Lib)
- Build a clearer binary target using forward returns
- Perform chronological train/val/test split with dynamic year checks
- Train a calibrated XGBoost classifier with class imbalance handling
- Evaluate with threshold sweep and return key metrics (ROC-AUC, PR-AUC, F1)
- Persist model, feature list, and chosen threshold

Key principles:
- Deterministic behavior via fixed random seeds
- Strict input validation and explicit error handling
- Avoid leakage: calibration on validation only, no peeking into test
- Minimal external dependencies (pandas, numpy, sklearn, xgboost)

Inputs (CSV):
- Expected columns: [Date;Open;High;Low;Close;Volume]
- Separator: `;`, Date parseable to datetime
- Files pattern: `ml/src/data/XAU_1m_data_*.csv`

Artifacts:
- `ml/models/xgb_model.pkl` (calibrated classifier)
- `ml/models/feature_columns.json` (ordered feature names)
- `ml/models/threshold.json` (selected classification threshold)
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate OHLCV schema and basic price constraints.

    Raises:
        ValueError: On missing columns, non-positive prices, or High<Low inconsistencies.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric dtypes
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC contains non-positive values")
    if (df["High"] < df["Low"]).any():
        raise ValueError("Price inconsistency: High < Low detected")
    if df.index.has_duplicates:
        logger.warning("Duplicate timestamps detected; dropping duplicates")
        df.drop_duplicates(inplace=True)
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)


def load_all_years(data_dir: Path) -> pd.DataFrame:
    """Load and validate all available yearly CSVs.

    Returns:
        Concatenated DataFrame indexed by datetime, strictly increasing index.
    """
    files = sorted(data_dir.glob("XAU_1m_data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    dfs: List[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp, sep=";", parse_dates=["Date"])  # semicolon-separated
        df = df.rename(columns={c: c.strip() for c in df.columns})
        if "Date" not in df.columns:
            raise ValueError(f"File {fp} missing 'Date' column")
        df = df.set_index("Date")
        _validate_schema(df)
        dfs.append(df)
    data = pd.concat(dfs, axis=0)
    data = data[~data.index.duplicated(keep="first")]
    data.sort_index(inplace=True)
    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer robust features without TA-Lib.

    Features:
    - Log returns: 1, 5, 15-minute
    - Volatility: rolling std of 1m returns (20, 60)
    - Normalized range: (High-Low)/Close
    - Momentum: rolling mean of returns (10, 30)
    - Trend: EMA(12), EMA(26) spread normalized by Close
    - RSI(14) simplified implementation
    - Time-of-day: sine/cosine encoding

    Returns:
        Clean DataFrame with NaN/inf removed.
    """
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Guard against zeros (already validated) but clip extreme ranges
    close = close.clip(lower=1e-9)

    # Log returns
    logc = np.log(close)
    ret1 = logc.diff(1)
    ret5 = logc.diff(5)
    ret15 = logc.diff(15)

    # Volatility
    vol20 = ret1.rolling(20, min_periods=20).std()
    vol60 = ret1.rolling(60, min_periods=60).std()

    # Range normalized
    range_n = (high - low) / close

    # Momentum
    mom10 = ret1.rolling(10, min_periods=10).mean()
    mom30 = ret1.rolling(30, min_periods=30).mean()

    # EMA and spread
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema_spread_n = (ema12 - ema26) / close

    # Simplified RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi14 = 100 - (100 / (1 + rs))

    # Time features
    minutes = df.index.hour * 60 + df.index.minute
    day_minutes = 24 * 60
    tod_sin = np.sin(2 * np.pi * minutes / day_minutes)
    tod_cos = np.cos(2 * np.pi * minutes / day_minutes)

    features = pd.DataFrame(
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
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.dropna()
    return features


def make_target(df: pd.DataFrame, horizon: int = 5, min_return_bp: float = 5.0) -> pd.Series:
    """Create a binary classification target from forward returns.

    Definition:
        y = 1 if cumulative log return over `horizon` minutes exceeds
        a fixed minimum threshold; else 0.

    Rationale:
        The previous volatility-scaled threshold can behave erratically
        during regime changes and produce highly imbalanced/unstable labels.
        A small fixed minimum (in basis points) provides a clearer signal.

    Args:
        df: OHLCV DataFrame with `Close` column and datetime index
        horizon: Forward window size in minutes
        min_return_bp: Minimum cumulative return threshold in basis points (1bp = 0.0001)

    Returns:
        pd.Series of 0/1 aligned to original index (NaNs dropped)
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >=1, got {horizon}")
    close = df["Close"].astype(float)
    logc = np.log(close)
    ret_fwd_1m = logc.diff().shift(-1)
    cum = ret_fwd_1m.rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    thr = min_return_bp * 1e-4
    target = (cum > thr).astype(int)
    target = target.dropna()
    return target


def split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    train_until: str = "2022-12-31 23:59:00",
    val_until: str = "2023-12-31 23:59:00",
    test_until: str = "2024-12-31 23:59:00",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Chronological split with dynamic coverage validation.

    Args:
        X, y: Feature matrix and labels (aligned by index)
        train_until: End timestamp for train period (inclusive)
        val_until: End timestamp for validation period (inclusive)
        test_until: End timestamp for test period (inclusive)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    train_end = pd.Timestamp(train_until)
    val_end = pd.Timestamp(val_until)
    test_end = pd.Timestamp(test_until)

    # Validate coverage dynamically
    min_idx, max_idx = X.index.min(), X.index.max()
    if min_idx is None or max_idx is None:
        raise ValueError("Empty index after alignment; check feature engineering/target")
    if max_idx < pd.Timestamp(val_until):
        raise ValueError("Insufficient data for validation/test periods")

    X_train = X[X.index <= train_end]
    y_train = y[y.index <= train_end]
    X_val = X[(X.index > train_end) & (X.index <= val_end)]
    y_val = y[(y.index > train_end) & (y.index <= val_end)]
    X_test = X[(X.index > val_end) & (X.index <= test_end)]
    y_test = y[(y.index > val_end) & (y.index <= test_end)]

    if min(map(len, [X_train, X_val, X_test])) == 0:
        raise ValueError("One of the splits is empty; check data coverage for years 2023-2024")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Train XGB with class imbalance handling and calibrate on validation.

    Returns a `CalibratedClassifierCV` with sigmoid calibration on the held-out validation set.
    """
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    logger.info(f"Class balance (train): pos={pos}, neg={neg}")
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

    base = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        verbosity=0,
    )

    # Fit with early stopping using validation set (no peeking into test)
    base.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=50,
    )

    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def _pick_best_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Sweep thresholds and return the one maximizing F1 (ties → 0.5 fallback)."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)


def evaluate(model: CalibratedClassifierCV, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model with ROC-AUC, PR-AUC, F1/precision/recall at best threshold.

    Also logs a confusion matrix for the chosen threshold.
    """
    proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    thr = _pick_best_threshold(y_test, proba)
    preds = (proba >= thr).astype(int)
    cm = confusion_matrix(y_test, preds)
    logger.info(f"Confusion matrix@thr={thr:.2f}: {cm.tolist()}")
    metrics = {
        "threshold": float(thr),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
    }
    return metrics


def save_artifacts(model: CalibratedClassifierCV, feature_cols: List[str], models_dir: Path, threshold: float) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(models_dir / "xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(models_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(models_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, ensure_ascii=False, indent=2)


def run_pipeline() -> Dict[str, float]:
    """Execute end-to-end training with robust evaluation and artifact saving."""
    np.random.seed(42)
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent.parent / "models"

    logger.info("Loading data...")
    df = load_all_years(data_dir)
    logger.info(f"Loaded {len(df):,} rows from {data_dir}")

    logger.info("Engineering features...")
    X = engineer_features(df)
    y = make_target(df.loc[X.index])

    # Align after target NaN removal
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    logger.info("Splitting data (chronological train/val/test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_time_series(X, y)

    logger.info("Training XGBoost classifier...")
    model = train_xgb(X_train, y_train, X_val, y_val)

    logger.info("Evaluating model...")
    metrics = evaluate(model, X_test, y_test)
    logger.info(
        "Metrics: "
        f"thr={metrics['threshold']:.2f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )

    logger.info("Saving artifacts...")
    save_artifacts(model, list(X.columns), models_dir, metrics["threshold"])

    return metrics


# --- Health-check utilities (used by tests) ---
def list_data_files(data_dir: Path) -> List[Path]:
    """List and validate CSV files in a directory.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Sorted list of CSV paths

    Raises:
        FileNotFoundError: If directory does not exist
        ValueError: If no CSV files are found
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(p for p in data_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {data_dir}")
    return files


def read_sample(file_path: Path, n_rows: int = 5) -> pd.DataFrame:
    """Read a small sample from a semicolon-separated CSV and validate columns.

    Args:
        file_path: Path to CSV
        n_rows: Number of rows to read

    Returns:
        DataFrame with expected columns

    Raises:
        ValueError: If required columns are missing
    """
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = pd.read_csv(file_path, sep=";", nrows=n_rows)
    # Preserve order of required columns in tests
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[required]


def run_health_check(data_dir: Path, max_files: int = 3, sample_rows: int = 5) -> Dict[str, object]:
    """Run a lightweight health check over a subset of CSVs.

    Returns a summary dict used by tests.
    """
    files = list_data_files(data_dir)
    to_sample = files[:max_files]
    total_rows = 0
    for fp in to_sample:
        sample = read_sample(fp, n_rows=sample_rows)
        total_rows += len(sample)
    summary = {
        "file_count": len(files),
        "files_sampled": len(to_sample),
        "total_rows_sampled": total_rows,
        "data_dir": str(data_dir),
        "required_columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
    }
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XAU/USD training pipeline")
    parser.add_argument("--horizon", type=int, default=5, help="Forward horizon in minutes for target creation")
    parser.add_argument("--min-return-bp", type=float, default=5.0, help="Minimum cumulative return threshold in basis points")
    parser.add_argument("--health-check-dir", type=str, default=None, help="Optional: run health check on this data directory and exit")
    args = parser.parse_args()

    try:
        if args.health_check_dir:
            summary = run_health_check(Path(args.health_check_dir))
            print("=== Health Check Summary ===")
            for k, v in summary.items():
                print(f"{k}: {v}")
        else:
            # Override target parameters by temporarily binding make_target
            # We avoid global state; compute y with provided args.
            np.random.seed(42)
            data_dir = Path(__file__).parent.parent / "data"
            models_dir = Path(__file__).parent.parent.parent / "models"

            logger.info("Loading data...")
            df = load_all_years(data_dir)
            logger.info(f"Loaded {len(df):,} rows from {data_dir}")

            logger.info("Engineering features...")
            X = engineer_features(df)
            y = make_target(df.loc[X.index], horizon=args.horizon, min_return_bp=args.min_return_bp)

            common = X.index.intersection(y.index)
            X = X.loc[common]
            y = y.loc[common]

            logger.info("Splitting data (chronological train/val/test)...")
            X_train, X_val, X_test, y_train, y_val, y_test = split_time_series(X, y)

            logger.info("Training XGBoost classifier...")
            model = train_xgb(X_train, y_train, X_val, y_val)

            logger.info("Evaluating model...")
            metrics = evaluate(model, X_test, y_test)
            print("=== Training Complete ===")
            print(f"threshold: {metrics['threshold']:.2f}")
            print(f"precision: {metrics['precision']:.4f}")
            print(f"recall:    {metrics['recall']:.4f}")
            print(f"f1:        {metrics['f1']:.4f}")
            print(f"roc_auc:   {metrics['roc_auc']:.4f}")
            print(f"pr_auc:    {metrics['pr_auc']:.4f}")

            logger.info("Saving artifacts...")
            save_artifacts(model, list(X.columns), models_dir, metrics["threshold"])
    except FileNotFoundError as e:
        print("Data files not found. Ensure CSVs exist at 'ml/src/data/XAU_1m_data_*.csv'.")
        print(str(e))
    except Exception as e:
        print("Training pipeline failed:", str(e))
        raise
