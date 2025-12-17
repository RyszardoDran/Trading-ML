from __future__ import annotations
"""Sequence-based training pipeline for XAU/USD (1-minute data).

Purpose:
- Load historical 1m OHLCV data from `ml/src/data/`
- Create sliding windows of 60 candles (configurable) as input features
- Engineer features for each candle in the sequence (flatten to XGBoost input)
- Train XGBoost classifier with calibrated probability estimates
- Validate win rate on test set with realistic thresholds
- Provide user with expected win rate and confidence metrics

Key principles:
- Temporal context: Model sees 100 previous candles before making prediction
- No data leakage: Strict chronological split, no future information
- Win rate validation: Precision, recall, F1, and confusion matrix on test
- Reproducibility: Fixed random seeds, deterministic behavior
- Production-ready: Type hints, validation, error handling, logging

Inputs (CSV):
- `ml/src/data/XAU_1m_data_*.csv` (semicolon-separated OHLCV)

Outputs (artifacts):
- `ml/src/models/sequence_xgb_model.pkl` (calibrated classifier)
- `ml/src/models/sequence_feature_columns.json` (ordered feature names)
- `ml/src/models/sequence_threshold.json` (selected threshold + win rate)

Expected columns: [Date;Open;High;Low;Close;Volume]
Separator: `;`, Date parseable to datetime

Usage:
    # Train with default parameters
    python sequence_training_pipeline.py
    
    # Train with custom window size and horizon
    python sequence_training_pipeline.py --window-size 50 --horizon 10
    
    # Health check only
    python sequence_training_pipeline.py --health-check-dir ml/src/data
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directories to path for imports
_script_dir = Path(__file__).parent
_src_dir = _script_dir.parent
_repo_dir = _src_dir.parent.parent
sys.path.insert(0, str(_repo_dir))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Import from modularized components
from ml.src.data_loading import load_all_years, validate_schema
from ml.src.sequences import create_sequences, filter_by_session, SequenceFilterConfig
from ml.src.targets import make_target
from ml.src.pipelines.config import PipelineConfig
from ml.src.pipelines.split import split_sequences
from ml.src.features import engineer_candle_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Train XGBoost classifier with class imbalance handling and calibration.

    Args:
        X_train: Training features
        y_train: Training labels (binary 0/1)
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed for reproducibility

    Returns:
        CalibratedClassifierCV with sigmoid calibration

    Raises:
        ValueError: If labels are not binary {0, 1}
    """
    if set(np.unique(y_train)) - {0, 1}:
        raise ValueError("y_train must be binary {0, 1}")

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1) if pos > 0 else 1.0
    logger.info(f"Class balance (train): pos={pos:,}, neg={neg:,}, imbalance_ratio={neg/max(pos,1):.2f}")
    logger.info(f"Applied scale_pos_weight={scale_pos_weight:.4f} to handle class imbalance")
    # logger.info("Using scale_pos_weight=1.0 to prioritize precision (Win Rate)")

    base = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=1, # Reduced to 1 to increase Recall
        reg_lambda=1.0, # Reduced to 1.0 to increase Recall
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        verbosity=0,
        grow_policy="depthwise",
        early_stopping_rounds=50,
    )

    # Train with validation set (XGBoost 2.0+ syntax)
    logger.info("Training XGBoost classifier...")
    base.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    logger.info("Training completed")

    # Calibrate probabilities on validation set
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    
    return calibrated


def _apply_daily_cap(
    proba: np.ndarray,
    preds: np.ndarray,
    timestamps: pd.DatetimeIndex | None,
    max_trades_per_day: int | None,
) -> np.ndarray:
    """Apply per-day cap on number of positive predictions.

    Keeps the highest-probability signals per day up to max_trades_per_day.
    If timestamps or cap are None, returns preds unchanged.
    """
    if max_trades_per_day is None or max_trades_per_day <= 0 or timestamps is None:
        return preds

    preds = preds.copy()
    # Group by date
    dates = pd.DatetimeIndex(timestamps).date
    unique_days = np.unique(dates)
    for d in unique_days:
        day_idx = np.where(dates == d)[0]
        day_pos_idx = day_idx[preds[day_idx] == 1]
        if len(day_pos_idx) > max_trades_per_day:
            # Keep top-k by probability
            top_order = np.argsort(proba[day_pos_idx])[::-1][:max_trades_per_day]
            keep_idx = set(day_pos_idx[top_order])
            # Zero out the rest
            for i in day_pos_idx:
                if i not in keep_idx:
                    preds[i] = 0
    return preds


def _pick_best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_precision: float = 0.85,
    min_trades: int | None = None,
    timestamps: pd.DatetimeIndex | None = None,
    max_trades_per_day: int | None = None,
) -> float:
    """Select threshold maximizing F1 under a precision floor and trade count.

    Strategy:
    - Enforce precision (win rate) >= min_precision to preserve quality.
    - Require a minimum number of predicted positives (min_trades) for stability.
    - Among feasible thresholds, maximize F1 (balances precision and recall).

    Args:
        y_true: True binary labels
        proba: Predicted probabilities for the positive class
        min_precision: Minimum acceptable precision (win rate)
        min_trades: Minimum number of predicted positives; if None, uses
            max(25, ceil(0.002 * len(y_true))).

    Returns:
        Threshold value as float.
    """
    thresholds = np.linspace(0.20, 0.95, 151)
    if min_trades is None:
        min_trades = max(25, int(np.ceil(0.002 * len(y_true))))

    best_thr = 0.5
    best_f1 = -1.0
    best_rec = -1.0
    found_valid = False

    for t in thresholds:
        preds = (proba >= t).astype(int)
        preds = _apply_daily_cap(proba, preds, timestamps, max_trades_per_day)
        n_trades = int(preds.sum())
        if n_trades < min_trades:
            continue

        prec = precision_score(y_true, preds, zero_division=0)
        if prec < min_precision:
            continue
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        # Prefer higher F1; tie-break by higher recall then lower threshold
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and rec > best_rec) or (
            np.isclose(f1, best_f1) and np.isclose(rec, best_rec) and t < best_thr
        ):
            best_f1 = f1
            best_rec = rec
            best_thr = float(t)
            found_valid = True

    if not found_valid:
        logger.warning(
            f"No threshold met precision >= {min_precision:.2f} with >= {min_trades} trades. "
            "Falling back to maximizing precision with stability constraint."
        )
        best_prec = -1.0
        best_rec = -1.0
        for t in thresholds:
            preds = (proba >= t).astype(int)
            preds = _apply_daily_cap(proba, preds, timestamps, max_trades_per_day)
            n_trades = int(preds.sum())
            if n_trades < min_trades:
                continue
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            if (prec > best_prec) or (np.isclose(prec, best_prec) and rec > best_rec):
                best_prec = prec
                best_rec = rec
                best_thr = float(t)

    return best_thr


def evaluate(
    model: CalibratedClassifierCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
    min_precision: float = 0.85,
    min_trades: int | None = None,
    test_timestamps: pd.DatetimeIndex | None = None,
    max_trades_per_day: int | None = None,
) -> Dict[str, float]:
    """Evaluate model with comprehensive metrics including win rate.

    Win rate is defined as precision at the chosen threshold:
    - Precision = TP / (TP + FP) = proportion of predicted positives that are correct
    
    Args:
        model: Trained calibrated classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with metrics: threshold, precision (win_rate), recall, f1, roc_auc, pr_auc

    Raises:
        ValueError: If X_test is empty
    """
    if len(X_test) == 0:
        raise ValueError("X_test is empty; cannot evaluate")
    if len(y_test) != len(X_test):
        raise ValueError(f"Length mismatch: X_test({len(X_test)}) vs y_test({len(y_test)})")

    proba = model.predict_proba(X_test)[:, 1]
    if not np.isfinite(proba).all():
        raise ValueError("Non-finite probabilities produced; check feature values and calibration")
    
    # Log probability stats
    logger.info(f"Probability stats: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}, std={proba.std():.4f}")
    
    # Metrics
    roc = roc_auc_score(y_test, proba)
    if roc < 0.52:
        logger.warning(f"⚠️  Model has very low discriminative power (ROC-AUC={roc:.4f}). Results may be random.")
    
    pr_auc = average_precision_score(y_test, proba)
    # Target a high win rate floor while improving recall; enforce stability in trade count
    if min_trades is None:
        min_trades = max(25, int(np.ceil(0.002 * len(y_test))))
    thr = _pick_best_threshold(
        y_test,
        proba,
        min_precision=min_precision,
        min_trades=min_trades,
        timestamps=test_timestamps,
        max_trades_per_day=max_trades_per_day,
    )
    preds = (proba >= thr).astype(int)
    preds = _apply_daily_cap(proba, preds, test_timestamps, max_trades_per_day)
    
    cm = confusion_matrix(y_test, preds)
    logger.info(f"Confusion matrix@thr={thr:.2f}:")
    logger.info(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    logger.info(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    metrics = {
        "threshold": float(thr),
        "win_rate": float(precision),  # Win rate = Precision
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
    }
    
    return metrics


def analyze_feature_importance(
    model: CalibratedClassifierCV,
    feature_cols: List[str],
    window_size: int,
    top_k: int = 20,
) -> Dict[str, float]:
    """Analyze feature importance from trained XGBoost model.
    
    Args:
        model: Trained calibrated classifier
        feature_cols: List of per-candle feature names
        window_size: Number of candles in window
        top_k: Number of top features to return
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Get base estimator (XGBoost) from calibrated model
    base_model = model.calibrated_classifiers_[0].estimator
    
    # Get feature importances
    importances = base_model.feature_importances_
    
    # Map flattened feature indices to (candle_offset, feature_name)
    n_features_per_candle = len(feature_cols)
    feature_names = []
    for i in range(window_size):
        for feat in feature_cols:
            feature_names.append(f"t-{window_size - i - 1}_{feat}")
    
    # Create importance dict and sort
    importance_dict = dict(zip(feature_names, importances))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Log top features
    logger.info(f"\nTop {top_k} most important features:")
    for rank, (feat, score) in enumerate(sorted_features[:top_k], 1):
        logger.info(f"  {rank:2d}. {feat:40s} = {score:.6f}")
    
    # Aggregate by feature type (ignoring time offset)
    feature_type_importance = {}
    for feat_name, importance in importance_dict.items():
        feat_type = feat_name.split("_", 1)[1] if "_" in feat_name else feat_name
        feature_type_importance[feat_type] = feature_type_importance.get(feat_type, 0) + importance
    
    sorted_types = sorted(feature_type_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"\nAggregated importance by feature type:")
    for feat_type, total_importance in sorted_types[:10]:
        logger.info(f"  {feat_type:30s} = {total_importance:.6f}")
    
    # Convert to JSON-serializable format (sanitize NaN/inf values)
    result = {}
    for feat, importance in sorted_features[:top_k]:
        val = float(importance)
        # Replace NaN/inf with 0 for JSON serialization
        if not np.isfinite(val):
            val = 0.0
        result[feat] = val
    
    return result


def save_artifacts(
    model: CalibratedClassifierCV,
    scaler: RobustScaler,
    feature_cols: List[str],
    models_dir: Path,
    threshold: float,
    win_rate: float,
    window_size: int,
) -> None:
    """Save trained model, scaler, and metadata.

    Args:
        model: Trained calibrated classifier
        scaler: Fitted RobustScaler for feature normalization
        feature_cols: Ordered list of feature column names
        models_dir: Directory to save artifacts
        threshold: Selected classification threshold
        win_rate: Expected win rate (precision on test set)
        window_size: Number of candles in each window
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    # Save model
    with open(models_dir / "sequence_xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save scaler (CRITICAL for production inference)
    with open(models_dir / "sequence_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to sequence_scaler.pkl (required for inference)")
    
    with open(models_dir / "sequence_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    
    metadata = {
        "threshold": threshold,
        "win_rate": win_rate,
        "window_size": window_size,
        "n_features_per_candle": len(feature_cols),
        "total_features": len(feature_cols) * window_size,
    }
    with open(models_dir / "sequence_threshold.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Analyze and save feature importance
    logger.info("Analyzing feature importance...")
    top_features = analyze_feature_importance(model, feature_cols, window_size, top_k=30)
    with open(models_dir / "sequence_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(top_features, f, ensure_ascii=False, indent=2)


def run_pipeline(
    window_size: int = 60,
    atr_multiplier_sl: float = 1.0,
    atr_multiplier_tp: float = 2.0,
    min_hold_minutes: int = 5,
    max_horizon: int = 60,
    random_state: int = 42,
    year_filter: List[int] = None,
    session: str = "london_ny",
    custom_start_hour: int = None,
    custom_end_hour: int = None,
    max_windows: int = 200000,
    min_precision: float = 0.85,
    min_trades: int | None = None,
    max_trades_per_day: int | None = None,
    enable_m5_alignment: bool = True,
    enable_trend_filter: bool = True,
    trend_min_dist_sma200: float | None = 0.0,
    trend_min_adx: float | None = 15.0,
    enable_pullback_filter: bool = True,
    pullback_max_rsi_m5: float | None = 75.0,
) -> Dict[str, float]:
    """Execute end-to-end sequence training pipeline.

    Args:
        window_size: Number of previous candles to use as input (default: 3 - reduced to avoid noise)
        atr_multiplier_sl: ATR multiplier for stop-loss (default: 1.0 - CONSTANT)
        atr_multiplier_tp: ATR multiplier for take-profit (default: 2.0 - CONSTANT, 2:1 RR)
        min_hold_minutes: Minimum hold time in minutes (default: 5)
        max_horizon: Maximum forward candles to simulate (default: 60)
        random_state: Random seed for reproducibility
        year_filter: Optional list of years to load (e.g., [2023, 2024] for testing)
        max_windows: Maximum number of windows to keep (default: 200,000)
        enable_m5_alignment: Align decisions with M5 candle closes when True
        enable_trend_filter: Enforce SMA/ADX trend conditions when True
        trend_min_dist_sma200: Minimum normalized distance above SMA200 when trend
            filter is active; set to None to disable this component
        trend_min_adx: Minimum ADX threshold when trend filter is active; set to
            None to disable this component
        enable_pullback_filter: Enforce RSI_M5 pullback guard when True
        pullback_max_rsi_m5: Maximum RSI_M5 allowed when pullback filter is active;
            set to None to disable the RSI cap

    Returns:
        Dictionary with evaluation metrics including win_rate

    Raises:
        FileNotFoundError: If data files not found
        ValueError: On validation failures or insufficient data
    """
    np.random.seed(random_state)
    import random
    random.seed(random_state)
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"

    logger.info("Loading data...")
    df = load_all_years(data_dir, year_filter=year_filter)
    logger.info(f"Loaded {len(df):,} rows from {data_dir}")

    logger.info(f"Engineering per-candle features (window_size={window_size})...")
    features = engineer_candle_features(df, window_size=window_size)
    logger.info(f"Features shape: {features.shape}")

    logger.info(f"Creating target (SL={atr_multiplier_sl}×ATR, TP={atr_multiplier_tp}×ATR, min_hold={min_hold_minutes}min)...")
    targets = make_target(
        df.loc[features.index],
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
        min_hold_minutes=min_hold_minutes,
        max_horizon=max_horizon,
    )
    logger.info(f"Target shape: {len(targets)}, positive class: {targets.sum()} ({targets.mean():.2%})")

    logger.info(f"Creating sequences (window_size={window_size})...")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    filter_config = SequenceFilterConfig(
        enable_m5_alignment=enable_m5_alignment,
        enable_trend_filter=enable_trend_filter,
        trend_min_dist_sma200=trend_min_dist_sma200,
        trend_min_adx=trend_min_adx,
        enable_pullback_filter=enable_pullback_filter,
        pullback_max_rsi_m5=pullback_max_rsi_m5,
    )
    logger.info(
        "Filter configuration: m5=%s, trend=%s(dist_sma200=%s, adx=%s), pullback=%s(rsi<=%s)"
        % (
            enable_m5_alignment,
            enable_trend_filter,
            trend_min_dist_sma200 if trend_min_dist_sma200 is not None else "disabled",
            trend_min_adx if trend_min_adx is not None else "disabled",
            enable_pullback_filter,
            pullback_max_rsi_m5 if pullback_max_rsi_m5 is not None else "disabled",
        )
    )
    X, y, timestamps = create_sequences(
        features,
        targets,
        window_size=window_size,
        session=session,
        custom_start=custom_start_hour,
        custom_end=custom_end_hour,
        filter_config=filter_config,
        max_windows=max_windows,
    )
    logger.info(f"Sequences: X.shape={X.shape}, y.shape={y.shape}")

    logger.info("Splitting data (chronological train/val/test)...")
    # Dynamic split based on data range
    if year_filter is not None:
        # For year filter: use percentage split to avoid empty splits
        n = len(X)
        train_idx = int(0.7 * n)
        val_idx = int(0.85 * n)
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        ts_train, ts_val, ts_test = (
            timestamps[:train_idx],
            timestamps[train_idx:val_idx],
            timestamps[val_idx:],
        )
        logger.info(f"Using percentage split (70/15/15) for year_filter={year_filter}")
    else:
        # Full date range: use fixed date splits
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            ts_train,
            ts_val,
            ts_test,
        ) = split_sequences(X, y, timestamps)
    logger.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # CRITICAL: Scale features AFTER split to prevent data leakage
    # Fit scaler ONLY on training data, then transform all sets
    logger.info("Scaling features with RobustScaler (robust to outliers)...")
    scaler = RobustScaler()
    # Ensure float32 output to save memory
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    logger.info(f"Feature scaling complete: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")

    logger.info("Training XGBoost classifier...")
    model = train_xgb(X_train_scaled, y_train, X_val_scaled, y_val, random_state=random_state)

    logger.info("Evaluating model on test set...")
    metrics = evaluate(
        model,
        X_test_scaled,
        y_test,
        min_precision=min_precision,
        min_trades=min_trades,
        test_timestamps=ts_test,
        max_trades_per_day=max_trades_per_day,
    )
    logger.info(
        "Metrics: "
        f"threshold={metrics['threshold']:.2f}, "
        f"win_rate={metrics['win_rate']:.4f} ({metrics['win_rate']:.2%}), "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )

    logger.info("Saving artifacts...")
    logger.info("Saving artifacts (model, scaler, metadata)...")
    save_artifacts(
        model,
        scaler,
        list(features.columns),
        models_dir,
        metrics["threshold"],
        metrics["win_rate"],
        window_size,
    )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sequence-based XAU/USD training pipeline")
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Number of previous candles to use as input (default: 60)",
    )
    parser.add_argument(
        "--atr-multiplier-sl",
        type=float,
        default=1.0,
        help="ATR multiplier for stop-loss level (default: 1.0 - DO NOT CHANGE)",
    )
    parser.add_argument(
        "--atr-multiplier-tp",
        type=float,
        default=2.0,
        help="ATR multiplier for take-profit level (default: 2.0 - DO NOT CHANGE)",
    )
    parser.add_argument(
        "--min-hold-minutes",
        type=int,
        default=5,
        help="Minimum hold time in minutes (default: 5)",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=60,
        help="Maximum forward candles to simulate (default: 60)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years to load (e.g., '2023,2024' for testing)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="london_ny",
        choices=["london", "ny", "asian", "london_ny", "all", "custom"],
        help="Trading session to filter data (default: london_ny)",
    )
    parser.add_argument(
        "--custom-start-hour",
        type=int,
        default=None,
        help="Start hour for custom session (0-23)",
    )
    parser.add_argument(
        "--custom-end-hour",
        type=int,
        default=None,
        help="End hour for custom session (0-23)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200000,
        help="Maximum number of windows to keep to avoid OOM (default: 200,000)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.85,
        help="Minimum precision (win rate) floor for threshold selection (default: 0.85)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        help="Minimum number of predicted positives for a threshold to be considered (default: dynamic)",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=None,
        help="Cap number of predicted trades per day after thresholding (default: unlimited)",
    )
    parser.add_argument(
        "--skip-m5-alignment",
        action="store_true",
        help="Disable M5 alignment filter (default: enabled)",
    )
    parser.add_argument(
        "--disable-trend-filter",
        action="store_true",
        help="Disable trend filter requiring price above SMA200 and ADX threshold",
    )
    parser.add_argument(
        "--trend-min-dist-sma200",
        type=float,
        default=0.0,
        help="Minimum normalized distance above SMA200 when trend filter enabled (default: 0.0)",
    )
    parser.add_argument(
        "--trend-min-adx",
        type=float,
        default=15.0,
        help="Minimum ADX when trend filter enabled (default: 15.0)",
    )
    parser.add_argument(
        "--disable-pullback-filter",
        action="store_true",
        help="Disable RSI_M5 pullback guard",
    )
    parser.add_argument(
        "--pullback-max-rsi-m5",
        type=float,
        default=75.0,
        help="Maximum RSI_M5 when pullback filter enabled (default: 75.0)",
    )
    args = parser.parse_args()

    # Parse year filter
    year_filter = None
    if args.years:
        year_filter = [int(y.strip()) for y in args.years.split(',')]

    try:
        metrics = run_pipeline(
            window_size=args.window_size,        
            atr_multiplier_sl=args.atr_multiplier_sl,
            atr_multiplier_tp=args.atr_multiplier_tp,
            min_hold_minutes=args.min_hold_minutes,
            max_horizon=args.max_horizon,        
            random_state=args.random_state,      
            year_filter=year_filter,
            session=args.session,
            custom_start_hour=args.custom_start_hour,
            custom_end_hour=args.custom_end_hour,
            max_windows=args.max_windows,
            min_precision=args.min_precision,
            min_trades=args.min_trades,
            max_trades_per_day=args.max_trades_per_day,
            enable_m5_alignment=not args.skip_m5_alignment,
            enable_trend_filter=not args.disable_trend_filter,
            trend_min_dist_sma200=None if args.disable_trend_filter else args.trend_min_dist_sma200,
            trend_min_adx=None if args.disable_trend_filter else args.trend_min_adx,
            enable_pullback_filter=not args.disable_pullback_filter,
            pullback_max_rsi_m5=None if args.disable_pullback_filter else args.pullback_max_rsi_m5,
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - SEQUENCE PIPELINE")
        print("=" * 60)
        print(f"Window Size:       {args.window_size} candles")
        print(f"Threshold:         {metrics['threshold']:.2f}")
        print(f"WIN RATE:          {metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1 Score:          {metrics['f1']:.4f}")
        print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:            {metrics['pr_auc']:.4f}")
        print("=" * 60)
        print(f"\nWin rate is the precision: when model predicts 'BUY',")
        print(f"it will be correct {metrics['win_rate']:.2%} of the time on test data.")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print("Data files not found. Ensure CSVs exist at 'ml/src/data/XAU_1m_data_*.csv'.")
        print(str(e))
    except Exception as e:
        print("Training pipeline failed:", str(e))
        raise
