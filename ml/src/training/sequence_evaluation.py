"""Model evaluation and threshold selection.

This module provides functions for evaluating trained classifiers and selecting
optimal classification thresholds with precision (win-rate) constraints or
Expected Value (EV) optimization.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _apply_daily_cap(
    proba: np.ndarray,
    preds: np.ndarray,
    timestamps: pd.DatetimeIndex | None,
    max_trades_per_day: int | None,
) -> np.ndarray:
    """Apply per-day cap on number of positive predictions.

    Keeps the highest-probability signals per day up to max_trades_per_day.
    If timestamps or cap are None, returns preds unchanged.
    
    Args:
        proba: Predicted probabilities for positive class
        preds: Binary predictions (0/1)
        timestamps: DatetimeIndex of prediction timestamps
        max_trades_per_day: Maximum trades allowed per calendar day, or None to disable
        
    Returns:
        Modified predictions array with daily cap applied
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


def _pick_best_threshold_ev(
    y_true: np.ndarray,
    proba: np.ndarray,
    win_coefficient: float = 1.0,
    loss_coefficient: float = -1.0,
    min_trades: int | None = None,
    timestamps: pd.DatetimeIndex | None = None,
    max_trades_per_day: int | None = None,
) -> float:
    """Select threshold maximizing Expected Value (EV) - profit/loss optimization.

    Strategy:
    - EV = Recall × win_coefficient - (1 - Recall) × abs(loss_coefficient)
    - Maximizes expected profit per trade regardless of precision threshold
    - Useful when you want to maximize profit rather than win rate floor

    Args:
        y_true: True binary labels
        proba: Predicted probabilities for the positive class
        win_coefficient: Profit multiplier for True Positives (default 1.0)
        loss_coefficient: Loss multiplier for False Positives (default -1.0)
        min_trades: Minimum number of predicted positives; if None, uses
            max(10, ceil(0.001 * len(y_true)))
        timestamps: DatetimeIndex of predictions, for daily cap filtering
        max_trades_per_day: Maximum trades per day, or None to disable

    Returns:
        Threshold value (float) that maximizes Expected Value
    """
    thresholds = np.linspace(0.10, 0.95, 171)
    if min_trades is None:
        min_trades = max(10, int(np.ceil(0.001 * len(y_true))))

    best_thr = 0.5
    best_ev = float('-inf')
    best_recall = -1.0

    for t in thresholds:
        preds = (proba >= t).astype(int)
        preds = _apply_daily_cap(proba, preds, timestamps, max_trades_per_day)
        n_trades = int(preds.sum())
        if n_trades < min_trades:
            continue

        # Calculate EV as: Recall × win_coeff - (1 - Recall) × loss_coeff
        recall = recall_score(y_true, preds, zero_division=0)
        ev = recall * win_coefficient - (1.0 - recall) * abs(loss_coefficient)

        # Prefer higher EV; tie-break by higher recall then lower threshold
        if (ev > best_ev) or (np.isclose(ev, best_ev) and recall > best_recall) or (
            np.isclose(ev, best_ev) and np.isclose(recall, best_recall) and t < best_thr
        ):
            best_ev = ev
            best_recall = recall
            best_thr = float(t)

    logger.info(
        f"EV-optimized threshold: {best_thr:.2f} with EV={best_ev:.4f} "
        f"(recall={best_recall:.4f}, win_coeff={win_coefficient}, loss_coeff={loss_coefficient})"
    )

    return best_thr


def _pick_best_threshold_hybrid(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_precision: float = 0.55,
    min_recall: float = 0.15,
    win_coefficient: float = 1.0,
    loss_coefficient: float = -1.0,
    min_trades: int | None = None,
    timestamps: pd.DatetimeIndex | None = None,
    max_trades_per_day: int | None = None,
) -> float:
    """Select threshold maximizing EV under precision AND recall constraints (HYBRID).

    Strategy:
    - Enforce BOTH precision floor AND recall floor for quality + quantity balance
    - Among feasible thresholds (meeting both constraints), maximize Expected Value
    - Best of both worlds: precision protection + better recall

    Args:
        y_true: True binary labels
        proba: Predicted probabilities for the positive class
        min_precision: Minimum acceptable precision (default: 0.55)
        min_recall: Minimum acceptable recall (default: 0.15)
        win_coefficient: Profit multiplier for True Positives (default 1.0)
        loss_coefficient: Loss multiplier for False Positives (default -1.0)
        min_trades: Minimum number of predicted positives; if None, uses
            max(10, ceil(0.001 * len(y_true)))
        timestamps: DatetimeIndex of predictions, for daily cap filtering
        max_trades_per_day: Maximum trades per day, or None to disable

    Returns:
        Threshold value (float) that maximizes EV under constraints
    """
    thresholds = np.linspace(0.10, 0.95, 171)
    if min_trades is None:
        min_trades = max(10, int(np.ceil(0.001 * len(y_true))))

    best_thr = 0.5
    best_ev = float('-inf')
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

        # Both constraints must be satisfied
        if prec < min_precision or rec < min_recall:
            continue

        # Among feasible thresholds, maximize EV
        ev = rec * win_coefficient - (1.0 - rec) * abs(loss_coefficient)

        if (ev > best_ev) or (np.isclose(ev, best_ev) and rec > best_rec) or (
            np.isclose(ev, best_ev) and np.isclose(rec, best_rec) and t < best_thr
        ):
            best_ev = ev
            best_prec = prec
            best_rec = rec
            best_thr = float(t)

    if best_ev == float('-inf'):
        logger.warning(
            f"No threshold met BOTH precision >= {min_precision:.2f} AND recall >= {min_recall:.2f} "
            f"with >= {min_trades} trades. Falling back to F1-optimized with precision floor."
        )
        best_thr = _pick_best_threshold(
            y_true,
            proba,
            min_precision=min_precision,
            min_trades=min_trades,
            timestamps=timestamps,
            max_trades_per_day=max_trades_per_day,
        )
        return best_thr

    logger.info(
        f"Hybrid-optimized threshold: {best_thr:.2f} with EV={best_ev:.4f} "
        f"(precision={best_prec:.4f}, recall={best_rec:.4f}, "
        f"win_coeff={win_coefficient}, loss_coeff={loss_coefficient})"
    )

    return best_thr


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
        min_precision: Minimum acceptable precision (win rate), default 0.85
        min_trades: Minimum number of predicted positives; if None, uses
            max(25, ceil(0.002 * len(y_true)))
        timestamps: DatetimeIndex of predictions, for daily cap filtering
        max_trades_per_day: Maximum trades per day, or None to disable

    Returns:
        Threshold value (float) that maximizes F1 under constraints
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
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    min_precision: float = 0.85,
    min_recall: float = 0.15,
    min_trades: int | None = None,
    test_timestamps: pd.DatetimeIndex | None = None,
    max_trades_per_day: int | None = None,
    use_ev_optimization: bool = False,
    use_hybrid_optimization: bool = True,
    ev_win_coefficient: float = 1.0,
    ev_loss_coefficient: float = -1.0,
) -> Dict[str, float]:
    """Evaluate model with comprehensive metrics including win rate.

    Win rate is defined as precision at the chosen threshold:
    - Precision = TP / (TP + FP) = proportion of predicted positives that are correct
    
    Args:
        model: Trained calibrated classifier
        X_test: Test features (n_samples, n_features)
        y_test: Test labels (binary 0/1)
        min_precision: Minimum acceptable precision (win rate), default 0.85
        min_trades: Minimum number of trades required, or None for auto calculation
        test_timestamps: DatetimeIndex of test set timestamps, for daily cap
        max_trades_per_day: Maximum trades per day, or None to disable
        use_ev_optimization: If True, optimize for Expected Value instead of F1
        ev_win_coefficient: Profit multiplier for correct predictions (default 1.0)
        ev_loss_coefficient: Loss multiplier for incorrect predictions (default -1.0)

    Returns:
        Dictionary with metrics:
        - threshold: Selected classification threshold
        - win_rate: Precision at selected threshold (expected win rate)
        - precision: Same as win_rate
        - recall: Recall at selected threshold
        - f1: F1 score at selected threshold
        - roc_auc: ROC-AUC (threshold-independent)
        - pr_auc: PR-AUC (threshold-independent)

    Raises:
        ValueError: If X_test is empty or mismatched with y_test
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
    
    # Select threshold strategy based on configuration
    if use_hybrid_optimization:
        logger.info("Using HYBRID optimization: EV with precision AND recall floors...")
        if min_trades is None:
            min_trades = max(10, int(np.ceil(0.001 * len(y_test))))
        thr = _pick_best_threshold_hybrid(
            y_test,
            proba,
            min_precision=min_precision,
            min_recall=min_recall,
            win_coefficient=ev_win_coefficient,
            loss_coefficient=ev_loss_coefficient,
            min_trades=min_trades,
            timestamps=test_timestamps,
            max_trades_per_day=max_trades_per_day,
        )
    elif use_ev_optimization:
        logger.info("Using Expected Value (EV) optimization for threshold selection...")
        if min_trades is None:
            min_trades = max(10, int(np.ceil(0.001 * len(y_test))))
        thr = _pick_best_threshold_ev(
            y_test,
            proba,
            win_coefficient=ev_win_coefficient,
            loss_coefficient=ev_loss_coefficient,
            min_trades=min_trades,
            timestamps=test_timestamps,
            max_trades_per_day=max_trades_per_day,
        )
    else:
        logger.info("Using F1-optimized threshold with precision floor...")
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
