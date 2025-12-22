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


def _max_recall_possible_under_daily_cap(
    y_true: np.ndarray,
    timestamps: pd.DatetimeIndex | None,
    max_trades_per_day: int | None,
) -> float | None:
    """Compute an upper bound on recall given a per-day trade cap.

    If we can execute at most K trades per day and the evaluation window spans D
    unique calendar days, then we can take at most K*D positive predictions.
    Since recall = TP / P, the maximum possible recall is bounded by:
        min(1, (K*D)/P)

    Returns None when the cap is disabled or inputs are missing.
    """
    if (
        timestamps is None
        or max_trades_per_day is None
        or max_trades_per_day <= 0
        or y_true.size == 0
    ):
        return None

    positives = int(np.sum(y_true == 1))
    if positives <= 0:
        return 1.0

    days = len(np.unique(pd.DatetimeIndex(timestamps).date))
    if days <= 0:
        return None

    max_trades_total = int(max_trades_per_day) * int(days)
    return float(min(1.0, max_trades_total / positives))


def _candidate_thresholds(
    proba: np.ndarray,
    *,
    n: int = 401,
    min_threshold: float = 0.001,
    max_threshold: float = 0.99,
) -> np.ndarray:
    """Build a robust, data-driven threshold grid.

    Why:
        Using a fixed linspace like [0.15..0.95] can pin the result to the grid
        boundary (e.g. always returning 0.15). A quantile-based grid adapts to
        the model's probability calibration and keeps resolution where the data
        actually lives.

    Returns:
        Sorted unique thresholds in (0, 1), clipped to [min_threshold, max_threshold].
    """
    if proba.size == 0:
        return np.array([0.5], dtype=float)

    min_threshold = float(np.clip(min_threshold, 0.0, 1.0))
    max_threshold = float(np.clip(max_threshold, 0.0, 1.0))
    if max_threshold <= min_threshold:
        max_threshold = min(1.0, min_threshold + 1e-6)

    q = np.linspace(0.0, 1.0, max(11, int(n)))
    thresholds = np.quantile(proba, q)
    thresholds = np.clip(thresholds, min_threshold, max_threshold)
    thresholds = np.unique(thresholds.astype(float))

    # Ensure bounds are present (useful for diagnostics)
    thresholds = np.unique(np.concatenate(([min_threshold], thresholds, [max_threshold])))
    thresholds.sort()
    return thresholds


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
        - EV per trade depends on win rate (precision), not recall:
                EV_per_trade = precision * win_coefficient + (1 - precision) * loss_coefficient
            where loss_coefficient is typically negative.
        - We maximize total EV across the evaluation window:
                EV_total = EV_per_trade * n_trades
        - This naturally balances quality (precision) vs frequency (n_trades), while
            still honoring min_trades and the optional daily cap.

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
    thresholds = _candidate_thresholds(proba)
    if min_trades is None:
        min_trades = max(10, int(np.ceil(0.001 * len(y_true))))

    best_thr = 0.5
    best_ev_total = float('-inf')
    best_ev_per_trade = float('-inf')
    best_prec = -1.0
    best_rec = -1.0
    best_trades = 0

    for t in thresholds:
        preds = (proba >= t).astype(int)
        preds = _apply_daily_cap(proba, preds, timestamps, max_trades_per_day)
        n_trades = int(preds.sum())
        if n_trades < min_trades:
            continue

        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)

        ev_per_trade = prec * win_coefficient + (1.0 - prec) * loss_coefficient
        ev_total = ev_per_trade * float(n_trades)

        # Prefer higher total EV; tie-break by higher EV/trade then more recall then lower threshold
        if (
            (ev_total > best_ev_total)
            or (np.isclose(ev_total, best_ev_total) and ev_per_trade > best_ev_per_trade)
            or (np.isclose(ev_total, best_ev_total) and np.isclose(ev_per_trade, best_ev_per_trade) and rec > best_rec)
            or (
                np.isclose(ev_total, best_ev_total)
                and np.isclose(ev_per_trade, best_ev_per_trade)
                and np.isclose(rec, best_rec)
                and t < best_thr
            )
        ):
            best_ev_total = float(ev_total)
            best_ev_per_trade = float(ev_per_trade)
            best_prec = float(prec)
            best_rec = float(rec)
            best_trades = int(n_trades)
            best_thr = float(t)

    if np.isclose(best_thr, float(thresholds[0])) or np.isclose(best_thr, float(thresholds[-1])):
        logger.warning(
            f"EV selected boundary threshold={best_thr:.4f} (grid=[{thresholds[0]:.4f}..{thresholds[-1]:.4f}], size={len(thresholds)}). "
            "This can indicate the optimum lies outside the current grid or probabilities are compressed."
        )

    logger.info(
        f"EV-optimized threshold: {best_thr:.4f} with EV_total={best_ev_total:.4f} "
        f"(EV/trade={best_ev_per_trade:.4f}, precision={best_prec:.4f}, recall={best_rec:.4f}, trades={best_trades}, "
        f"win_coeff={win_coefficient}, loss_coeff={loss_coefficient})"
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
        - Enforce precision floor (win-rate protection) and an optional recall floor.
        - Among feasible thresholds, maximize total Expected Value (EV_total), where
            EV is based on precision (not recall):
                EV_per_trade = precision * win_coefficient + (1 - precision) * loss_coefficient
                EV_total = EV_per_trade * n_trades
        - This behaves sensibly under a daily cap: thresholds that only change
            "how many candidates enter top-K" are not incorrectly rewarded by recall-only EV.

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
    max_recall_possible = _max_recall_possible_under_daily_cap(
        y_true, timestamps, max_trades_per_day
    )
    if max_recall_possible is not None and max_recall_possible + 1e-12 < min_recall:
        logger.warning(
            "Recall floor is infeasible under daily cap: "
            f"min_recall={min_recall:.2f} but max_possible_recall≈{max_recall_possible:.4f} "
            f"(max_trades_per_day={max_trades_per_day}). "
            "Relaxing min_recall to the feasible maximum for HYBRID search."
        )
        min_recall = max_recall_possible

    thresholds = _candidate_thresholds(proba)
    if min_trades is None:
        min_trades = max(10, int(np.ceil(0.001 * len(y_true))))

    best_thr = 0.5
    best_ev_total = float('-inf')
    best_ev_per_trade = float('-inf')
    best_prec = -1.0
    best_rec = -1.0
    best_trades = 0

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

        ev_per_trade = prec * win_coefficient + (1.0 - prec) * loss_coefficient
        ev_total = ev_per_trade * float(n_trades)

        if (
            (ev_total > best_ev_total)
            or (np.isclose(ev_total, best_ev_total) and ev_per_trade > best_ev_per_trade)
            or (np.isclose(ev_total, best_ev_total) and np.isclose(ev_per_trade, best_ev_per_trade) and rec > best_rec)
            or (
                np.isclose(ev_total, best_ev_total)
                and np.isclose(ev_per_trade, best_ev_per_trade)
                and np.isclose(rec, best_rec)
                and t < best_thr
            )
        ):
            best_ev_total = float(ev_total)
            best_ev_per_trade = float(ev_per_trade)
            best_prec = float(prec)
            best_rec = float(rec)
            best_trades = int(n_trades)
            best_thr = float(t)

    if best_ev_total == float('-inf'):
        logger.warning(
            f"No threshold met BOTH precision >= {min_precision:.2f} AND recall >= {min_recall:.2f} "
            f"with >= {min_trades} trades. Using improved fallback: maximize F1 with relaxed constraints."
        )
        
        # Improved fallback: Find threshold that best balances precision and recall
        # Priority: maximize F1 score among all thresholds
        best_f1 = -1.0
        best_thr_f1 = 0.5
        best_prec_f1 = -1.0
        best_rec_f1 = -1.0
        
        for t in thresholds:
            preds = (proba >= t).astype(int)
            preds = _apply_daily_cap(proba, preds, timestamps, max_trades_per_day)
            n_trades = int(preds.sum())
            if n_trades < min_trades:
                continue
            
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            
            # Find threshold with highest F1
            if f1 > best_f1:
                best_f1 = f1
                best_thr_f1 = float(t)
                best_prec_f1 = prec
                best_rec_f1 = rec
        
        if best_f1 > -1.0:
            if np.isclose(best_thr_f1, float(thresholds[0])) or np.isclose(best_thr_f1, float(thresholds[-1])):
                logger.warning(
                    f"Fallback selected boundary threshold={best_thr_f1:.4f} (grid=[{thresholds[0]:.4f}..{thresholds[-1]:.4f}], size={len(thresholds)}). "
                    "If this repeats, consider widening candidate grid or reviewing probability calibration."
                )
            logger.info(
                f"Fallback F1-optimized threshold: {best_thr_f1:.4f} "
                f"(F1={best_f1:.4f}, precision={best_prec_f1:.4f}, recall={best_rec_f1:.4f})"
            )
            return best_thr_f1
        else:
            logger.error(
                f"CRITICAL: No valid threshold found. Using min_trades fallback (threshold=0.50)"
            )
            return 0.5

    if np.isclose(best_thr, float(thresholds[0])) or np.isclose(best_thr, float(thresholds[-1])):
        logger.warning(
            f"Hybrid selected boundary threshold={best_thr:.4f} (grid=[{thresholds[0]:.4f}..{thresholds[-1]:.4f}], size={len(thresholds)})."
        )

    logger.info(
        f"Hybrid-optimized threshold: {best_thr:.4f} with EV_total={best_ev_total:.4f} "
        f"(EV/trade={best_ev_per_trade:.4f}, precision={best_prec:.4f}, recall={best_rec:.4f}, trades={best_trades}, "
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
    thresholds = _candidate_thresholds(proba)
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

    if np.isclose(best_thr, float(thresholds[0])) or np.isclose(best_thr, float(thresholds[-1])):
        logger.warning(
            f"F1 selected boundary threshold={best_thr:.4f} (grid=[{thresholds[0]:.4f}..{thresholds[-1]:.4f}], size={len(thresholds)})."
        )

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
    
    # Log probability stats + selection constraints (helps explain "stuck" thresholds)
    logger.info(
        "Probability stats: "
        f"min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}, std={proba.std():.4f}"
    )
    logger.info(
        f"Threshold constraints: min_precision={min_precision:.2f}, min_recall={min_recall:.2f}, "
        f"min_trades={min_trades if min_trades is not None else 'auto'}, max_trades_per_day={max_trades_per_day}"
    )

    max_recall_possible = _max_recall_possible_under_daily_cap(
        y_test, test_timestamps, max_trades_per_day
    )
    if max_recall_possible is not None:
        logger.info(
            f"Daily-cap recall upper bound: max_possible_recall≈{max_recall_possible:.4f} "
            f"(max_trades_per_day={max_trades_per_day})"
        )
    
    # Metrics
    roc = roc_auc_score(y_test, proba)
    if roc < 0.52:
        logger.warning(f"[WARNING] Model has very low discriminative power (ROC-AUC={roc:.4f}). Results may be random.")
    
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

    # If a daily cap is active, many thresholds below the (per-day) top-K cutoff
    # produce identical final predictions (cap dominates). To keep the stored
    # threshold meaningful for downstream consumers, normalize it to the
    # effective cutoff implied by the finally selected trades.
    if max_trades_per_day is not None and max_trades_per_day > 0 and test_timestamps is not None:
        selected_idx = np.where(preds == 1)[0]
        if selected_idx.size > 0:
            effective_thr = float(np.min(proba[selected_idx]))
            if np.isfinite(effective_thr) and 0.0 <= effective_thr <= 1.0:
                # Recompute predictions using the effective threshold for consistency.
                thr = effective_thr
                preds = (proba >= thr).astype(int)
                preds = _apply_daily_cap(proba, preds, test_timestamps, max_trades_per_day)
    
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    logger.info(f"Confusion matrix@thr={thr:.4f}:")
    logger.info(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    logger.info(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")

    if max_trades_per_day is not None and max_trades_per_day > 0 and test_timestamps is not None:
        selected = np.where(preds == 1)[0]
        if selected.size > 0:
            logger.info(
                f"Effective min proba among selected trades (after daily cap) = {float(np.min(proba[selected])):.4f}"
            )

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
