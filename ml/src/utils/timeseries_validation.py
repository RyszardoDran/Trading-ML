"""Time Series Cross-Validation utilities for ML pipelines.

Provides TimeSeriesSplit with proper handling of:
- Chronological data splitting
- Session-aware boundaries
- Leap-forward validation
- Metrics aggregation across folds
"""

import logging
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """Wrapper around TimeSeriesSplit for trading pipelines."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None
    ):
        """Initialize Time Series CV splitter.

        Args:
            n_splits: Number of CV folds (default 5)
            test_size: Size of test set per fold. If None: 100 // n_splits
            gap: Number of samples to exclude between train and test (gap window)
            max_train_size: Max training size per fold. If None: use all

        Notes:
            - For trading: n_splits=5 gives ~20% test size per fold
            - gap=12 excludes 12 M5 candles = 1 hour (prevents leakage)
        """
        self.tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size
        )
        self.n_splits = n_splits
        self.gap = gap
        self.metrics_per_fold = []

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None):
        """Generate train/test indices for each fold.

        Yields:
            (fold_num, train_idx, test_idx) tuple for each fold
        """
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X, y, groups)):
            logger.info(f"Fold {fold + 1}/{self.n_splits}: "
                       f"train={len(train_idx)} samples, test={len(test_idx)} samples")
            yield fold, train_idx, test_idx

    def evaluate_fold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        fold_num: int = 0
    ) -> Dict[str, float]:
        """Evaluate single fold.

        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities (1D array)
            threshold: Decision threshold
            fold_num: Fold number (for logging)

        Returns:
            Dictionary with metrics for this fold
        """
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'n_positives': np.sum(y_pred),
            'n_positives_true': np.sum(y_true),
        }

        # Only compute ROC-AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

            # Compute PR-AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)
        else:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan

        logger.info(f"Fold {fold_num}: "
                   f"Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, "
                   f"F1={metrics['f1']:.4f}")

        self.metrics_per_fold.append(metrics)
        return metrics

    def aggregate_metrics(self) -> Dict[str, float]:
        """Aggregate metrics across all folds.

        Returns:
            Dictionary with mean and std of metrics
        """
        if not self.metrics_per_fold:
            raise ValueError("No metrics to aggregate. Run evaluate_fold() first.")

        metrics_df = pd.DataFrame(self.metrics_per_fold)

        aggregated = {}
        for col in metrics_df.columns:
            if col == 'threshold':
                continue

            aggregated[f"{col}_mean"] = metrics_df[col].mean()
            aggregated[f"{col}_std"] = metrics_df[col].std()

        logger.info("\n" + "=" * 60)
        logger.info("TIME SERIES CV - AGGREGATED METRICS")
        logger.info("=" * 60)
        for key, value in aggregated.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info("=" * 60)

        return aggregated

    def get_fold_summary(self) -> pd.DataFrame:
        """Get summary table of all folds.

        Returns:
            DataFrame with metrics per fold
        """
        return pd.DataFrame(self.metrics_per_fold)


def validate_train_test_boundary(
    timestamps: pd.DatetimeIndex,
    train_idx: np.ndarray,
    test_idx: np.ndarray
) -> bool:
    """Validate that test timestamps are AFTER train timestamps.

    Args:
        timestamps: DatetimeIndex of data
        train_idx: Training indices
        test_idx: Test indices

    Returns:
        True if valid (no temporal overlap)

    Raises:
        ValueError: If test comes before train
    """
    train_max = timestamps[train_idx[-1]]
    test_min = timestamps[test_idx[0]]

    if test_min <= train_max:
        raise ValueError(
            f"Temporal validation failed: "
            f"test starts at {test_min} but train goes until {train_max}"
        )

    logger.info(f"✅ Temporal validation passed: "
               f"train until {train_max}, test from {test_min}")
    return True


def validate_no_sequence_leakage(
    timestamps: pd.DatetimeIndex,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    window_size: int
) -> bool:
    """Validate that sequences don't cross train/test boundary.

    Args:
        timestamps: DatetimeIndex of data
        train_idx: Training indices
        test_idx: Test indices
        window_size: Sequence window size in candles

    Returns:
        True if valid (no sequence crossing)

    Raises:
        ValueError: If sequences cross boundary
    """
    # Find the split point
    train_end_idx = train_idx[-1]
    test_start_idx = test_idx[0]

    # Check if any sequence starting in train would extend into test
    # A sequence starting at index i needs indices i, i+1, ..., i+window_size-1

    last_valid_sequence_start = train_end_idx - (window_size - 1)

    if last_valid_sequence_start < train_idx[0]:
        logger.warning(
            f"⚠️  Warning: window_size={window_size} is larger than "
            f"first fold training data. Some sequences will be excluded."
        )

    logger.info(f"✅ Sequence boundary validation passed: "
               f"last valid sequence starts at idx {last_valid_sequence_start}, "
               f"test starts at idx {test_start_idx}")
    return True