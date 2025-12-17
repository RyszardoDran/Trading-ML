"""XGBoost classifier training.

This module provides the main training function for the sequence-based
XAU/USD trading model with class imbalance handling.
"""

import logging
from typing import Tuple

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> XGBClassifier:
    """Train XGBoost classifier with class imbalance handling.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (binary 0/1)
        X_val: Validation features (n_samples, n_features)
        y_val: Validation labels (binary 0/1)
        random_state: Random seed for reproducibility

    Returns:
        Trained XGBClassifier with raw probability estimates

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

    base = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=1,
        reg_lambda=1.0,
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

    return base
