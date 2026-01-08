"""XGBoost classifier training.

This module provides the main training function for the sequence-based
XAU/USD trading model with class imbalance handling.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
    n_estimators: int = 600,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> XGBClassifier:
    """Train XGBoost classifier with class imbalance and cost-sensitive learning.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (binary 0/1)
        X_val: Optional validation features for early stopping (n_samples, n_features)
        y_val: Optional validation labels for early stopping (binary 0/1)
        random_state: Random seed for reproducibility
        sample_weight: Optional sample weights for cost-sensitive learning (n_samples,)
                      If provided, higher weights penalize misclassification more
        n_estimators: Number of boosting trees (default 600, use ~100-200 for cross-validation)

    Returns:
        Trained XGBClassifier with raw probability estimates

    Raises:
        ValueError: If labels are not binary {0, 1} or sample_weight has wrong shape
    """
    if set(np.unique(y_train)) - {0, 1}:
        raise ValueError("y_train must be binary {0, 1}")
    
    if sample_weight is not None and len(sample_weight) != len(y_train):
        raise ValueError(f"sample_weight must have same length as y_train, got {len(sample_weight)} vs {len(y_train)}")

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1) if pos > 0 else 1.0
    logger.info(f"Class balance (train): pos={pos:,}, neg={neg:,}, imbalance_ratio={neg/max(pos,1):.2f}")
    logger.info(f"Applied scale_pos_weight={scale_pos_weight:.4f} to handle class imbalance")
    
    if sample_weight is not None:
        logger.info(f"[POINT 1] Cost-Sensitive Learning: using sample weights (mean={sample_weight.mean():.4f})")

    resolved_params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": random_state,
        "n_jobs": 4,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "verbosity": 0,
        "grow_policy": "depthwise",
        "early_stopping_rounds": 20 if X_val is not None else None,
    }

    profile_params = dict(xgb_params or {})
    if "n_estimators" not in profile_params:
        profile_params["n_estimators"] = n_estimators

    resolved_params.update(profile_params)

    logger.info("Final XGBoost parameters (including overrides):")
    for key in sorted(resolved_params):
        logger.info("  %s = %s", key, resolved_params[key])

    base = XGBClassifier(**resolved_params)

    # Train with or without eval_set
    logger.info("Training XGBoost classifier...")
    if X_val is not None and y_val is not None:
        base.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weight,  # Cost-sensitive learning weights
            verbose=False,
        )
    else:
        base.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,  # Cost-sensitive learning weights
            verbose=False,
        )
    
    logger.info("Training completed")

    return base
