"""Training module for sequence-based XAU/USD trading model.

This module contains functions for training and evaluating XGBoost classifiers:
- train_xgb: Train calibrated XGBoost classifier with class imbalance handling
- evaluate: Comprehensive evaluation with win-rate focused metrics
- analyze_feature_importance: Feature importance analysis
- save_artifacts: Model serialization and persistence

Example:
    from ml.src.training import train_xgb, evaluate
    
    model = train_xgb(X_train, y_train, X_val, y_val)
    metrics = evaluate(model, X_test, y_test)
"""

from .sequence_xgb_trainer import train_xgb
from .sequence_evaluation import evaluate, evaluate_with_fixed_threshold, optimize_threshold_on_val
from .sequence_feature_analysis import analyze_feature_importance
from .sequence_artifacts import save_artifacts

__all__ = [
    "train_xgb",
    "evaluate",
    "evaluate_with_fixed_threshold",
    "optimize_threshold_on_val",
    "analyze_feature_importance",
    "save_artifacts",
]
