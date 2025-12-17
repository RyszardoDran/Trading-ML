"""Feature importance analysis for trained models.

This module provides functions for analyzing and visualizing feature importance
from trained XGBoost classifiers.
"""

import logging
from typing import Dict, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


def analyze_feature_importance(
    model: CalibratedClassifierCV,
    feature_cols: List[str],
    window_size: int,
    top_k: int = 20,
) -> Dict[str, float]:
    """Analyze feature importance from trained XGBoost model.
    
    Extracts feature importances from the base XGBoost estimator in a calibrated
    classifier, maps them to per-candle feature names with time offsets, and
    returns top-k features for analysis.
    
    Args:
        model: Trained CalibratedClassifierCV wrapping XGBoost classifier
        feature_cols: List of per-candle feature names (e.g., ['open', 'high', ...])
        window_size: Number of candles in each input window
        top_k: Number of top features to return, default 20
        
    Returns:
        Dictionary mapping feature names (with time offset) to importance scores.
        Feature names follow format: "t-{offset}_{feature_name}"
        where offset=0 is the most recent candle.
        
    Example:
        >>> importance = analyze_feature_importance(model, ['open', 'high', 'low', 'close'], window_size=100)
        >>> # Result: {'t-0_close': 0.0523, 't-1_high': 0.0412, ...}
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
