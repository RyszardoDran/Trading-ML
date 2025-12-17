"""Model artifact serialization and storage.

This module provides functions for saving trained models, scalers, and metadata
to disk for later use in production inference.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler

from ml.src.training.sequence_feature_analysis import analyze_feature_importance

logger = logging.getLogger(__name__)


def save_artifacts(
    model: CalibratedClassifierCV,
    scaler: RobustScaler,
    feature_cols: List[str],
    models_dir: Path,
    threshold: float,
    win_rate: float,
    window_size: int,
) -> None:
    """Save trained model, scaler, and metadata to disk.

    Saves all artifacts needed for production inference:
    - Trained and calibrated XGBoost model (pickle)
    - Fitted RobustScaler for feature normalization (pickle)
    - Ordered feature column names (JSON)
    - Classification threshold and metadata (JSON)
    - Top feature importances (JSON)

    Args:
        model: Trained CalibratedClassifierCV classifier
        scaler: Fitted RobustScaler for feature normalization
        feature_cols: Ordered list of per-candle feature column names
        models_dir: Directory path where artifacts will be saved
        threshold: Selected classification threshold for binary decision
        win_rate: Expected win rate (precision on test set)
        window_size: Number of candles in each input window

    Raises:
        IOError: If unable to write to models_dir
        
    Example:
        >>> save_artifacts(model, scaler, features, Path('ml/src/models'),
        ...                threshold=0.65, win_rate=0.87, window_size=100)
        # Saves:
        # - ml/src/models/sequence_xgb_model.pkl
        # - ml/src/models/sequence_scaler.pkl
        # - ml/src/models/sequence_feature_columns.json
        # - ml/src/models/sequence_threshold.json
        # - ml/src/models/sequence_feature_importance.json
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(models_dir / "sequence_xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to sequence_xgb_model.pkl")
    
    # Save scaler (CRITICAL for production inference)
    with open(models_dir / "sequence_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to sequence_scaler.pkl (required for inference)")
    
    # Save feature column names (order is critical!)
    with open(models_dir / "sequence_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    logger.info("Saved feature columns to sequence_feature_columns.json")
    
    # Save threshold and metadata
    metadata = {
        "threshold": threshold,
        "win_rate": win_rate,
        "window_size": window_size,
        "n_features_per_candle": len(feature_cols),
        "total_features": len(feature_cols) * window_size,
    }
    with open(models_dir / "sequence_threshold.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Saved metadata to sequence_threshold.json")
    
    # Analyze and save feature importance
    logger.info("Analyzing feature importance...")
    top_features = analyze_feature_importance(model, feature_cols, window_size, top_k=30)
    with open(models_dir / "sequence_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(top_features, f, ensure_ascii=False, indent=2)
    logger.info("Saved feature importance to sequence_feature_importance.json")
