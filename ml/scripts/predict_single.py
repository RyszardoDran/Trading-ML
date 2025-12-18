#!/usr/bin/env python3
"""
ML Model Prediction Script

Wczytuje wytrenowany model XGBoost i wykonuje predykcję na danych świeczek.

Usage:
    python predict_single.py \
        --input-file data.json \
        --models-dir path/to/models \
        --output-file output.json

Input JSON (data.json):
    [
      {"timestamp": "2025-01-01T00:00:00Z", "open": 2000, "high": 2010, "low": 1990, "close": 2005, "volume": 100000},
      ...
    ]

Output JSON (output.json):
    {
      "probability": 0.753,
      "prediction": 1,
      "features_computed": 900,
      "threshold": 0.63
    }
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
import pickle
import numpy as np


def load_model_artifacts(models_dir: Path) -> Tuple[Any, List[str], float]:
    """
    Load XGBoost model, feature columns, and threshold.
    
    Args:
        models_dir: Directory containing model artifacts
        
    Returns:
        Tuple of (model, feature_columns, threshold)
        
    Raises:
        FileNotFoundError: If required artifacts not found
    """
    print(f"[INFO] Loading model artifacts from {models_dir}")
    
    # Load feature columns
    feature_cols_path = models_dir / "sequence_feature_columns.json"
    with open(feature_cols_path, 'r') as f:
        feature_columns = json.load(f)
    print(f"[INFO] Loaded {len(feature_columns)} feature columns")
    
    # Load threshold and metadata
    threshold_path = models_dir / "sequence_threshold.json"
    with open(threshold_path, 'r') as f:
        metadata = json.load(f)
    threshold = metadata['threshold']
    print(f"[INFO] Threshold: {threshold:.2%}")
    
    # Load model
    model_path = models_dir / "sequence_xgb_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded successfully")
    
    return model, feature_columns, threshold


def generate_features(candles: List[Dict[str, float]], window_size: int = 60) -> np.ndarray:
    """
    Generate features from candle data.
    
    In a real scenario, this would:
    1. Aggregate to different timeframes (M5, M15, M60)
    2. Calculate indicators (RSI, BB, SMA, etc.)
    3. Create sequences
    
    For now, returns dummy features for demonstration.
    
    Args:
        candles: List of OHLCV dicts
        window_size: Number of historical candles
        
    Returns:
        Feature array suitable for model input
    """
    print(f"[INFO] Generating features from {len(candles)} candles")
    
    if len(candles) < window_size:
        print(f"[WARNING] Only {len(candles)} candles, window size is {window_size}")
    
    # Placeholder: Generate 900 dummy features (15 per candle × 60 candles)
    # In production, you would:
    # 1. Load scaler
    # 2. Calculate technical indicators
    # 3. Normalize features
    
    n_features = 15 * window_size  # 900 features
    features = np.random.randn(1, n_features).astype(np.float32)
    
    print(f"[INFO] Generated {n_features} features")
    return features


def predict(model: Any, features: np.ndarray, threshold: float) -> Dict[str, Any]:
    """
    Run model prediction.
    
    Args:
        model: Trained XGBoost model
        features: Input feature array
        threshold: Decision threshold
        
    Returns:
        Dict with prediction results
    """
    print("[INFO] Running prediction...")
    
    try:
        # Get prediction probability
        probability = model.predict_proba(features)[0][1]  # Class 1 (BUY) probability
        prediction = int(probability >= threshold)  # 0 or 1
        
        print(f"[INFO] Prediction: {prediction} (probability: {probability:.2%})")
        
        return {
            "probability": float(probability),
            "prediction": int(prediction),
            "features_computed": features.shape[1],
            "threshold": float(threshold)
        }
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ML model prediction on candle data")
    parser.add_argument("--input-file", required=True, help="Input JSON file with candles")
    parser.add_argument("--models-dir", required=True, help="Directory with model artifacts")
    parser.add_argument("--output-file", required=True, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input_file)
        models_dir = Path(args.models_dir)
        output_path = Path(args.output_file)
        
        # Validate inputs
        if not input_path.exists():
            print(f"[ERROR] Input file not found: {input_path}")
            return 1
        
        if not models_dir.exists():
            print(f"[ERROR] Models directory not found: {models_dir}")
            return 1
        
        # Load candles
        print(f"[INFO] Loading candles from {input_path}")
        with open(input_path, 'r') as f:
            candles = json.load(f)
        print(f"[INFO] Loaded {len(candles)} candles")
        
        # Load model
        model, feature_columns, threshold = load_model_artifacts(models_dir)
        
        # Generate features
        features = generate_features(candles)
        
        # Predict
        result = predict(model, features, threshold)
        
        # Save output
        print(f"[INFO] Saving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print("[INFO] Prediction complete")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
