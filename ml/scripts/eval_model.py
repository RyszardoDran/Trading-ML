"""CLI script for evaluating trained XGBoost classification models.

This script loads a trained model, scaler, and test data, then performs
comprehensive model evaluation and saves metrics to ml/outputs/metrics/.

Features:
- Loads pickled model and scaler from ml/outputs/models/
- Loads test data from specified path
- Computes comprehensive metrics: win_rate, precision, recall, F1, ROC-AUC, PR-AUC
- Saves metrics to ml/outputs/metrics/
- Supports threshold specification and daily trade caps

Example:
    # Evaluate with default test data
    python ml/scripts/eval_model.py
    
    # Evaluate with custom data path
    python ml/scripts/eval_model.py --data-path /path/to/test_data.pkl
    
    # Evaluate with custom model path
    python ml/scripts/eval_model.py --model-path ml/outputs/models/
    
    # Evaluate with custom threshold
    python ml/scripts/eval_model.py --threshold 0.65
"""

import sys
from pathlib import Path as PathlibPath

# Add project root to Python path
project_root = PathlibPath(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler

from ml.src.training import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("ml/outputs/logs/eval_model.log"),
            mode="a",
        ),
    ],
)
logger = logging.getLogger(__name__)


def load_model_artifacts(
    models_dir: Path,
) -> Tuple[CalibratedClassifierCV, RobustScaler, dict]:
    """Load trained model, scaler, and metadata from disk.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Tuple of (model, scaler, metadata_dict)

    Raises:
        FileNotFoundError: If required artifact files are missing
        pickle.UnpicklingError: If artifact files are corrupted
    """
    logger.info(f"Loading artifacts from {models_dir}")

    # Load model
    model_path = models_dir / "sequence_xgb_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"✅ Loaded model from {model_path}")

    # Load scaler
    scaler_path = models_dir / "sequence_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"✅ Loaded scaler from {scaler_path}")

    # Load metadata
    metadata_path = models_dir / "sequence_threshold.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"✅ Loaded metadata from {metadata_path}")

    return model, scaler, metadata


def load_test_data(
    data_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DatetimeIndex]]:
    """Load test features and labels.

    Expects data to be saved as pickle with structure:
    - If dict: {'X': features, 'y': labels, 'timestamps': (optional)}
    - If tuple/list: (X, y) or (X, y, timestamps)

    Args:
        data_path: Path to pickled test data

    Returns:
        Tuple of (X_test, y_test, timestamps or None)

    Raises:
        FileNotFoundError: If data file not found
        ValueError: If data structure is invalid
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    logger.info(f"Loading test data from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Parse different data formats
    if isinstance(data, dict):
        if "X" not in data or "y" not in data:
            raise ValueError("Dict data must contain 'X' and 'y' keys")
        X_test = data["X"]
        y_test = data["y"]
        timestamps = data.get("timestamps", None)
    elif isinstance(data, (tuple, list)) and len(data) >= 2:
        X_test = data[0]
        y_test = data[1]
        timestamps = data[2] if len(data) > 2 else None
    else:
        raise ValueError("Data must be dict with 'X'/'y' keys or tuple/list (X, y)")

    # Validate shapes
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Length mismatch: X_test ({len(X_test)}) vs y_test ({len(y_test)})"
        )

    logger.info(f"✅ Loaded test data: X_test shape={X_test.shape}, y_test shape={y_test.shape}")
    if timestamps is not None:
        logger.info(f"✅ Loaded timestamps: {len(timestamps)} entries")

    return X_test, y_test, timestamps


def evaluate_model(
    model: CalibratedClassifierCV,
    scaler: RobustScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    min_precision: float = 0.85,
    max_trades_per_day: Optional[int] = 5,
) -> Dict[str, float]:
    """Evaluate model on test data.

    Args:
        model: Trained calibrated classifier
        scaler: Fitted scaler for feature normalization
        X_test: Test features
        y_test: Test labels
        timestamps: Optional timestamps for daily cap
        min_precision: Minimum required precision (win rate)
        max_trades_per_day: Maximum trades per day constraint

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting model evaluation...")

    # Scale test data using fitted scaler
    X_test_scaled = scaler.transform(X_test)
    logger.info(f"✅ Scaled test features: shape={X_test_scaled.shape}")

    # Evaluate
    metrics = evaluate(
        model=model,
        X_test=X_test_scaled,
        y_test=y_test,
        min_precision=min_precision,
        test_timestamps=timestamps,
        max_trades_per_day=max_trades_per_day,
    )

    logger.info("✅ Evaluation complete!")
    logger.info("=" * 70)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 70)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name:20s} = {metric_value:.6f}")
    logger.info("=" * 70)

    return metrics


def save_metrics(
    metrics: Dict[str, float],
    output_dir: Path,
    prefix: str = "eval_model",
) -> Path:
    """Save evaluation metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save metrics
        prefix: Filename prefix (default: eval_model)

    Returns:
        Path to saved metrics file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / f"{prefix}_{timestamp}.json"

    # Save metrics with additional context
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Metrics saved to {metrics_path}")
    return metrics_path


def main():
    """Main entry point for eval_model CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained XGBoost classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default paths
  python ml/scripts/eval_model.py
  
  # Evaluate with custom data path
  python ml/scripts/eval_model.py --data-path data/test_data.pkl
  
  # Evaluate with custom model path
  python ml/scripts/eval_model.py --model-path ml/outputs/models/
  
  # Evaluate with custom minimum precision requirement
  python ml/scripts/eval_model.py --min-precision 0.80
        """,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/outputs/models"),
        help="Directory containing model artifacts (default: ml/outputs/models)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("ml/outputs/test_data.pkl"),
        help="Path to test data pickle file (default: ml/outputs/test_data.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/outputs/metrics"),
        help="Directory to save metrics (default: ml/outputs/metrics)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.85,
        help="Minimum required precision/win-rate (default: 0.85)",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=5,
        help="Maximum allowed trades per day (default: 5)",
    )

    args = parser.parse_args()

    try:
        # Load artifacts
        model, scaler, metadata = load_model_artifacts(args.model_path)
        logger.info(f"Model metadata: {metadata}")

        # Load test data
        X_test, y_test, timestamps = load_test_data(args.data_path)

        # Evaluate model
        metrics = evaluate_model(
            model=model,
            scaler=scaler,
            X_test=X_test,
            y_test=y_test,
            timestamps=timestamps,
            min_precision=args.min_precision,
            max_trades_per_day=args.max_trades_per_day,
        )

        # Save metrics
        metrics_path = save_metrics(metrics, args.output_dir)
        logger.info(f"✅ Evaluation completed successfully!")
        logger.info(f"📊 Metrics saved to: {metrics_path}")

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise SystemExit(1) from e
    except ValueError as e:
        logger.error(f"❌ Invalid data: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
