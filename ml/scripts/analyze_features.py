"""CLI script for analyzing feature importance from trained XGBoost models.

This script loads a trained model and performs comprehensive feature importance
analysis, saving results to ml/outputs/analysis/ in JSON format.

Features:
- Loads pickled model from ml/outputs/models/
- Loads feature column names and window size from model metadata
- Analyzes feature importance using XGBoost feature_importances_
- Maps feature importance to human-readable names with time offsets
- Saves detailed analysis to JSON

Example:
    # Analyze with default model path
    python ml/scripts/analyze_features.py
    
    # Analyze with custom model path
    python ml/scripts/analyze_features.py --model-path ml/outputs/models/
    
    # Analyze top-30 features instead of default
    python ml/scripts/analyze_features.py --top-k 30
"""

import sys
from pathlib import Path as PathlibPath

# Add project root to Python path
project_root = PathlibPath(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from ml.src.training import analyze_feature_importance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("ml/outputs/logs/analyze_features.log"),
            mode="a",
        ),
    ],
)
logger = logging.getLogger(__name__)


def load_model_and_metadata(
    models_dir: Path,
) -> Tuple[CalibratedClassifierCV, List[str], int]:
    """Load trained model and extract metadata.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Tuple of (model, feature_cols, window_size)

    Raises:
        FileNotFoundError: If required artifact files are missing
        pickle.UnpicklingError: If artifact files are corrupted
        ValueError: If metadata is invalid
    """
    import pickle

    logger.info(f"Loading model and metadata from {models_dir}")

    # Load model
    model_path = models_dir / "sequence_xgb_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"✅ Loaded model from {model_path}")

    # Load feature columns
    feature_cols_path = models_dir / "sequence_feature_columns.json"
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {feature_cols_path}")
    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    logger.info(f"✅ Loaded {len(feature_cols)} feature columns")

    # Load metadata to get window_size
    metadata_path = models_dir / "sequence_threshold.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    window_size = metadata.get("window_size")
    if window_size is None:
        raise ValueError("window_size not found in metadata")
    logger.info(f"✅ Loaded window_size={window_size}")

    return model, feature_cols, window_size


def create_feature_importance_report(
    model: CalibratedClassifierCV,
    feature_cols: List[str],
    window_size: int,
    top_k: int = 20,
) -> Dict:
    """Create comprehensive feature importance report.

    Args:
        model: Trained calibrated classifier
        feature_cols: List of per-candle feature names
        window_size: Number of candles in input window
        top_k: Number of top features to include

    Returns:
        Dictionary containing detailed analysis report
    """
    logger.info("Analyzing feature importance...")

    # Get top features
    top_features = analyze_feature_importance(
        model=model,
        feature_cols=feature_cols,
        window_size=window_size,
        top_k=top_k,
    )

    # Analyze time distribution (how importance spreads across time steps)
    logger.info("Analyzing time distribution...")
    time_importance = {}
    for feat_name, importance in top_features.items():
        # Extract time offset from feature name (t-0_feat, t-1_feat, etc.)
        if "_" in feat_name:
            time_offset = feat_name.split("_")[0]  # e.g., "t-0", "t-1"
            time_importance[time_offset] = (
                time_importance.get(time_offset, 0) + importance
            )

    sorted_time = sorted(time_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top time steps by aggregated importance:")
    for time_offset, total_importance in sorted_time[:10]:
        logger.info(f"  {time_offset:10s} = {total_importance:.6f}")

    # Analyze feature type distribution
    logger.info("Analyzing feature type distribution...")
    feature_type_importance = {}
    for feat_name, importance in top_features.items():
        # Extract feature type from feature name
        if "_" in feat_name:
            feat_type = feat_name.split("_", 1)[1]  # e.g., "close", "rsi"
        else:
            feat_type = feat_name
        feature_type_importance[feat_type] = (
            feature_type_importance.get(feat_type, 0) + importance
        )

    sorted_types = sorted(
        feature_type_importance.items(), key=lambda x: x[1], reverse=True
    )
    logger.info("Feature types by aggregated importance:")
    for feat_type, total_importance in sorted_types[:15]:
        logger.info(f"  {feat_type:30s} = {total_importance:.6f}")

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "feature_columns": feature_cols,
            "window_size": window_size,
            "total_features": len(feature_cols) * window_size,
        },
        "top_features": top_features,
        "time_distribution": dict(sorted_time),
        "feature_type_distribution": dict(sorted_types),
    }

    return report


def save_analysis_report(
    report: Dict,
    output_dir: Path,
    prefix: str = "feature_importance",
) -> Path:
    """Save feature importance analysis to JSON file.

    Args:
        report: Analysis report dictionary
        output_dir: Directory to save analysis
        prefix: Filename prefix (default: feature_importance)

    Returns:
        Path to saved report file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{prefix}_{timestamp}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Report saved to {report_path}")
    return report_path


def print_summary(report: Dict):
    """Print summary of feature importance analysis.

    Args:
        report: Analysis report dictionary
    """
    logger.info("=" * 70)
    logger.info("FEATURE IMPORTANCE ANALYSIS SUMMARY")
    logger.info("=" * 70)

    model_info = report["model_info"]
    logger.info(f"Model Info:")
    logger.info(f"  Feature columns: {len(model_info['feature_columns'])}")
    logger.info(f"  Window size: {model_info['window_size']}")
    logger.info(f"  Total features: {model_info['total_features']}")

    logger.info(f"\nTop 10 Features:")
    top_features = list(report["top_features"].items())[:10]
    for rank, (feat_name, importance) in enumerate(top_features, 1):
        logger.info(f"  {rank:2d}. {feat_name:40s} = {importance:.6f}")

    logger.info(f"\nTop 5 Feature Types:")
    feature_types = list(report["feature_type_distribution"].items())[:5]
    for rank, (feat_type, importance) in enumerate(feature_types, 1):
        logger.info(f"  {rank}. {feat_type:30s} = {importance:.6f}")

    logger.info("=" * 70)


def main():
    """Main entry point for analyze_features CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze feature importance from trained XGBoost model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with default model path
  python ml/scripts/analyze_features.py
  
  # Analyze with custom model path
  python ml/scripts/analyze_features.py --model-path ml/outputs/models/
  
  # Analyze top-30 features
  python ml/scripts/analyze_features.py --top-k 30
  
  # Save to custom output directory
  python ml/scripts/analyze_features.py --output-dir ml/analysis/
        """,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/outputs/models"),
        help="Directory containing model artifacts (default: ml/outputs/models)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/outputs/analysis"),
        help="Directory to save analysis (default: ml/outputs/analysis)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top features to analyze (default: 20)",
    )

    args = parser.parse_args()

    try:
        # Load model and metadata
        model, feature_cols, window_size = load_model_and_metadata(args.model_path)

        # Create analysis report
        report = create_feature_importance_report(
            model=model,
            feature_cols=feature_cols,
            window_size=window_size,
            top_k=args.top_k,
        )

        # Save report
        report_path = save_analysis_report(report, args.output_dir)

        # Print summary
        print_summary(report)

        logger.info(f"✅ Feature importance analysis completed successfully!")
        logger.info(f"📊 Analysis saved to: {report_path}")

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
