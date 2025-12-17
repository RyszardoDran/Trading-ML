"""Machine learning pipelines for training and evaluation.

Modules:
    - data_loading: Load and validate OHLCV data
    - features: Engineer technical indicators
    - targets: Create binary target labels
    - sequences: Create sliding windows and filters
    - training: Train and evaluate models (future)
    - utils: Utility functions
    - config: Central pipeline configuration
    - split: Chronological train/val/test splitting
    - sequence_training_pipeline: Main pipeline orchestration

Example:
    >>> from ml.src.pipelines.sequence_training_pipeline import run_pipeline
    >>> metrics = run_pipeline()
"""

__all__ = ["sequence_training_pipeline"]
