"""Sequence-based training pipeline configuration.

Purpose:
    Define default paths, thresholds, and parameters specific to the sequence training pipeline
    for XAU/USD trading model.

Example:
    >>> config = PipelineConfig()
    >>> print(config.models_dir)
    >>> print(config.window_size)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Central configuration for the sequence training pipeline.
    
    Attributes:
        data_dir: Directory containing input CSV files (XAU_1m_data_*.csv)
        models_dir: Directory for saving trained models and artifacts
        outputs_dir: Directory for saving results (metrics, analysis, logs)
        window_size: Number of candles in each sequence
        atr_multiplier_sl: ATR multiplier for stop-loss
        atr_multiplier_tp: ATR multiplier for take-profit
        min_hold_minutes: Minimum holding time
        max_horizon: Maximum forward horizon
        random_state: Random seed for reproducibility
    """
    
    # Paths (relative to ml/ directory)
    data_dir: Path = Path("src/data")
    models_dir: Path = Path("src/models")
    outputs_dir: Path = Path("outputs")
    
    # Thresholds and parameters
    window_size: int = 60
    atr_multiplier_sl: float = 1.0
    atr_multiplier_tp: float = 2.0
    min_hold_minutes: int = 5
    max_horizon: int = 60
    random_state: int = 42
    
    # Session and filtering
    session: str = "london_ny"
    enable_m5_alignment: bool = True
    enable_trend_filter: bool = True
    enable_pullback_filter: bool = True
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects if needed."""
        if isinstance(self.data_dir, str):
            object.__setattr__(self, "data_dir", Path(self.data_dir))
        if isinstance(self.models_dir, str):
            object.__setattr__(self, "models_dir", Path(self.models_dir))
        if isinstance(self.outputs_dir, str):
            object.__setattr__(self, "outputs_dir", Path(self.outputs_dir))
