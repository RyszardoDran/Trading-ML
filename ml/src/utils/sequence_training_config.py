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

from ml.src.utils.risk_config import SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER

# Get absolute path to ml/ directory
_config_file = Path(__file__)
_utils_dir = _config_file.parent
_src_dir = _utils_dir.parent
_ml_dir = _src_dir.parent


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
    
    # Paths (absolute, relative to ml/ directory)
    data_dir: Path = _ml_dir / "src" / "data"
    models_dir: Path = _ml_dir / "src" / "models"
    outputs_dir: Path = _ml_dir / "outputs"
    
    # Thresholds and parameters
    window_size: int = 60
    atr_multiplier_sl: float = SL_ATR_MULTIPLIER
    atr_multiplier_tp: float = TP_ATR_MULTIPLIER
    min_hold_minutes: int = 5
    max_horizon: int = 60
    random_state: int = 42
    
    # Session and filtering
    session: str = "london_ny"
    enable_m5_alignment: bool = True
    enable_trend_filter: bool = True
    enable_pullback_filter: bool = True
    
    @property
    def outputs_logs_dir(self) -> Path:
        """Directory for training logs."""
        return self.outputs_dir / "logs"
    
    @property
    def outputs_models_dir(self) -> Path:
        """Directory for trained models."""
        return self.outputs_dir / "models"
    
    @property
    def outputs_metrics_dir(self) -> Path:
        """Directory for metrics and evaluation results."""
        return self.outputs_dir / "metrics"
    
    @property
    def outputs_analysis_dir(self) -> Path:
        """Directory for analysis (feature importance, etc.)."""
        return self.outputs_dir / "analysis"
