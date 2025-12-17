"""Sequence creation and filtering module.

Purpose:
    Create sliding windows of features and apply session/trend/pullback filters.

Public API:
    - SequenceFilterConfig: Configuration dataclass for filters

Example:
    >>> from ml.src.sequences.config import SequenceFilterConfig
    >>> config = SequenceFilterConfig(enable_trend_filter=True)
"""

from .config import SequenceFilterConfig

__all__ = ["SequenceFilterConfig"]
