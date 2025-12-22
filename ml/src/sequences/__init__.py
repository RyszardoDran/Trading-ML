"""Sequence creation and filtering module.

Purpose:
    Create sliding windows of features and apply session/trend/pullback filters.

Exports:
    - create_sequences(): Main function to create sliding windows
    - filter_by_session(): Apply session time filters
    - SequenceFilterConfig: Configuration dataclass for filters

Example:
    >>> from ml.src.sequences import create_sequences
    >>> X, y, ts = create_sequences(features, targets, window_size=100)
"""

from .sequencer import create_sequences
from .filters import filter_by_session
from .config import SequenceFilterConfig

__all__ = ["create_sequences", "filter_by_session", "SequenceFilterConfig"]

__all__ = ["SequenceFilterConfig"]
