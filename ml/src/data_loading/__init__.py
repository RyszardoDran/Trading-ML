"""Data loading and validation module.

Purpose:
    Load and validate OHLCV time-series data from CSV files.

Public API:
    - load_all_years(): Load multiple yearly CSV files
    - validate_schema(): Validate OHLCV schema and constraints

Example:
    >>> from ml.src.data_loading import load_all_years, validate_schema
    >>> from pathlib import Path
    >>> df = load_all_years(Path('ml/src/data'))
    >>> validate_schema(df)
"""

from .loaders import load_all_years
from .validators import validate_schema

__all__ = ["load_all_years", "validate_schema"]
