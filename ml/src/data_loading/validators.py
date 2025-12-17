"""OHLCV data validation module.

Purpose:
    Validate schema, data types, price constraints, and handle invalid data.

Example:
    >>> df = load_data()
    >>> validate_schema(df)  # Raises ValueError if invalid
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate OHLCV schema and basic price constraints.

    Args:
        df: DataFrame with OHLCV columns

    Raises:
        ValueError: On missing columns, non-positive prices, or High<Low inconsistencies
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric dtypes (coerce and drop bad rows)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs after coercion
    before = len(df)
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with invalid numeric values")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC contains non-positive values")
    if (df["Volume"] < 0).any():
        raise ValueError("Volume contains negative values")
    if (df["High"] < df["Low"]).any():
        raise ValueError("Price inconsistency: High < Low detected")
    if df.index.has_duplicates:
        logger.warning("Duplicate timestamps detected; dropping duplicates")
        df.drop_duplicates(inplace=True)
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
