"""Data loading module for OHLCV time-series data.

Purpose:
    Load and concatenate yearly CSV files with validation.

Example:
    >>> from pathlib import Path
    >>> data_dir = Path('ml/src/data')
    >>> df = load_all_years(data_dir)
    >>> df = load_all_years(data_dir, year_filter=[2023, 2024])
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .validators import validate_schema

logger = logging.getLogger(__name__)


def load_all_years(data_dir: Path, year_filter: Optional[List[int]] = None) -> pd.DataFrame:
    """Load and validate all available yearly CSVs.

    Args:
        data_dir: Directory containing XAU_1m_data_*.csv files
        year_filter: Optional list of years to load (e.g., [2023, 2024])

    Returns:
        Concatenated DataFrame indexed by datetime, strictly increasing

    Raises:
        FileNotFoundError: If no data files found
        ValueError: On schema validation failures
    """
    files = sorted(data_dir.glob("XAU_1m_data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    # Filter by year if specified
    if year_filter:
        filtered_files = []
        for fp in files:
            year_str = fp.stem.split('_')[-1]  # Extract year from XAU_1m_data_YYYY
            if year_str.isdigit() and int(year_str) in year_filter:
                filtered_files.append(fp)
        files = filtered_files
        logger.info(f"Year filter applied: loading only {year_filter}")
    
    if not files:
        raise FileNotFoundError(f"No data files found matching filter in {data_dir}")
    
    dfs: List[pd.DataFrame] = []
    
    # Optimize CSV reading with explicit dtypes
    dtype_dict = {
        "Open": np.float32,
        "High": np.float32,
        "Low": np.float32,
        "Close": np.float32,
        "Volume": np.float32,
    }
    
    for fp in files:
        try:
            df = pd.read_csv(
                fp,
                sep=";",
                parse_dates=["Date"],
                dayfirst=False,
                encoding="utf-8",
                on_bad_lines="warn",
                dtype=dtype_dict,
            )
        except ValueError:
            # Fallback if columns don't match exactly (e.g. extra spaces)
            df = pd.read_csv(
                fp,
                sep=";",
                parse_dates=["Date"],
                dayfirst=False,
                encoding="utf-8",
                on_bad_lines="warn",
            )
            
        df = df.rename(columns={c: c.strip() for c in df.columns})
        if "Date" not in df.columns:
            raise ValueError(f"File {fp} missing 'Date' column")
        
        # Drop rows with invalid dates
        bad_dates = df["Date"].isna().sum()
        if bad_dates:
            logger.warning(f"File {fp}: Dropping {bad_dates} rows with invalid Date")
            df = df.dropna(subset=["Date"])
        
        df = df.set_index("Date")
        validate_schema(df)
        dfs.append(df)
    
    data = pd.concat(dfs, axis=0)
    data = data[~data.index.duplicated(keep="first")]
    data.sort_index(inplace=True)
    return data
