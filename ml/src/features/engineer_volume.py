import pandas as pd

def calculate_volume_roc(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Oblicz Rate of Change wolumenu (procentowa zmiana wolumenu)."""
    return df['Volume'].pct_change(period)
