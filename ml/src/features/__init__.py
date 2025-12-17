"""Feature engineering module.

Purpose:
    Engineer technical indicators and candle-level features.

Exports:
    - engineer_candle_features(): Main feature engineering function
    - Technical indicators (EMA, RSI, ADX, MACD, etc.)
    - M5 context features
    - Time-based features
"""

from ml.src.features.engineer import engineer_candle_features

__all__ = ["engineer_candle_features"]

