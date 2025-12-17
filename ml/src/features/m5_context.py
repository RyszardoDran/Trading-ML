"""M5 context features (5-minute timeframe resampling).

This module provides features derived from resampling 1-minute data to 5-minute
timeframe (M5). These features provide a higher timeframe context for the model.

Key features computed:
- ATR_M5: Average True Range on 5-minute bars
- RSI_M5: Relative Strength Index on 5-minute bars
- SMA_20_M5: 20-period moving average on 5-minute bars
- MACD_M5: MACD on 5-minute bars
- Bollinger Bands: BB position on 5-minute bars
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_m5_context(df: pd.DataFrame) -> dict:
    """Compute M5 context features by resampling 1-minute data.
    
    Resamples OHLCV data to 5-minute bars and computes indicators.
    Results are reindexed back to 1-minute data using forward fill.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex (1-minute frequency)
           Must contain columns: Open, High, Low, Close, Volume
    
    Returns:
        Dictionary with keys:
        - 'atr_m5_n': Normalized ATR (M5)
        - 'rsi_m5': RSI (M5)
        - 'dist_sma_20_m5': Distance from 20-period SMA (M5)
        - 'macd_n_m5': Normalized MACD line (M5)
        - 'bb_pos_m5': Bollinger Band position (M5)
        - 'atr_m5': ATR (M5) in original price units (for target calculation)
    
    Side effects:
        Adds 'ATR_M5' column to df for target calculation (side effect intended)
    """
    logger.info("Computing M5 context features...")
    
    close = df["Close"].astype(np.float32).clip(lower=1e-9)
    
    # Resample to 5-minute bars
    df_m5 = df.resample("5min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    
    m5_high = df_m5["High"]
    m5_low = df_m5["Low"]
    m5_close = df_m5["Close"]
    
    # -------- M5 ATR (14) --------
    m5_tr1 = m5_high - m5_low
    m5_tr2 = (m5_high - m5_close.shift(1)).abs()
    m5_tr3 = (m5_low - m5_close.shift(1)).abs()
    m5_tr = pd.concat([m5_tr1, m5_tr2, m5_tr3], axis=1).max(axis=1)
    m5_atr = m5_tr.rolling(14).mean()
    
    # -------- M5 RSI (14) --------
    m5_delta = m5_close.diff()
    m5_gain = m5_delta.clip(lower=0).rolling(14).mean()
    m5_loss = (-m5_delta.clip(upper=0)).rolling(14).mean()
    m5_rs = m5_gain / (m5_loss + 1e-9)
    m5_rsi = 100 - (100 / (1 + m5_rs))
    
    # -------- M5 SMA (20) --------
    m5_sma_20 = m5_close.rolling(20).mean()
    m5_dist_sma_20 = (m5_close - m5_sma_20) / (m5_sma_20 + 1e-9)
    
    # -------- M5 MACD (12, 26, 9) --------
    m5_ema_12 = m5_close.ewm(span=12, adjust=False).mean()
    m5_ema_26 = m5_close.ewm(span=26, adjust=False).mean()
    m5_macd = m5_ema_12 - m5_ema_26
    m5_signal = m5_macd.ewm(span=9, adjust=False).mean()
    m5_macd_n = m5_macd / (m5_close + 1e-9)
    
    # -------- M5 Bollinger Bands (20, 2) --------
    m5_bb_mid = m5_close.rolling(20).mean()
    m5_bb_std = m5_close.rolling(20).std()
    m5_bb_upper = m5_bb_mid + 2 * m5_bb_std
    m5_bb_lower = m5_bb_mid - 2 * m5_bb_std
    m5_bb_pos = (m5_close - m5_bb_lower) / (m5_bb_upper - m5_bb_lower + 1e-9)
    
    # -------- Reindex back to M1 (forward fill) --------
    # Align M5 indicators with original 1-minute timestamps
    atr_m5_aligned = m5_atr.reindex(df.index, method="ffill").fillna(0)
    rsi_m5_aligned = m5_rsi.reindex(df.index, method="ffill").fillna(50)
    dist_sma_20_m5_aligned = m5_dist_sma_20.reindex(df.index, method="ffill").fillna(0)
    macd_n_m5_aligned = m5_macd_n.reindex(df.index, method="ffill").fillna(0)
    bb_pos_m5_aligned = m5_bb_pos.reindex(df.index, method="ffill").fillna(0.5)
    
    # Store M5 ATR in original DF for target calculation (side effect intended)
    df["ATR_M5"] = atr_m5_aligned
    
    # Normalized M5 ATR for the model
    atr_m5_n = atr_m5_aligned / (close + 1e-9)
    
    return {
        "atr_m5_n": atr_m5_n,
        "rsi_m5": rsi_m5_aligned,
        "dist_sma_20_m5": dist_sma_20_m5_aligned,
        "macd_n_m5": macd_n_m5_aligned,
        "bb_pos_m5": bb_pos_m5_aligned,
        "atr_m5": atr_m5_aligned,  # For target calculation
    }
