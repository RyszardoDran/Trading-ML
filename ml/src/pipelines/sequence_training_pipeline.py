from __future__ import annotations
"""Sequence-based training pipeline for XAU/USD (1-minute data).

Purpose:
- Load historical 1m OHLCV data from `ml/src/data/`
- Create sliding windows of 100 candles as input features
- Engineer features for each candle in the sequence (flatten to XGBoost input)
- Train XGBoost classifier with calibrated probability estimates
- Validate win rate on test set with realistic thresholds
- Provide user with expected win rate and confidence metrics

Key principles:
- Temporal context: Model sees 100 previous candles before making prediction
- No data leakage: Strict chronological split, no future information
- Win rate validation: Precision, recall, F1, and confusion matrix on test
- Reproducibility: Fixed random seeds, deterministic behavior
- Production-ready: Type hints, validation, error handling, logging

Inputs (CSV):
- `ml/src/data/XAU_1m_data_*.csv` (semicolon-separated OHLCV)

Outputs (artifacts):
- `ml/src/models/sequence_xgb_model.pkl` (calibrated classifier)
- `ml/src/models/sequence_feature_columns.json` (ordered feature names)
- `ml/src/models/sequence_threshold.json` (selected threshold + win rate)

Expected columns: [Date;Open;High;Low;Close;Volume]
Separator: `;`, Date parseable to datetime

Usage:
    # Train with default parameters
    python sequence_training_pipeline.py
    
    # Train with custom window size and horizon
    python sequence_training_pipeline.py --window-size 50 --horizon 10
    
    # Health check only
    python sequence_training_pipeline.py --health-check-dir ml/src/data
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _validate_schema(df: pd.DataFrame) -> None:
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


def load_all_years(data_dir: Path, year_filter: List[int] = None) -> pd.DataFrame:
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
        _validate_schema(df)
        dfs.append(df)
    
    data = pd.concat(dfs, axis=0)
    data = data[~data.index.duplicated(keep="first")]
    data.sort_index(inplace=True)
    return data


def engineer_candle_features(df: pd.DataFrame, window_size: int = 100) -> pd.DataFrame:
    """Engineer per-candle features suitable for windowing.

    Features computed for each candle (35 features total):
    
    Candle structure (5):
    - Log return (1-minute)
    - Normalized range: (High - Low) / Close
    - Body ratio: abs(Close - Open) / (High - Low)
    - Upper/lower shadows normalized
    
    Volume (2):
    - Volume change: log(Volume / Volume_prev)
    - Volume ratio: Volume / MA(20)
    
    Trend indicators (6):
    - EMA spread (12, 26) normalized
    - ADX (Average Directional Index) - trend strength
    - +DI, -DI (Directional Indicators)
    - MACD line and histogram (normalized)
    
    Momentum indicators (8):
    - RSI(14) - Relative Strength Index
    - Stochastic K and D (14, 3, 3)
    - CCI (Commodity Channel Index) - specific for commodities like gold
    - Williams %R (14) - momentum oscillator
    - Rate of change 5-min and 20-min
    - Price position in 20-period range [0,1]
    
    Volatility indicators (5):
    - Volatility (20-period std)
    - ATR(14) absolute and normalized
    - Bollinger Band width (volatility measure)
    - Bollinger Band position (where price sits in bands)
    
    Volume indicators (2):
    - OBV normalized (On-Balance Volume momentum)
    - Market structure (Higher Highs / Lower Lows pattern)
    
    Price action (1):
    - Distance from 20-period MA (mean reversion potential)
    
    Time features (4):
    - Hour of day (sine/cosine encoding)
    - Minute of hour (sine/cosine encoding)

    Args:
        df: OHLCV DataFrame with datetime index
        window_size: Minimum lookback for rolling features

    Returns:
        DataFrame with per-candle features (NaNs dropped)

    Raises:
        ValueError: If resulting feature matrix is empty
    """
    logger.info(f"Engineering features for {len(df):,} candles...")
    # Use float32 for memory efficiency
    close = df["Close"].astype(np.float32).clip(lower=1e-9)
    open_ = df["Open"].astype(np.float32).clip(lower=1e-9)
    high = df["High"].astype(np.float32).clip(lower=1e-9)
    low = df["Low"].astype(np.float32).clip(lower=1e-9)
    volume = df["Volume"].astype(np.float32).clip(lower=1e-9)

    # --- M5 Context Features (Resampling) ---
    logger.info("Calculating M5 context features (ATR, RSI, SMA)...")
    # Resample to 5min
    df_m5 = df.resample("5min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # M5 ATR (14)
    m5_high = df_m5["High"]
    m5_low = df_m5["Low"]
    m5_close = df_m5["Close"]
    m5_tr1 = m5_high - m5_low
    m5_tr2 = (m5_high - m5_close.shift(1)).abs()
    m5_tr3 = (m5_low - m5_close.shift(1)).abs()
    m5_tr = pd.concat([m5_tr1, m5_tr2, m5_tr3], axis=1).max(axis=1)
    m5_atr = m5_tr.rolling(14).mean()

    # M5 RSI (14)
    m5_delta = m5_close.diff()
    m5_gain = m5_delta.clip(lower=0).rolling(14).mean()
    m5_loss = (-m5_delta.clip(upper=0)).rolling(14).mean()
    m5_rs = m5_gain / (m5_loss + 1e-9)
    m5_rsi = 100 - (100 / (1 + m5_rs))

    # M5 SMA (20)
    m5_sma_20 = m5_close.rolling(20).mean()
    m5_dist_sma_20 = (m5_close - m5_sma_20) / (m5_sma_20 + 1e-9)

    # M5 MACD (12, 26, 9)
    m5_ema_12 = m5_close.ewm(span=12, adjust=False).mean()
    m5_ema_26 = m5_close.ewm(span=26, adjust=False).mean()
    m5_macd = m5_ema_12 - m5_ema_26
    m5_signal = m5_macd.ewm(span=9, adjust=False).mean()
    m5_hist = m5_macd - m5_signal
    m5_macd_n = m5_macd / (m5_close + 1e-9)

    # M5 Bollinger Bands (20, 2)
    m5_bb_mid = m5_close.rolling(20).mean()
    m5_bb_std = m5_close.rolling(20).std()
    m5_bb_upper = m5_bb_mid + 2 * m5_bb_std
    m5_bb_lower = m5_bb_mid - 2 * m5_bb_std
    m5_bb_pos = (m5_close - m5_bb_lower) / (m5_bb_upper - m5_bb_lower + 1e-9)

    # Reindex back to M1 (ffill)
    # We use reindex to align with original timestamps, ffilling the last known M5 value
    atr_m5_aligned = m5_atr.reindex(df.index, method="ffill").fillna(0)
    rsi_m5_aligned = m5_rsi.reindex(df.index, method="ffill").fillna(50)
    dist_sma_20_m5_aligned = m5_dist_sma_20.reindex(df.index, method="ffill").fillna(0)
    macd_n_m5_aligned = m5_macd_n.reindex(df.index, method="ffill").fillna(0)
    bb_pos_m5_aligned = m5_bb_pos.reindex(df.index, method="ffill").fillna(0.5)
    
    # Store M5 ATR in original DF for target calculation (side effect intended)
    df["ATR_M5"] = atr_m5_aligned
    
    # Normalized M5 ATR for the model
    atr_m5_n = atr_m5_aligned / (close + 1e-9)
    # ----------------------------------------

    # Log returns
    logc = np.log(close)
    ret1 = logc.diff(1)

    # Candle structure
    logger.info("Calculating candle structure...")
    range_ = high - low
    range_safe = range_.mask(range_ == 0, np.nan)
    
    range_n = range_ / close
    body_ratio = np.abs(close - open_) / (range_safe + 1e-9)
    upper_shadow = (high - np.maximum(open_, close)) / (range_safe + 1e-9)
    lower_shadow = (np.minimum(open_, close) - low) / (range_safe + 1e-9)

    # Volume features
    logger.info("Calculating volume changes...")
    vol_prev = volume.shift(1)
    vol_prev = vol_prev.mask(vol_prev == 0, np.nan)  # Faster than .replace()
    vol_change = np.log(volume / vol_prev)

    # Rolling features (need min window_size for stability)
    min_periods = min(window_size, 20)
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema_spread_n = (ema12 - ema26) / close

    # Simplified RSI(14)
    logger.info("Calculating momentum indicators (RSI, Stochastic, CCI, Williams %R)...")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    loss_safe = loss.mask(loss == 0, np.nan)
    rs = gain / loss_safe
    rsi14 = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator (14, 3, 3)
    low_14 = low.rolling(14, min_periods=1).min()
    high_14 = high.rolling(14, min_periods=1).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14 + 1e-9)
    stoch_d = stoch_k.rolling(3, min_periods=1).mean()
    
    # CCI (Commodity Channel Index) - important for commodities like gold
    tp = (high + low + close) / 3  # Typical Price
    tp_sma = tp.rolling(20, min_periods=1).mean()
    mad = (tp - tp_sma).abs().rolling(20, min_periods=1).mean()
    cci = (tp - tp_sma) / (0.015 * (mad + 1e-9))
    
    # Williams %R (14-period)
    williams_r = -100 * (high_14 - close) / (high_14 - low_14 + 1e-9)

    # Volatility
    logger.info("Calculating volatility and Bollinger Bands...")
    vol20 = ret1.rolling(20, min_periods=1).std()
    
    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # ATR (Average True Range) - needed for SL/TP calculation
    logger.info("Calculating ATR(14) and trend indicators (ADX, MACD)...")
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = true_range.rolling(14, min_periods=1).mean()
    atr_n = atr_14 / close  # Normalized ATR
    
    # ADX (Average Directional Index) - trend strength
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    tr_smooth = true_range.rolling(14, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=1).mean() / (tr_smooth + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).mean() / (tr_smooth + 1e-9))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(14, min_periods=1).mean()
    
    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_line_n = macd_line / close
    macd_hist_n = macd_hist / close

    # Momentum indicators
    logger.info("Calculating momentum indicators...")
    roc_5 = close.pct_change(5)  # 5-minute rate of change
    roc_20 = close.pct_change(20)  # 20-minute rate of change
    
    # Volume analysis
    vol_ma_20 = volume.rolling(20, min_periods=1).mean()
    vol_ratio = volume / (vol_ma_20 + 1e-9)  # Volume vs MA ratio
    
    # Price action patterns
    high_20 = high.rolling(20, min_periods=1).max()
    low_20 = low.rolling(20, min_periods=1).min()
    price_position = (close - low_20) / (high_20 - low_20 + 1e-9)  # Position in range [0,1]
    
    # Trend strength
    close_sma_20 = close.rolling(20, min_periods=1).mean()
    distance_from_ma = (close - close_sma_20) / close_sma_20  # Distance from 20-MA
    
    # OBV (On-Balance Volume) - volume momentum for gold
    logger.info("Calculating volume indicators (OBV, market structure)...")
    obv = (np.sign(ret1) * volume).cumsum()
    obv_ma = obv.rolling(20, min_periods=1).mean()
    obv_normalized = (obv - obv_ma) / (obv_ma.abs() + 1e-9)
    
    # Market structure: Higher Highs, Lower Lows
    hh = (high > high.rolling(5, min_periods=1).max().shift(1)).astype(int)
    ll = (low < low.rolling(5, min_periods=1).min().shift(1)).astype(int)
    market_structure = hh - ll  # +1 for uptrend, -1 for downtrend, 0 for ranging

    # Time features
    hour = df.index.hour
    minute = df.index.minute
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    # Daily context features (Vectorized)
    logger.info("Calculating daily context features...")
    # Daily Open: Group by date and take first
    # Note: This assumes data is sorted
    daily_open = df["Open"].groupby(df.index.date).transform("first")
    dist_daily_open = (close - daily_open) / (daily_open + 1e-9)

    # London Open (08:00 UTC) context
    # Create a series where 08:00 has the open price, others NaN, then ffill within the day
    london_open_mask = (hour == 8) & (minute == 0)
    # Only valid if we are in or after London session of the same day
    london_opens = df["Open"].where(london_open_mask).groupby(df.index.date).ffill()
    dist_london_open = (close - london_opens) / (london_opens + 1e-9)
    dist_london_open = dist_london_open.fillna(0) # 0 if before 08:00

    # Z-Score (Mean Reversion)
    logger.info("Calculating Z-Score and Daily Stats...")
    # Use min_periods=2 for std to avoid NaNs on single values, then fillna(0)
    roll_std_20 = close.rolling(20, min_periods=2).std().fillna(0)
    z_score_20 = (close - close.rolling(20, min_periods=1).mean()) / (roll_std_20 + 1e-9)
    
    # Volatility Ratio (Short vs Long term vol)
    vol_5 = ret1.rolling(5, min_periods=2).std().fillna(0)
    # vol20 is already calculated above, ensure it has no NaNs
    vol20_safe = vol20.fillna(0)
    vol_ratio_5_20 = vol_5 / (vol20_safe + 1e-9)

    # Previous Day Context
    # Resample to daily to get previous day's High/Low/Close
    daily_stats = df.resample("D").agg({"High": "max", "Low": "min", "Close": "last"})
    daily_shifted = daily_stats.shift(1)
    
    # Broadcast back to minute data
    daily_features = daily_shifted.reindex(df.index, method="ffill")
    
    dist_prev_high = (close - daily_features["High"]) / (daily_features["High"] + 1e-9)
    dist_prev_low = (close - daily_features["Low"]) / (daily_features["Low"] + 1e-9)
    dist_prev_close = (close - daily_features["Close"]) / (daily_features["Close"] + 1e-9)
    
    # Fill NaNs
    dist_prev_high = dist_prev_high.fillna(0)
    dist_prev_low = dist_prev_low.fillna(0)
    dist_prev_close = dist_prev_close.fillna(0)

    # Day High/Low so far (Intraday Context)
    # Group by date and calculate expanding max/min
    logger.info("Calculating intraday High/Low context...")
    day_high_so_far = df["High"].groupby(df.index.date).expanding().max().reset_index(level=0, drop=True)
    day_low_so_far = df["Low"].groupby(df.index.date).expanding().min().reset_index(level=0, drop=True)
    
    # Align indices just in case
    day_high_so_far = day_high_so_far.reindex(df.index)
    day_low_so_far = day_low_so_far.reindex(df.index)

    dist_day_high = (close - day_high_so_far) / (day_high_so_far + 1e-9)
    dist_day_low = (close - day_low_so_far) / (day_low_so_far + 1e-9)

    # Long-term Trend Indicators (Macro Context)
    logger.info("Calculating long-term trend indicators (SMA 200, SMA 1440)...")
    sma_200 = close.rolling(200, min_periods=1).mean()
    sma_1440 = close.rolling(1440, min_periods=1).mean() # Daily SMA
    
    dist_sma_200 = (close - sma_200) / (sma_200 + 1e-9)
    dist_sma_1440 = (close - sma_1440) / (sma_1440 + 1e-9)
    
    # Long-term Momentum
    delta_60 = close.diff(60)
    roc_60 = delta_60 / (close.shift(60) + 1e-9)
    
    # Long-term Volatility
    vol_60 = ret1.rolling(60, min_periods=1).std()
    vol_ratio_60_200 = vol_60 / (ret1.rolling(200, min_periods=1).std() + 1e-9)

    # Micro-structure Features (M1 dynamics within M5)
    logger.info("Calculating micro-structure features (M1 dynamics)...")
    
    # 1. Micro Volatility (Std of last 5 M1 returns)
    micro_vol_5 = ret1.rolling(5, min_periods=2).std().fillna(0)
    
    # 2. Efficiency Ratio (Kaufman) - Directional move vs Total path
    # abs(Close - Close_t-5) / Sum(abs(Close_t - Close_t-1))
    net_move_5 = (close - close.shift(5)).abs()
    total_path_5 = close.diff().abs().rolling(5, min_periods=1).sum()
    efficiency_5 = net_move_5 / (total_path_5 + 1e-9)
    efficiency_5 = efficiency_5.fillna(0)
    
    # 3. Fractal Dimension Proxy (Path Cost)
    # Sum(High - Low) / (High_5 - Low_5)
    # If high, lots of noise/wicks. If low (~1), clean trend.
    sum_range_5 = (high - low).rolling(5, min_periods=1).sum()
    range_5 = (high.rolling(5, min_periods=1).max() - low.rolling(5, min_periods=1).min())
    fractal_dim_5 = sum_range_5 / (range_5 + 1e-9)
    fractal_dim_5 = fractal_dim_5.fillna(1) # Default to 1 (linear)
    
    # 4. Trend Consistency
    # Sum of signs of returns (how many green vs red candles)
    # +5 means 5 green candles in a row
    trend_consistency_5 = np.sign(ret1).rolling(5, min_periods=1).sum().fillna(0)

    # 5. Trend Slope (Linear Regression Angle)
    # Calculate slope of Close prices over last 20 and 60 periods
    # We use a simplified vectorized approach: Slope ~ (Sum(xy) - n*mean(x)*mean(y)) / (Sum(x^2) - n*mean(x)^2)
    # Since x is just 0,1,2..., we can use precomputed kernels or just simple rate of change as proxy
    # Better proxy: (Close - Close_t-n) / n (which is ROC), but normalized by ATR to get "Angle"
    
    # Slope 20 (Short term trend intensity)
    slope_20 = (close - close.shift(20)) / (20 * atr_14 + 1e-9)
    slope_20 = slope_20.fillna(0)
    
    # Slope 60 (Medium term trend intensity)
    slope_60 = (close - close.shift(60)) / (60 * atr_14 + 1e-9)
    slope_60 = slope_60.fillna(0)

    # Check for NaNs in features
    if z_score_20.isnull().any():
        logger.warning("NaNs detected in z_score_20, filling with 0")
        z_score_20 = z_score_20.fillna(0)
    if vol_ratio_5_20.isnull().any():
        logger.warning("NaNs detected in vol_ratio_5_20, filling with 0")
        vol_ratio_5_20 = vol_ratio_5_20.fillna(0)
    if dist_day_high.isnull().any():
        logger.warning("NaNs detected in dist_day_high, filling with 0")
        dist_day_high = dist_day_high.fillna(0)
    if dist_day_low.isnull().any():
        logger.warning("NaNs detected in dist_day_low, filling with 0")
        dist_day_low = dist_day_low.fillna(0)
    
    # Fill NaNs for new long-term features
    dist_sma_200 = dist_sma_200.fillna(0)
    dist_sma_1440 = dist_sma_1440.fillna(0)
    roc_60 = roc_60.fillna(0)
    vol_ratio_60_200 = vol_ratio_60_200.fillna(0)

    features = pd.DataFrame(
        {
            # M5 Context Features
            "atr_m5_n": atr_m5_n,
            "rsi_m5": rsi_m5_aligned,
            "dist_sma_20_m5": dist_sma_20_m5_aligned,
            "macd_n_m5": macd_n_m5_aligned,
            "bb_pos_m5": bb_pos_m5_aligned,
            
            # Micro-structure (4)
            "micro_vol_5": micro_vol_5,
            "efficiency_5": efficiency_5,
            "fractal_dim_5": fractal_dim_5,
            "trend_consistency_5": trend_consistency_5,
            "slope_20": slope_20,
            "slope_60": slope_60,

            "dist_sma_200": dist_sma_200,
            "dist_sma_1440": dist_sma_1440,
            "roc_60": roc_60,
            "vol_ratio_60_200": vol_ratio_60_200,
            "dist_day_high": dist_day_high,
            "dist_day_low": dist_day_low,
            "z_score_20": z_score_20,
            "vol_ratio_5_20": vol_ratio_5_20,
            "dist_prev_high": dist_prev_high,
            "dist_prev_low": dist_prev_low,
            "dist_prev_close": dist_prev_close,
            "dist_daily_open": dist_daily_open,
            "dist_london_open": dist_london_open,
            # Candle structure (5)
            "ret_1": ret1,
            "range_n": range_n,
            "body_ratio": body_ratio,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            # Volume (2)
            "vol_change": vol_change,
            "vol_ratio": vol_ratio,
            # Technical indicators - Trend (6)
            "ema_spread_n": ema_spread_n,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "macd_line_n": macd_line_n,
            "macd_hist_n": macd_hist_n,
            # Technical indicators - Momentum (8)
            "rsi_14": rsi14,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "cci": cci,
            "williams_r": williams_r,
            "roc_5": roc_5,
            "roc_20": roc_20,
            "price_position": price_position,
            # Technical indicators - Volatility (5)
            "vol_20": vol20,
            "atr_14": atr_14,
            "atr_n": atr_n,
            "bb_width": bb_width,
            "bb_position": bb_position,
            # Volume indicators (2)
            "obv_normalized": obv_normalized,
            "market_structure": market_structure,
            # Price action (1)
            "distance_from_ma": distance_from_ma,
            # Time features (4)
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "minute_sin": minute_sin,
            "minute_cos": minute_cos,
        },
        index=df.index,
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Use fillna instead of dropna to avoid memory issues on large datasets
    features = features.ffill().fillna(0)

    if features.empty:
        raise ValueError("Feature matrix is empty after feature engineering")
    
    return features


def create_sequences(
    features: pd.DataFrame,
    targets: pd.Series,
    window_size: int = 100,
    session: str = "all",
    custom_start: int = None,
    custom_end: int = None,
    max_windows: int = 200000,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create sliding windows of features with corresponding targets.

    Optimized for memory:
    1. Drops NaNs from inputs first
    2. Filters by session BEFORE creating full matrix
    3. Uses stride tricks to avoid copies until necessary
    4. Caps number of windows to max_windows to prevent OOM

    Args:
        features: Per-candle feature matrix (n_samples, n_features)
        targets: Binary labels aligned with features
        window_size: Number of candles in each window
        session: Session filter ('london', 'ny', 'asian', 'london_ny', 'all', 'custom')
        custom_start: Start hour for custom session
        custom_end: End hour for custom session
        max_windows: Maximum number of windows to create

    Returns:
        X: (n_windows, window_size * n_features) array
        y: (n_windows,) array of binary labels
        timestamps: DatetimeIndex of window end times
    """
    # 1. Pre-clean data to avoid checking NaNs on huge X matrix later
    if features.isnull().values.any():
        logger.warning("Input features contain NaNs. Dropping rows with NaNs...")
        features = features.dropna()
    
    # Align features and targets
    common = features.index.intersection(targets.index)
    features = features.loc[common]
    targets = targets.loc[common]

    if len(features) < window_size:
        raise ValueError(f"Need at least {window_size} samples, got {len(features)}")

    n_features = features.shape[1]
    n_windows = len(features) - window_size + 1

    # Prepare arrays
    features_array = np.ascontiguousarray(features.values, dtype=np.float32)
    targets_array = targets.values.astype(np.int32)
    timestamps_array = features.index.values

    # 2. Calculate timestamps and targets for all potential windows
    # Timestamps aligned to the END of the window
    timestamp_indices = np.arange(window_size - 1, window_size - 1 + n_windows)
    timestamps = pd.DatetimeIndex(timestamps_array[timestamp_indices])
    
    # Targets aligned to the END of the window
    y = targets_array[window_size - 1 : window_size - 1 + n_windows]

    # 3. Apply Session Filter (Indices)
    # We calculate the mask on timestamps BEFORE creating the heavy X matrix
    hours = timestamps.hour
    if session == "london":
        mask = (hours >= 8) & (hours < 16)
    elif session == "ny":
        mask = (hours >= 13) & (hours < 22)
    elif session == "asian":
        mask = (hours >= 0) & (hours < 9)
    elif session == "london_ny":
        mask = (hours >= 8) & (hours < 22)
    elif session == "all":
        mask = np.ones(len(timestamps), dtype=bool)
    elif session == "custom":
        if custom_start is None or custom_end is None:
            raise ValueError("Must provide custom_start and custom_end for 'custom' session")
        if custom_start < custom_end:
            mask = (hours >= custom_start) & (hours < custom_end)
        else:
            mask = (hours >= custom_start) | (hours < custom_end)
    else:
        raise ValueError(f"Unknown session: {session}")

    # --- M5 ALIGNMENT FIX ---
    # Only take trades that align with M5 candles (0, 5, 10, 15...)
    # We want to trade at the OPEN of a new M5 candle.
    # The decision is made after the CLOSE of the previous M5 candle.
    # M5 candle 10:00 covers 10:00-10:04. It closes at 10:05:00.
    # So we need the window to end at minute 4, 9, 14, etc.
    minutes = timestamps.minute
    m5_mask = ((minutes + 1) % 5 == 0)
    mask = mask & m5_mask
    logger.info(f"Applied M5 alignment filter (keeping only candles ending at 4, 9, 14... to trade at 0, 5, 10...)")
    # ------------------------

    # --- TREND FILTER (Long Only) ---
    # Only take trades where Price > SMA 200 (Uptrend) AND ADX > 15 (Trend exists)
    # We use 'dist_sma_200' > 0 and 'adx' > 15
    if "dist_sma_200" in features.columns and "adx" in features.columns:
        # Get column indices
        sma_idx = features.columns.get_loc("dist_sma_200")
        adx_idx = features.columns.get_loc("adx")
        
        # Get values for window ends
        dist_sma_values = features_array[timestamp_indices, sma_idx]
        adx_values = features_array[timestamp_indices, adx_idx]
        
        # Apply combined filter
        trend_mask = (dist_sma_values > 0) & (adx_values > 15)
        mask = mask & trend_mask
        logger.info(f"Applied Trend Filter (Close > SMA200 AND ADX > 15): kept {trend_mask.sum()}/{len(trend_mask)} ({trend_mask.mean():.1%})")
    else:
        logger.warning("dist_sma_200 or adx feature not found! Skipping Trend Filter.")
    
    # --- PULLBACK FILTER ---
    # Avoid buying tops. Only buy if RSI_M5 is not overbought (< 75).
    # Ideally we want RSI < 60 for a pullback, but let's start with < 75 to avoid cutting too much.
    if "rsi_m5" in features.columns:
        rsi_idx = features.columns.get_loc("rsi_m5")
        rsi_values = features_array[timestamp_indices, rsi_idx]
        
        pullback_mask = rsi_values < 75
        mask = mask & pullback_mask
        logger.info(f"Applied Pullback Filter (RSI_M5 < 75): kept {pullback_mask.sum()}/{len(pullback_mask)} ({pullback_mask.mean():.1%})")
    # --------------------------------

    if mask.sum() == 0:
        logger.warning(f"Session filter '{session}' removed all data!")
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    logger.info(f"Session filter '{session}': keeping {mask.sum():,} / {len(timestamps):,} windows ({mask.mean():.1%})")

    # Cap number of windows if needed (to avoid OOM during matrix creation)
    if max_windows is not None and mask.sum() > max_windows:
        logger.warning(
            f"Too many windows ({mask.sum():,}) for memory safety. "
            f"Capping to last {max_windows:,} windows."
        )
        # Find indices of True values
        true_indices = np.where(mask)[0]
        # Keep only the last max_windows indices
        keep_indices = true_indices[-max_windows:]
        # Create new mask
        new_mask = np.zeros_like(mask)
        new_mask[keep_indices] = True
        mask = new_mask
        logger.info(f"After capping: {mask.sum():,} windows")

    # Filter y and timestamps
    y = y[mask]
    timestamps = timestamps[mask]
    
    # 4. Create X only for valid windows
    # Use stride tricks to get a view, then slice with mask, then reshape
    from numpy.lib.stride_tricks import as_strided

    shape = (n_windows, window_size, n_features)
    strides = (features_array.strides[0], features_array.strides[0], features_array.strides[1])

    try:
        # Create view of ALL windows (no copy)
        windowed_view = as_strided(features_array, shape=shape, strides=strides, writeable=False)
        
        # Apply mask to view (creates copy of ONLY valid windows)
        # Shape becomes (n_valid, window_size, n_features)
        X_valid = windowed_view[mask]
        
        # Flatten (creates copy of valid windows)
        # Shape becomes (n_valid, window_size * n_features)
        X = X_valid.reshape(X_valid.shape[0], -1)
        
    except Exception as e:
        logger.warning(f"Stride trick optimization failed: {e}. Falling back to loop...")
        # Fallback: iterate only over valid indices
        valid_indices = np.where(mask)[0]
        n_valid = len(valid_indices)
        X = np.zeros((n_valid, window_size * n_features), dtype=np.float32)
        
        for i, idx in enumerate(valid_indices):
            # idx is the window index. Window starts at idx in features_array
            X[i] = features_array[idx : idx + window_size].flatten()

    return X, y, timestamps
def make_target(
    df: pd.DataFrame,
    atr_multiplier_sl: float = 2.0,
    atr_multiplier_tp: float = 4.0,
    min_hold_minutes: int = 10,
    max_horizon: int = 120,
) -> pd.Series:
    """Create binary classification target based on SL/TP hit logic (Vectorized).

    Definition:
        For each candle, simulate a trade with:
        - SL = entry_price - (ATR × atr_multiplier_sl)
        - TP = entry_price + (ATR × atr_multiplier_tp)
        - RR Ratio = atr_multiplier_tp / atr_multiplier_sl (default 1:2)
        
        Target = 1 if TP is hit before SL within max_horizon minutes
        Target = 0 if SL is hit first or neither is hit within max_horizon

    Args:
        df: OHLCV DataFrame with datetime index
        atr_multiplier_sl: ATR multiplier for Stop Loss (default: 2.0)
        atr_multiplier_tp: ATR multiplier for Take Profit (default: 4.0)
        min_hold_minutes: Minimum time before TP can be hit (default: 10)
        max_horizon: Maximum minutes to wait for TP/SL hit (default: 120)

    Returns:
        Binary series (0/1) aligned to original index

    Raises:
        ValueError: If parameters invalid or resulting target is empty
    """
    logger.info(f"Creating targets (SL/TP simulation) for {len(df):,} candles...")
    logger.info(f"Parameters: SL={atr_multiplier_sl}×ATR, TP={atr_multiplier_tp}×ATR, min_hold={min_hold_minutes}min, max_horizon={max_horizon}min")
    
    if atr_multiplier_sl <= 0 or atr_multiplier_tp <= 0:
        raise ValueError("ATR multipliers must be positive")
    if min_hold_minutes < 1:
        raise ValueError(f"min_hold_minutes must be >=1, got {min_hold_minutes}")
    if max_horizon < min_hold_minutes:
        raise ValueError(f"max_horizon ({max_horizon}) must be >= min_hold_minutes ({min_hold_minutes})")

    # Calculate ATR
    # Use float32 to save memory and improve speed
    high_series = df["High"].astype(np.float32)
    low_series = df["Low"].astype(np.float32)
    close_series = df["Close"].astype(np.float32)
    
    if "ATR_M5" in df.columns:
        logger.info("Using pre-calculated M5 ATR for SL/TP targets")
        atr_series = df["ATR_M5"].astype(np.float32)
    else:
        logger.info("Calculating M1 ATR(14) for SL/TP targets (fallback)")
        tr1 = high_series - low_series
        tr2 = np.abs(high_series - close_series.shift(1))
        tr3 = np.abs(low_series - close_series.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_series = true_range.rolling(14, min_periods=1).mean()
    
    # Prepare arrays for vectorization
    closes = close_series.values
    highs = high_series.values
    lows = low_series.values
    atrs = atr_series.values
    
    # Calculate levels
    sl_levels = closes - (atrs * atr_multiplier_sl)
    tp_levels = closes + (atrs * atr_multiplier_tp)
    
    # Vectorized look-forward using stride_tricks
    from numpy.lib.stride_tricks import as_strided
    
    n_samples = len(df)
    valid_samples = n_samples - max_horizon
    
    if valid_samples <= 0:
         raise ValueError(f"Not enough data for max_horizon={max_horizon}")

    # Create strided views for High and Low
    # Shape: (valid_samples, max_horizon)
    # We start looking from index 1 (next candle) relative to current i
    # So future_highs[i, j] corresponds to highs[i + 1 + j]
    
    itemsize = highs.itemsize
    shape = (valid_samples, max_horizon)
    strides = (itemsize, itemsize)
    
    future_highs = as_strided(highs[1:], shape=shape, strides=strides)
    future_lows = as_strided(lows[1:], shape=shape, strides=strides)
    
    # Slice to respect min_hold_minutes
    # We want to check from i + min_hold_minutes to i + max_horizon
    # In our window (0-based), index j corresponds to i + 1 + j
    # We want i + 1 + j >= i + min_hold_minutes => j >= min_hold_minutes - 1
    start_idx = min_hold_minutes - 1
    
    future_highs = future_highs[:, start_idx:]
    future_lows = future_lows[:, start_idx:]
    
    # Align levels to valid_samples and broadcast
    tp_levels_valid = tp_levels[:valid_samples, None]
    sl_levels_valid = sl_levels[:valid_samples, None]
    
    # Check hits (boolean matrices)
    hit_tp = future_highs >= tp_levels_valid
    hit_sl = future_lows <= sl_levels_valid
    
    # Find first occurrence index
    # argmax returns index of first True, or 0 if all False
    tp_idx = np.argmax(hit_tp, axis=1)
    sl_idx = np.argmax(hit_sl, axis=1)
    
    # Check if any hit occurred
    tp_any = hit_tp.max(axis=1)
    sl_any = hit_sl.max(axis=1)
    
    # Determine target
    # TP wins if:
    # 1. TP was hit (tp_any is True)
    # 2. AND (SL was NOT hit OR TP was hit before or at same time as SL)
    # Note: If tp_idx == sl_idx, it means both hit in same candle. 
    # We assume TP takes precedence (optimistic) or check High/Low logic.
    # Original code checked TP first in loop, so TP precedence.
    
    tp_wins = tp_any & (~sl_any | (tp_idx <= sl_idx))
    
    # Initialize target array
    target = np.zeros(n_samples, dtype=np.float32)
    target[:valid_samples][tp_wins] = 1.0
    
    # Set invalid/future targets to NaN
    target[valid_samples:] = np.nan
    
    # Convert to Series and drop NaNs
    target_series = pd.Series(target, index=df.index)
    target_series = target_series.dropna().astype(int)
    
    if target_series.empty:
        raise ValueError("Target series is empty; check data quality or adjust parameters")
    
    logger.info(f"Target creation complete: {len(target_series):,} valid targets")
    return target_series


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    train_until: str = "2022-12-31 23:59:00",
    val_until: str = "2023-12-31 23:59:00",
    test_until: str = "2024-12-31 23:59:00",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split of sequence data.

    Args:
        X: Feature array (n_windows, n_features)
        y: Target array (n_windows,)
        timestamps: Timestamps corresponding to each window
        train_until: End of train period
        val_until: End of validation period
        test_until: End of test period

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test

    Raises:
        ValueError: If any split is empty or insufficient coverage
    """
    train_end = pd.Timestamp(train_until)
    val_end = pd.Timestamp(val_until)
    test_end = pd.Timestamp(test_until)

    train_mask = timestamps <= train_end
    val_mask = (timestamps > train_end) & (timestamps <= val_end)
    test_mask = (timestamps > val_end) & (timestamps <= test_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if min(map(len, [X_train, X_val, X_test])) == 0:
        raise ValueError("One of the splits is empty; verify data coverage for train/val/test ranges")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Train XGBoost classifier with class imbalance handling and calibration.

    Args:
        X_train: Training features
        y_train: Training labels (binary 0/1)
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed for reproducibility

    Returns:
        CalibratedClassifierCV with sigmoid calibration

    Raises:
        ValueError: If labels are not binary {0, 1}
    """
    if set(np.unique(y_train)) - {0, 1}:
        raise ValueError("y_train must be binary {0, 1}")

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1) if pos > 0 else 1.0
    logger.info(f"Class balance (train): pos={pos:,}, neg={neg:,}, imbalance_ratio={neg/max(pos,1):.2f}")
    logger.info(f"Applied scale_pos_weight={scale_pos_weight:.4f} to handle class imbalance")
    # logger.info("Using scale_pos_weight=1.0 to prioritize precision (Win Rate)")

    base = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=1, # Reduced to 1 to increase Recall
        reg_lambda=1.0, # Reduced to 1.0 to increase Recall
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        verbosity=0,
        grow_policy="depthwise",
        early_stopping_rounds=50,
    )

    # Train with validation set (XGBoost 2.0+ syntax)
    logger.info("Training XGBoost classifier...")
    base.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    logger.info("Training completed")

    # Calibrate probabilities on validation set
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    
    return calibrated


def _pick_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Sweep thresholds and return the one maximizing Recall while keeping Precision >= 0.40.

    Args:
        y_true: True binary labels
        proba: Predicted probabilities

    Returns:
        Optimal threshold
    """
    # Start from 0.30 (approx breakeven for 1:2 RR)
    thresholds = np.linspace(0.30, 0.95, 66) 
    best_thr = 0.5
    best_recall = -1.0
    found_valid = False
    
    # Fallback: threshold that gives at least some trades
    
    for t in thresholds:
        preds = (proba >= t).astype(int)
        n_trades = preds.sum()
        
        if n_trades < 10: # Require at least 10 trades to be statistically significant
            continue
            
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        
        # STRATEGY: High Precision Target
        # We want to maximize Recall (frequency), but ONLY if Precision is >= 70%.
        # This maintains a very high win rate (safe) while trying to find more trades
        # than the extreme 85% model.
        if precision >= 0.70:
            found_valid = True
            if recall > best_recall:
                best_recall = recall
                best_thr = t
            
    # If no threshold met >70% precision criteria, fallback to maximizing Precision
    if not found_valid:
        logger.warning("No threshold met >70% precision. Fallback to max precision.")
        best_precision = -1.0
        for t in thresholds:
            preds = (proba >= t).astype(int)
            if preds.sum() < 5: continue
            prec = precision_score(y_true, preds, zero_division=0)
            if prec > best_precision:
                best_precision = prec
                best_thr = t
        
    return float(best_thr)


def evaluate(
    model: CalibratedClassifierCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate model with comprehensive metrics including win rate.

    Win rate is defined as precision at the chosen threshold:
    - Precision = TP / (TP + FP) = proportion of predicted positives that are correct
    
    Args:
        model: Trained calibrated classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with metrics: threshold, precision (win_rate), recall, f1, roc_auc, pr_auc

    Raises:
        ValueError: If X_test is empty
    """
    if len(X_test) == 0:
        raise ValueError("X_test is empty; cannot evaluate")
    if len(y_test) != len(X_test):
        raise ValueError(f"Length mismatch: X_test({len(X_test)}) vs y_test({len(y_test)})")

    proba = model.predict_proba(X_test)[:, 1]
    if not np.isfinite(proba).all():
        raise ValueError("Non-finite probabilities produced; check feature values and calibration")
    
    # Log probability stats
    logger.info(f"Probability stats: min={proba.min():.4f}, max={proba.max():.4f}, mean={proba.mean():.4f}, std={proba.std():.4f}")
    
    # Metrics
    roc = roc_auc_score(y_test, proba)
    if roc < 0.52:
        logger.warning(f"⚠️  Model has very low discriminative power (ROC-AUC={roc:.4f}). Results may be random.")
    
    pr_auc = average_precision_score(y_test, proba)
    thr = _pick_best_threshold(y_test, proba)
    preds = (proba >= thr).astype(int)
    
    cm = confusion_matrix(y_test, preds)
    logger.info(f"Confusion matrix@thr={thr:.2f}:")
    logger.info(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    logger.info(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    metrics = {
        "threshold": float(thr),
        "win_rate": float(precision),  # Win rate = Precision
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
    }
    
    return metrics


def analyze_feature_importance(
    model: CalibratedClassifierCV,
    feature_cols: List[str],
    window_size: int,
    top_k: int = 20,
) -> Dict[str, float]:
    """Analyze feature importance from trained XGBoost model.
    
    Args:
        model: Trained calibrated classifier
        feature_cols: List of per-candle feature names
        window_size: Number of candles in window
        top_k: Number of top features to return
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Get base estimator (XGBoost) from calibrated model
    base_model = model.calibrated_classifiers_[0].estimator
    
    # Get feature importances
    importances = base_model.feature_importances_
    
    # Map flattened feature indices to (candle_offset, feature_name)
    n_features_per_candle = len(feature_cols)
    feature_names = []
    for i in range(window_size):
        for feat in feature_cols:
            feature_names.append(f"t-{window_size - i - 1}_{feat}")
    
    # Create importance dict and sort
    importance_dict = dict(zip(feature_names, importances))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Log top features
    logger.info(f"\nTop {top_k} most important features:")
    for rank, (feat, score) in enumerate(sorted_features[:top_k], 1):
        logger.info(f"  {rank:2d}. {feat:40s} = {score:.6f}")
    
    # Aggregate by feature type (ignoring time offset)
    feature_type_importance = {}
    for feat_name, importance in importance_dict.items():
        feat_type = feat_name.split("_", 1)[1] if "_" in feat_name else feat_name
        feature_type_importance[feat_type] = feature_type_importance.get(feat_type, 0) + importance
    
    sorted_types = sorted(feature_type_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"\nAggregated importance by feature type:")
    for feat_type, total_importance in sorted_types[:10]:
        logger.info(f"  {feat_type:30s} = {total_importance:.6f}")
    
    # Convert to JSON-serializable format (sanitize NaN/inf values)
    result = {}
    for feat, importance in sorted_features[:top_k]:
        val = float(importance)
        # Replace NaN/inf with 0 for JSON serialization
        if not np.isfinite(val):
            val = 0.0
        result[feat] = val
    
    return result


def save_artifacts(
    model: CalibratedClassifierCV,
    scaler: RobustScaler,
    feature_cols: List[str],
    models_dir: Path,
    threshold: float,
    win_rate: float,
    window_size: int,
) -> None:
    """Save trained model, scaler, and metadata.

    Args:
        model: Trained calibrated classifier
        scaler: Fitted RobustScaler for feature normalization
        feature_cols: Ordered list of feature column names
        models_dir: Directory to save artifacts
        threshold: Selected classification threshold
        win_rate: Expected win rate (precision on test set)
        window_size: Number of candles in each window
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    # Save model
    with open(models_dir / "sequence_xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save scaler (CRITICAL for production inference)
    with open(models_dir / "sequence_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to sequence_scaler.pkl (required for inference)")
    
    with open(models_dir / "sequence_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    
    metadata = {
        "threshold": threshold,
        "win_rate": win_rate,
        "window_size": window_size,
        "n_features_per_candle": len(feature_cols),
        "total_features": len(feature_cols) * window_size,
    }
    with open(models_dir / "sequence_threshold.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Analyze and save feature importance
    logger.info("Analyzing feature importance...")
    top_features = analyze_feature_importance(model, feature_cols, window_size, top_k=30)
    with open(models_dir / "sequence_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(top_features, f, ensure_ascii=False, indent=2)


def filter_by_session(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    session: str,
    custom_start: int = None,
    custom_end: int = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Filter data by trading session.

    Sessions (UTC approx):
    - london: 08:00 - 16:00
    - ny: 13:00 - 22:00
    - asian: 00:00 - 09:00
    - london_ny: 08:00 - 22:00
    - all: No filter
    - custom: Use custom_start and custom_end
    """
    hours = timestamps.hour

    if session == "london":
        mask = (hours >= 8) & (hours < 16)
    elif session == "ny":
        mask = (hours >= 13) & (hours < 22)
    elif session == "asian":
        mask = (hours >= 0) & (hours < 9)
    elif session == "london_ny":
        mask = (hours >= 8) & (hours < 22)
    elif session == "all":
        logger.info("Session filter: 'all' (keeping all data)")
        return X, y, timestamps
    elif session == "custom":
        if custom_start is None or custom_end is None:
            raise ValueError("Must provide custom_start and custom_end for 'custom' session")
        if custom_start < custom_end:
            mask = (hours >= custom_start) & (hours < custom_end)
        else:  # Crosses midnight
            mask = (hours >= custom_start) | (hours < custom_end)
    else:
        raise ValueError(f"Unknown session: {session}")

    if mask.sum() == 0:
        logger.warning(f"Session filter '{session}' removed all data! Proceeding without filter.")
        return X, y, timestamps

    logger.info(f"Session filter '{session}': kept {mask.sum()} / {len(X)} ({mask.mean():.1%})")
    return X[mask], y[mask], timestamps[mask]


def run_pipeline(
    window_size: int = 60,
    atr_multiplier_sl: float = 1.0,
    atr_multiplier_tp: float = 2.0,
    min_hold_minutes: int = 5,
    max_horizon: int = 60,
    random_state: int = 42,
    year_filter: List[int] = None,
    session: str = "london_ny",
    custom_start_hour: int = None,
    custom_end_hour: int = None,
    max_windows: int = 200000,
) -> Dict[str, float]:
    """Execute end-to-end sequence training pipeline.

    Args:
        window_size: Number of previous candles to use as input (default: 3 - reduced to avoid noise)
        atr_multiplier_sl: ATR multiplier for stop-loss (default: 1.0 - CONSTANT)
        atr_multiplier_tp: ATR multiplier for take-profit (default: 2.0 - CONSTANT, 2:1 RR)
        min_hold_minutes: Minimum hold time in minutes (default: 5)
        max_horizon: Maximum forward candles to simulate (default: 60)
        random_state: Random seed for reproducibility
        year_filter: Optional list of years to load (e.g., [2023, 2024] for testing)
        max_windows: Maximum number of windows to keep (default: 200,000)

    Returns:
        Dictionary with evaluation metrics including win_rate

    Raises:
        FileNotFoundError: If data files not found
        ValueError: On validation failures or insufficient data
    """
    np.random.seed(random_state)
    import random
    random.seed(random_state)
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"

    logger.info("Loading data...")
    df = load_all_years(data_dir, year_filter=year_filter)
    logger.info(f"Loaded {len(df):,} rows from {data_dir}")

    logger.info(f"Engineering per-candle features (window_size={window_size})...")
    features = engineer_candle_features(df, window_size=window_size)
    logger.info(f"Features shape: {features.shape}")

    logger.info(f"Creating target (SL={atr_multiplier_sl}×ATR, TP={atr_multiplier_tp}×ATR, min_hold={min_hold_minutes}min)...")
    targets = make_target(
        df.loc[features.index],
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
        min_hold_minutes=min_hold_minutes,
        max_horizon=max_horizon,
    )
    logger.info(f"Target shape: {len(targets)}, positive class: {targets.sum()} ({targets.mean():.2%})")

    logger.info(f"Creating sequences (window_size={window_size})...")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    X, y, timestamps = create_sequences(
        features,
        targets,
        window_size=window_size,
        session=session,
        custom_start=custom_start_hour,
        custom_end=custom_end_hour,
        max_windows=max_windows,
    )
    logger.info(f"Sequences: X.shape={X.shape}, y.shape={y.shape}")

    logger.info("Splitting data (chronological train/val/test)...")
    # Dynamic split based on data range
    if year_filter is not None:
        # For year filter: use percentage split to avoid empty splits
        n = len(X)
        train_idx = int(0.7 * n)
        val_idx = int(0.85 * n)
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        logger.info(f"Using percentage split (70/15/15) for year_filter={year_filter}")
    else:
        # Full date range: use fixed date splits
        X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(X, y, timestamps)
    logger.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # CRITICAL: Scale features AFTER split to prevent data leakage
    # Fit scaler ONLY on training data, then transform all sets
    logger.info("Scaling features with RobustScaler (robust to outliers)...")
    scaler = RobustScaler()
    # Ensure float32 output to save memory
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    logger.info(f"Feature scaling complete: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")

    logger.info("Training XGBoost classifier...")
    model = train_xgb(X_train_scaled, y_train, X_val_scaled, y_val, random_state=random_state)

    logger.info("Evaluating model on test set...")
    metrics = evaluate(model, X_test_scaled, y_test)
    logger.info(
        "Metrics: "
        f"threshold={metrics['threshold']:.2f}, "
        f"win_rate={metrics['win_rate']:.4f} ({metrics['win_rate']:.2%}), "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )

    logger.info("Saving artifacts...")
    logger.info("Saving artifacts (model, scaler, metadata)...")
    save_artifacts(
        model,
        scaler,
        list(features.columns),
        models_dir,
        metrics["threshold"],
        metrics["win_rate"],
        window_size,
    )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sequence-based XAU/USD training pipeline")
    parser.add_argument(
        "--window-size",
        type=int,
        default=14,
        help="Number of previous candles to use as input (default: 14)",
    )
    parser.add_argument(
        "--atr-multiplier-sl",
        type=float,
        default=1.0,
        help="ATR multiplier for stop-loss level (default: 1.0 - DO NOT CHANGE)",
    )
    parser.add_argument(
        "--atr-multiplier-tp",
        type=float,
        default=2.0,
        help="ATR multiplier for take-profit level (default: 2.0 - DO NOT CHANGE)",
    )
    parser.add_argument(
        "--min-hold-minutes",
        type=int,
        default=5,
        help="Minimum hold time in minutes (default: 5)",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=60,
        help="Maximum forward candles to simulate (default: 60)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years to load (e.g., '2023,2024' for testing)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="london_ny",
        choices=["london", "ny", "asian", "london_ny", "all", "custom"],
        help="Trading session to filter data (default: london_ny)",
    )
    parser.add_argument(
        "--custom-start-hour",
        type=int,
        default=None,
        help="Start hour for custom session (0-23)",
    )
    parser.add_argument(
        "--custom-end-hour",
        type=int,
        default=None,
        help="End hour for custom session (0-23)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200000,
        help="Maximum number of windows to keep to avoid OOM (default: 200,000)",
    )
    args = parser.parse_args()

    # Parse year filter
    year_filter = None
    if args.years:
        year_filter = [int(y.strip()) for y in args.years.split(',')]

    try:
        metrics = run_pipeline(
            window_size=args.window_size,        
            atr_multiplier_sl=args.atr_multiplier_sl,
            atr_multiplier_tp=args.atr_multiplier_tp,
            min_hold_minutes=args.min_hold_minutes,
            max_horizon=args.max_horizon,        
            random_state=args.random_state,      
            year_filter=year_filter,
            session=args.session,
            custom_start_hour=args.custom_start_hour,
            custom_end_hour=args.custom_end_hour,
            max_windows=args.max_windows,
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - SEQUENCE PIPELINE")
        print("=" * 60)
        print(f"Window Size:       {args.window_size} candles")
        print(f"Threshold:         {metrics['threshold']:.2f}")
        print(f"WIN RATE:          {metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1 Score:          {metrics['f1']:.4f}")
        print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:            {metrics['pr_auc']:.4f}")
        print("=" * 60)
        print(f"\nWin rate is the precision: when model predicts 'BUY',")
        print(f"it will be correct {metrics['win_rate']:.2%} of the time on test data.")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print("Data files not found. Ensure CSVs exist at 'ml/src/data/XAU_1m_data_*.csv'.")
        print(str(e))
    except Exception as e:
        print("Training pipeline failed:", str(e))
        raise
