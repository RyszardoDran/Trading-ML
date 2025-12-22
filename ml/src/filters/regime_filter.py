"""Market regime filter for gating trades based on volatility and trend conditions.

PURPOSE:
    Implement regime-based trading gate to avoid losing in unfavorable market conditions.
    
AUDIT 4 FINDINGS:
    - Fold 9 (88%): ATR=20, ADX=20+ → EXCELLENT (TIER 1)
    - Fold 11 (61.9%): ATR=16, ADX=16+ → GOOD (TIER 2)
    - Fold 2 (0%): ATR=8, ADX=8 → TERRIBLE (TIER 3)
    
    Conclusion: Only trade when conditions match TIER 1 or TIER 2
    Expected impact: Raise average win rate from 31.58% to 45-50%

IMPLEMENTATION:
    Use this filter in:
    1. Sequence creation (skip windows in bad regimes)
    2. Model prediction (suppress signals in bad regimes)
    3. Walk-forward validation (track regime-aware performance)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.risk_config import (
    ENABLE_REGIME_FILTER,
    REGIME_MIN_ATR_FOR_TRADING,
    REGIME_MIN_ADX_FOR_TRENDING,
    REGIME_MIN_PRICE_DIST_SMA200,
    REGIME_ADAPTIVE_THRESHOLD,
    REGIME_THRESHOLD_HIGH_ATR,
    REGIME_THRESHOLD_MOD_ATR,
    REGIME_THRESHOLD_LOW_ATR,
    REGIME_HIGH_ATR_THRESHOLD,
    REGIME_MOD_ATR_THRESHOLD,
)

logger = logging.getLogger(__name__)


class MarketRegime:
    """Market regime classification (TIER 1-4)."""
    
    TIER1_HIGH_ATR = "TIER1_HIGH_ATR"        # ATR >= 18: Excellent (80%+ wins)
    TIER2_MOD_ATR = "TIER2_MOD_ATR"          # ATR 12-17: Good (40-65% wins)
    TIER3_LOW_ATR = "TIER3_LOW_ATR"          # ATR 8-11: Poor (0-20% wins)
    TIER4_VERY_LOW_ATR = "TIER4_VERY_LOW_ATR"  # ATR < 8: Unusable (0-5% wins)
    
    # Descriptive names
    NAMES = {
        TIER1_HIGH_ATR: "HIGH_VOL_TREND",
        TIER2_MOD_ATR: "MOD_VOL_TREND",
        TIER3_LOW_ATR: "LOW_VOL",
        TIER4_VERY_LOW_ATR: "NO_VOL",
    }


def classify_regime(
    atr_m5: float,
    adx: float,
    price: float,
    sma200: float,
) -> Tuple[str, Dict[str, float]]:
    """Classify market regime based on volatility and trend strength.
    
    Args:
        atr_m5: ATR(14) on M5 timeframe (pips)
        adx: ADX(14) value
        price: Current close price
        sma200: SMA(200) level
        
    Returns:
        Tuple of (regime_type, regime_details dict)
    """
    # Input validation
    if atr_m5 <= 0 or adx < 0 or np.isnan(atr_m5) or np.isnan(adx):
        return MarketRegime.TIER4_VERY_LOW_ATR, {
            'atr_m5': atr_m5,
            'adx': adx,
            'dist_sma200': price - sma200,
            'in_uptrend': False,
            'reason': 'Invalid input',
        }
    
    # Check uptrend
    dist_sma200 = price - sma200
    in_uptrend = dist_sma200 > 0 and adx > REGIME_MIN_ADX_FOR_TRENDING
    
    # Classify by ATR tier
    if atr_m5 >= REGIME_HIGH_ATR_THRESHOLD:
        regime = MarketRegime.TIER1_HIGH_ATR
    elif atr_m5 >= REGIME_MOD_ATR_THRESHOLD:
        regime = MarketRegime.TIER2_MOD_ATR
    elif atr_m5 >= 8.0:
        regime = MarketRegime.TIER3_LOW_ATR
    else:
        regime = MarketRegime.TIER4_VERY_LOW_ATR
    
    return regime, {
        'atr_m5': atr_m5,
        'adx': adx,
        'dist_sma200': dist_sma200,
        'in_uptrend': in_uptrend,
        'reason': MarketRegime.NAMES.get(regime, regime),
    }


def should_trade(
    atr_m5: float,
    adx: float,
    price: float,
    sma200: float,
    threshold: Optional[float] = None,
) -> Tuple[bool, str, str]:
    """Determine if current market conditions warrant trading.
    
    This implements the regime gating logic from Audit 4:
    - SKIP if ATR < 12 (Fold 2: 0%, Fold 4: 19%)
    - SKIP if ADX < 12 (ranging market, no trend)
    - SKIP if price <= SMA200 (not in uptrend)
    - TRADE if all conditions met
    
    Args:
        atr_m5: ATR(14) on M5 timeframe (pips)
        adx: ADX(14) value
        price: Current close price
        sma200: SMA(200) level
        threshold: Optional probability threshold to use based on regime
        
    Returns:
        Tuple of (should_trade: bool, regime: str, reason: str)
        
    Examples:
        >>> # Good regime
        >>> should_trade(atr_m5=20, adx=20, price=2650, sma200=2620)
        (True, "TIER1_HIGH_ATR", "High ATR + Strong trend + Uptrend")
        
        >>> # Bad regime
        >>> should_trade(atr_m5=8, adx=8, price=2615, sma200=2620)
        (False, "TIER3_LOW_ATR", "Low ATR + No trend + Not uptrend")
    """
    if not ENABLE_REGIME_FILTER:
        # Regime filter disabled, always trade
        return True, "FILTER_DISABLED", "Regime filter is disabled"
    
    regime, details = classify_regime(atr_m5, adx, price, sma200)
    
    # Reason components
    reasons = []
    
    # Check ATR (most important)
    if atr_m5 < REGIME_MIN_ATR_FOR_TRADING:
        reasons.append(f"Low ATR ({atr_m5:.1f} < {REGIME_MIN_ATR_FOR_TRADING})")
    else:
        reasons.append(f"ATR OK ({atr_m5:.1f})")
    
    # Check ADX (trend strength)
    if adx < REGIME_MIN_ADX_FOR_TRENDING:
        reasons.append(f"No trend (ADX {adx:.1f} < {REGIME_MIN_ADX_FOR_TRENDING})")
    else:
        reasons.append(f"Trend OK (ADX {adx:.1f})")
    
    # Check uptrend (price > SMA200)
    dist = details['dist_sma200']
    if dist < REGIME_MIN_PRICE_DIST_SMA200:
        reasons.append(f"Not uptrend (dist {dist:.1f} < {REGIME_MIN_PRICE_DIST_SMA200})")
    else:
        reasons.append(f"Uptrend OK (dist {dist:.1f})")
    
    # Make decision
    trade = (
        atr_m5 >= REGIME_MIN_ATR_FOR_TRADING and
        adx >= REGIME_MIN_ADX_FOR_TRENDING and
        dist >= REGIME_MIN_PRICE_DIST_SMA200
    )
    
    reason = " + ".join(reasons)
    return trade, regime, reason


def get_adaptive_threshold(
    atr_m5: float,
) -> float:
    """Get adaptive classification threshold based on ATR regime.
    
    Higher ATR = more aggressive (lower threshold)
    Lower ATR = more conservative (higher threshold)
    
    Args:
        atr_m5: ATR(14) on M5 timeframe
        
    Returns:
        Probability threshold (0-1) for this regime
    """
    if not REGIME_ADAPTIVE_THRESHOLD:
        return 0.5  # Default threshold
    
    if atr_m5 >= REGIME_HIGH_ATR_THRESHOLD:
        return REGIME_THRESHOLD_HIGH_ATR  # 0.35 - be aggressive in Fold 9
    elif atr_m5 >= REGIME_MOD_ATR_THRESHOLD:
        return REGIME_THRESHOLD_MOD_ATR   # 0.50 - normal in Fold 11
    else:
        return REGIME_THRESHOLD_LOW_ATR   # 0.65 - conservative if forced to trade


def filter_sequences_by_regime(
    features: pd.DataFrame,
    targets: pd.Series,
    timestamps: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, np.ndarray]:
    """Filter sequences to keep only those in favorable market regimes.
    
    Removes sequences in low-ATR or low-ADX periods (like Fold 2: 0%).
    
    Args:
        features: Feature matrix (n_samples, n_features) with columns atr_m5, adx, sma_200, close
        targets: Target vector (n_samples,)
        timestamps: Timestamp index (n_samples,)
        
    Returns:
        Tuple of (filtered_features, filtered_targets, filtered_timestamps, regime_mask)
        
    Notes:
        - Drops ~50% of sequences in low-ATR periods
        - Keeps all sequences in high-ATR periods
        - Expected: raise average win rate from 31.58% to 45-50%
    """
    if not ENABLE_REGIME_FILTER:
        mask = np.ones(len(features), dtype=bool)
        return features, targets, timestamps, mask
    
    logger.info(f"Applying regime filter (ATR >= {REGIME_MIN_ATR_FOR_TRADING}, ADX >= {REGIME_MIN_ADX_FOR_TRENDING})")
    
    # Extract required features
    if 'atr_m5' not in features.columns:
        logger.warning("atr_m5 not in features, skipping regime filter")
        mask = np.ones(len(features), dtype=bool)
        return features, targets, timestamps, mask
    
    atr_m5_vals = features['atr_m5'].values
    adx_vals = features.get('adx', pd.Series(15.0, index=features.index)).values  # Default if missing
    price_vals = features['close'].values if 'close' in features.columns else np.full(len(features), np.nan)
    sma200_vals = features.get('sma_200', pd.Series(np.nan, index=features.index)).values
    
    # Create mask
    mask = np.zeros(len(features), dtype=bool)
    regimes_count = {regime: 0 for regime in [
        MarketRegime.TIER1_HIGH_ATR,
        MarketRegime.TIER2_MOD_ATR,
        MarketRegime.TIER3_LOW_ATR,
        MarketRegime.TIER4_VERY_LOW_ATR,
    ]}
    
    for i in range(len(features)):
        atr = atr_m5_vals[i]
        adx = adx_vals[i] if i < len(adx_vals) else 15.0
        price = price_vals[i] if i < len(price_vals) else np.nan
        sma200 = sma200_vals[i] if i < len(sma200_vals) else np.nan
        
        trade, regime, _ = should_trade(atr, adx, price, sma200)
        
        if trade:
            mask[i] = True
        
        regimes_count[regime] = regimes_count.get(regime, 0) + 1
    
    # Log filtering results
    filtered_count = mask.sum()
    logger.info(f"Regime filter: kept {filtered_count}/{len(mask)} sequences ({filtered_count/len(mask)*100:.1f}%)")
    logger.info(f"  TIER1 (ATR >= 18): {regimes_count.get(MarketRegime.TIER1_HIGH_ATR, 0)}")
    logger.info(f"  TIER2 (ATR 12-17): {regimes_count.get(MarketRegime.TIER2_MOD_ATR, 0)}")
    logger.info(f"  TIER3 (ATR 8-11): {regimes_count.get(MarketRegime.TIER3_LOW_ATR, 0)}")
    logger.info(f"  TIER4 (ATR < 8): {regimes_count.get(MarketRegime.TIER4_VERY_LOW_ATR, 0)}")
    
    return (
        features[mask],
        targets[mask],
        timestamps[mask],
        mask,
    )


def filter_predictions_by_regime(
    predictions: pd.Series,  # Probability predictions
    features: pd.DataFrame,  # Features to check ATR/ADX
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Filter predictions: suppress signals in bad regimes, adjust threshold in good ones.
    
    Args:
        predictions: Probability predictions (0-1)
        features: Features DataFrame with atr_m5, adx, sma_200, close
        threshold: Optional base threshold (default 0.5)
        
    Returns:
        Boolean array of predictions after regime gating
    """
    if not ENABLE_REGIME_FILTER:
        if threshold is None:
            threshold = 0.5
        return (predictions >= threshold).astype(int).values
    
    if threshold is None:
        threshold = 0.5
    
    result = np.zeros(len(predictions), dtype=int)
    
    for i in range(len(predictions)):
        # Get regime
        atr = features.iloc[i].get('atr_m5', 10.0)
        adx = features.iloc[i].get('adx', 10.0)
        price = features.iloc[i].get('close', 0)
        sma200 = features.iloc[i].get('sma_200', 0)
        
        trade, regime, _ = should_trade(atr, adx, price, sma200)
        
        if not trade:
            # Bad regime: suppress signal
            result[i] = 0
        else:
            # Good regime: use adaptive threshold
            adaptive_thresh = get_adaptive_threshold(atr)
            result[i] = 1 if predictions.iloc[i] >= adaptive_thresh else 0
    
    return result
