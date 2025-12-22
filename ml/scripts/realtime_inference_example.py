"""
Real-Time Inference Example: Using Regime Filter in Production

This shows how to integrate Regime Filter into real-time prediction pipeline.

ARCHITECTURE:
    1. Load trained model & scaler
    2. Get current market data
    3. Engineer features
    4. Generate prediction
    5. ← NEW: Gate prediction by regime
    6. Execute trade
"""

import logging
import pickle
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ml.src.filters.regime_filter import RegimeFilter
from ml.src.utils.risk_config import ENABLE_REGIME_FILTER

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of prediction with regime gating."""
    signal: int  # 0 or 1
    confidence: float  # Model confidence [0, 1]
    original_signal: int  # Signal before regime filtering
    regime: str  # Market regime (TIER 1-4)
    suppressed: bool  # Was signal suppressed by regime?
    indicators: Dict  # {atr, adx, sma200, close}


class RealtimePredictor:
    """Real-time prediction with regime-aware signal gating (Opcja B)."""
    
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize predictor.
        
        Args:
            model_path: Path to pickled trained model
            scaler_path: Path to pickled feature scaler
        """
        self.model = pickle.load(open(model_path, 'rb'))
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        
        # Initialize regime filter
        self.regime_filter = RegimeFilter() if ENABLE_REGIME_FILTER else None
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Loaded scaler from {scaler_path}")
        if self.regime_filter:
            logger.info("Regime filter: ENABLED (Opcja B)")
        else:
            logger.info("Regime filter: DISABLED")
    
    def predict(
        self,
        features: np.ndarray,
        market_indicators: Dict[str, float],
        confidence_threshold: float = 0.5
    ) -> PredictionResult:
        """Generate prediction with regime gating.
        
        Args:
            features: Engineered feature vector [n_features]
            market_indicators: Dict with:
                - atr: Current ATR value
                - adx: Current ADX value
                - close: Current close price
                - sma200: Current SMA200 value
            confidence_threshold: Threshold for generating signal
            
        Returns:
            PredictionResult with signal, regime info, etc.
        """
        
        # Validate inputs
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Generate prediction
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        confidence = prediction_proba[1]  # Probability of class 1 (trade)
        original_signal = 1 if confidence > confidence_threshold else 0
        
        # Initialize result
        signal = original_signal
        regime = "UNKNOWN"
        suppressed = False
        
        # ← REGIME FILTERING (Opcja B): Gate signal by market conditions
        if self.regime_filter is not None and original_signal == 1:
            
            # Filter signal
            filtered_signal = self.regime_filter.filter_predictions_by_regime(
                signals=np.array([original_signal]),
                confidence=np.array([confidence]),
                indicators=market_indicators
            )
            signal = filtered_signal[0]
            
            # Classify regime for logging
            regime = self.regime_filter.classify_regime(
                atr=market_indicators['atr']
            )
            
            # Track suppression
            suppressed = (signal != original_signal)
            
            if suppressed:
                logger.info(
                    f"Signal suppressed by regime filter. "
                    f"Regime: {regime}, "
                    f"Confidence: {confidence:.2%}, "
                    f"ATR: {market_indicators['atr']:.2f}, "
                    f"ADX: {market_indicators['adx']:.2f}"
                )
        
        return PredictionResult(
            signal=signal,
            confidence=float(confidence),
            original_signal=original_signal,
            regime=regime,
            suppressed=suppressed,
            indicators=market_indicators
        )
    
    def predict_batch(
        self,
        features: np.ndarray,
        market_indicators_list: list[Dict[str, float]],
        confidence_threshold: float = 0.5
    ) -> list[PredictionResult]:
        """Generate predictions for multiple timesteps.
        
        Args:
            features: Feature matrix [n_samples, n_features]
            market_indicators_list: List of indicator dicts for each sample
            confidence_threshold: Threshold for signal generation
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for i in range(len(features)):
            result = self.predict(
                features=features[i],
                market_indicators=market_indicators_list[i],
                confidence_threshold=confidence_threshold
            )
            results.append(result)
        
        return results


def example_real_time_trading():
    """Example real-time trading loop with regime filtering.
    
    This shows the complete flow from market data to trade execution.
    """
    
    # ← SETUP
    logger.info("Initializing real-time trading system...")
    
    predictor = RealtimePredictor(
        model_path='ml/outputs/models/model.pkl',
        scaler_path='ml/outputs/models/scaler.pkl'
    )
    
    # ← TRADING LOOP (pseudo-code - replace with real data source)
    try:
        iteration = 0
        while iteration < 100:  # Example: 100 iterations
            
            iteration += 1
            
            # 1️⃣ Get current market data
            # In real system, this comes from broker API
            current_price = 2000.0 + np.random.randn() * 5  # Placeholder
            current_atr = 15.0 + np.random.randn()  # Placeholder
            current_adx = 25.0 + np.random.randn()  # Placeholder
            current_sma200 = 1995.0  # Placeholder
            
            # 2️⃣ Engineer features
            # In real system, calculate from market data
            features = np.array([
                current_price,
                current_atr,
                current_adx,
                # ... other 21 features
            ] + [0.0] * 18)  # Placeholder for remaining features
            
            # 3️⃣ Generate prediction with regime filtering
            result = predictor.predict(
                features=features,
                market_indicators={
                    'atr': current_atr,
                    'adx': current_adx,
                    'close': current_price,
                    'sma200': current_sma200
                },
                confidence_threshold=0.5
            )
            
            # 4️⃣ Execute trade (only if signal = 1)
            if result.signal == 1:
                logger.info(
                    f"[{iteration}] EXECUTE TRADE (signal=1, "
                    f"confidence={result.confidence:.2%}, "
                    f"regime={result.regime})"
                )
                # place_order(...)  # Real trade execution
            else:
                if result.suppressed:
                    logger.info(
                        f"[{iteration}] SKIP TRADE (signal suppressed by regime, "
                        f"regime={result.regime})"
                    )
                else:
                    logger.debug(f"[{iteration}] No signal (confidence too low)")
            
            # 5️⃣ Sleep before next iteration
            # time.sleep(60)  # Real system: wait 1 minute
    
    except KeyboardInterrupt:
        logger.info("Trading loop stopped by user")
    
    logger.info("Trading system stopped")


def example_metrics_collection():
    """Example of collecting metrics for production monitoring.
    
    Track key metrics to ensure regime filter is working correctly.
    """
    
    predictor = RealtimePredictor(
        model_path='ml/outputs/models/model.pkl',
        scaler_path='ml/outputs/models/scaler.pkl'
    )
    
    # Simulate 1000 predictions
    n_predictions = 1000
    results = []
    
    for _ in range(n_predictions):
        features = np.random.randn(24)  # 24 features
        
        # Simulate different regime conditions
        atr = 8 + np.random.exponential(5)  # ATR typically 8-20
        adx = 15 + np.random.randn(10)  # ADX typically 15-35
        
        result = predictor.predict(
            features=features,
            market_indicators={
                'atr': atr,
                'adx': max(adx, 0),
                'close': 2000.0,
                'sma200': 1995.0
            }
        )
        results.append(result)
    
    # ← COLLECT METRICS
    signals_generated = sum(1 for r in results if r.original_signal == 1)
    signals_suppressed = sum(1 for r in results if r.suppressed)
    signals_executed = sum(1 for r in results if r.signal == 1)
    
    tier1_count = sum(1 for r in results if r.regime == "TIER1_HIGH_ATR")
    tier2_count = sum(1 for r in results if r.regime == "TIER2_MOD_ATR")
    tier3_count = sum(1 for r in results if r.regime == "TIER3_LOW_ATR")
    tier4_count = sum(1 for r in results if r.regime == "TIER4_VERY_LOW_ATR")
    
    # ← PRINT PRODUCTION METRICS
    logger.info("\n" + "=" * 70)
    logger.info("PRODUCTION MONITORING METRICS (1000 predictions)")
    logger.info("=" * 70)
    
    logger.info(f"\nSignal Generation:")
    logger.info(f"  Original signals: {signals_generated}")
    logger.info(f"  Suppressed:       {signals_suppressed}")
    logger.info(f"  Executed:         {signals_executed}")
    logger.info(f"  Suppression rate: {100*signals_suppressed/max(signals_generated,1):.1f}%")
    
    logger.info(f"\nRegime Distribution:")
    logger.info(f"  TIER 1 (ATR ≥ 18):    {tier1_count:4d} ({100*tier1_count/n_predictions:.1f}%)")
    logger.info(f"  TIER 2 (ATR 12-17):   {tier2_count:4d} ({100*tier2_count/n_predictions:.1f}%)")
    logger.info(f"  TIER 3 (ATR 8-11):    {tier3_count:4d} ({100*tier3_count/n_predictions:.1f}%)")
    logger.info(f"  TIER 4 (ATR < 8):     {tier4_count:4d} ({100*tier4_count/n_predictions:.1f}%)")
    
    logger.info(f"\nAverage Confidence by Regime:")
    for regime in ["TIER1_HIGH_ATR", "TIER2_MOD_ATR", "TIER3_LOW_ATR", "TIER4_VERY_LOW_ATR"]:
        regime_results = [r for r in results if r.regime == regime]
        if regime_results:
            avg_conf = np.mean([r.confidence for r in regime_results])
            logger.info(f"  {regime:20s}: {avg_conf:.2%}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Production metrics collected. Monitor regularly.")
    logger.info("=" * 70)


# ← ALERTS: Conditions to monitor in production
PRODUCTION_ALERTS = {
    "suppression_rate_too_high": {
        "threshold": 0.50,  # Alert if > 50% of signals suppressed
        "message": "Unusual market conditions - many signals being suppressed",
        "action": "Review regime filter thresholds, consider temporary disable"
    },
    "suppression_rate_too_low": {
        "threshold": 0.20,  # Alert if < 20% of signals suppressed
        "message": "Regime filter may not be working - too few signals suppressed",
        "action": "Check regime filter implementation, verify indicators calculated"
    },
    "high_tier1_percentage": {
        "threshold": 0.60,  # Alert if > 60% in TIER 1
        "message": "Highly favorable market - all signals likely executing",
        "action": "Expect high win rates, monitor for market shift"
    },
    "low_tier1_percentage": {
        "threshold": 0.10,  # Alert if < 10% in TIER 1
        "message": "Unfavorable market - most signals being suppressed",
        "action": "Market in poor regime, expect low win rates"
    }
}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run examples
    logger.info("=" * 70)
    logger.info("REAL-TIME INFERENCE WITH REGIME FILTER")
    logger.info("=" * 70)
    
    logger.info("\n[Example 1] Metrics Collection")
    logger.info("-" * 70)
    example_metrics_collection()
    
    logger.info("\n[Example 2] Real-time Trading Loop")
    logger.info("-" * 70)
    logger.info("(Run example_real_time_trading() for live demo)")
    logger.info("See ml/PRODUCTION_INTEGRATION_GUIDE.md for integration steps")
