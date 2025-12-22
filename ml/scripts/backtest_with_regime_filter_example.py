"""
Example: Integrating Regime Filter into Backtest Strategy

This shows EXACTLY where and how to add regime filter to your
backtest_strategy.py script using Opcja B (Prediction Gating).

KEY CONCEPT:
    Training: Use 100% of data (no filtering)
    Inference: Gate predictions when market regime is poor
    Expected: WIN RATE improvement from 31.58% → 45-50%
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict

# ← NEW IMPORT: Regime filter
from ml.src.filters.regime_filter import RegimeFilter
from ml.src.utils.risk_config import (
    ENABLE_REGIME_FILTER,
    REGIME_MIN_ATR_FOR_TRADING,
    REGIME_MIN_ADX_FOR_TRENDING,
)

logger = logging.getLogger(__name__)


def calculate_technical_indicators(
    prices: pd.Series,
    atr_period: int = 14,
    adx_period: int = 14,
    sma_period: int = 200
) -> Dict[str, pd.Series]:
    """Calculate technical indicators needed for regime filter.
    
    Args:
        prices: Series of close prices
        atr_period: ATR lookback period
        adx_period: ADX lookback period
        sma_period: SMA lookback period
        
    Returns:
        Dict with {atr, adx, sma200} series aligned with prices
        
    NOTE: Implementation shown assumes you have a TA library
          Replace with your existing indicator calculation
    """
    try:
        import talib
    except ImportError:
        logger.warning("talib not available, using simple fallback")
        # Fallback: simple ATR estimation
        high_low = prices  # Placeholder
        return {
            'atr': pd.Series(np.full(len(prices), 1.0), index=prices.index),
            'adx': pd.Series(np.full(len(prices), 20.0), index=prices.index),
            'sma200': prices.rolling(sma_period).mean()
        }
    
    # Calculate indicators (your existing code)
    atr = talib.ATR(high=prices, low=prices, close=prices, timeperiod=atr_period)
    adx = talib.ADX(high=prices, low=prices, close=prices, timeperiod=adx_period)
    sma200 = talib.SMA(prices, timeperiod=sma_period)
    
    return {
        'atr': atr,
        'adx': adx,
        'sma200': sma200
    }


def backtest_with_regime_filter(
    prices: np.ndarray,
    signals: np.ndarray,
    confidence_scores: np.ndarray,
    indicators: Dict[str, np.ndarray],
    initial_capital: float = 100000,
    commission: float = 0.001,
    spread: float = 0.0002,
) -> Tuple[np.ndarray, Dict]:
    """Backtest strategy WITH regime filter (Opcja B).
    
    Args:
        prices: Array of prices [N]
        signals: Array of trading signals (0 or 1) [N]
        confidence_scores: Model confidence scores [N]
        indicators: Dict with {atr, adx, sma200} arrays [N]
        initial_capital: Starting capital
        commission: Transaction cost
        spread: Bid-ask spread
        
    Returns:
        equity: Equity curve [N]
        metrics: Performance metrics dict
    """
    
    # ← NEW: Initialize regime filter
    regime_filter = RegimeFilter() if ENABLE_REGIME_FILTER else None
    
    # Initialize equity curve
    equity = np.zeros(len(prices))
    equity[0] = initial_capital
    
    trade_pnl = []
    suppressed_count = 0  # Track how many signals were gated
    
    # ← MAIN LOOP: Process each candle
    for i in range(1, len(prices)):
        
        # Original signal from model
        original_signal = signals[i - 1]
        confidence = confidence_scores[i - 1]
        
        # ← NEW: Gate signal by market regime (Opcja B)
        signal = original_signal
        
        if regime_filter is not None and original_signal == 1:
            # Filter signal using market regime
            filtered_signal = regime_filter.filter_predictions_by_regime(
                signals=np.array([original_signal]),
                confidence=np.array([confidence]),
                indicators={
                    'atr': indicators['atr'][i - 1],
                    'adx': indicators['adx'][i - 1],
                    'close': prices[i - 1],
                    'sma200': indicators['sma200'][i - 1]
                }
            )
            signal = filtered_signal[0]
            
            # Track suppression
            if signal != original_signal:
                suppressed_count += 1
                logger.debug(
                    f"Candle {i}: Signal suppressed (regime gating). "
                    f"ATR={indicators['atr'][i-1]:.2f}, "
                    f"ADX={indicators['adx'][i-1]:.2f}, "
                    f"Confidence={confidence:.2%}"
                )
        
        # ← EXECUTE TRADE: Only if signal = 1 (after regime filtering)
        if signal == 1:
            entry_price = prices[i - 1] * (1 + spread / 2)
            exit_price = prices[i] * (1 - spread / 2)
            entry_cost = entry_price * (1 + commission)
            exit_value = exit_price * (1 - commission)
            
            pnl_pct = (exit_value - entry_cost) / entry_cost
            trade_pnl.append(pnl_pct)
            
            equity[i] = equity[i - 1] * (1 + pnl_pct)
        else:
            # No trade: carry forward equity
            equity[i] = equity[i - 1]
    
    # ← CALCULATE METRICS
    total_return = (equity[-1] / initial_capital - 1) * 100
    winning_trades = sum(1 for pnl in trade_pnl if pnl > 0)
    win_rate = (winning_trades / len(trade_pnl) * 100) if trade_pnl else 0
    
    metrics = {
        "total_return_pct": float(total_return),
        "win_rate_pct": float(win_rate),
        "num_trades": int(len(trade_pnl)),
        "num_signals_generated": int(np.sum(signals)),
        "num_signals_suppressed": int(suppressed_count),
        "suppression_rate_pct": float(100 * suppressed_count / max(np.sum(signals), 1)),
        "final_equity": float(equity[-1]),
    }
    
    return equity, metrics


def backtest_without_regime_filter(
    prices: np.ndarray,
    signals: np.ndarray,
    confidence_scores: np.ndarray,
    initial_capital: float = 100000,
    commission: float = 0.001,
    spread: float = 0.0002,
) -> Tuple[np.ndarray, Dict]:
    """Backtest strategy WITHOUT regime filter (baseline).
    
    Args:
        prices: Array of prices [N]
        signals: Array of trading signals (0 or 1) [N]
        confidence_scores: Model confidence scores [N] (not used for filtering)
        initial_capital: Starting capital
        commission: Transaction cost
        spread: Bid-ask spread
        
    Returns:
        equity: Equity curve [N]
        metrics: Performance metrics dict
    """
    
    equity = np.zeros(len(prices))
    equity[0] = initial_capital
    
    trade_pnl = []
    
    for i in range(1, len(prices)):
        signal = signals[i - 1]
        
        if signal == 1:
            entry_price = prices[i - 1] * (1 + spread / 2)
            exit_price = prices[i] * (1 - spread / 2)
            entry_cost = entry_price * (1 + commission)
            exit_value = exit_price * (1 - commission)
            
            pnl_pct = (exit_value - entry_cost) / entry_cost
            trade_pnl.append(pnl_pct)
            
            equity[i] = equity[i - 1] * (1 + pnl_pct)
        else:
            equity[i] = equity[i - 1]
    
    total_return = (equity[-1] / initial_capital - 1) * 100
    winning_trades = sum(1 for pnl in trade_pnl if pnl > 0)
    win_rate = (winning_trades / len(trade_pnl) * 100) if trade_pnl else 0
    
    metrics = {
        "total_return_pct": float(total_return),
        "win_rate_pct": float(win_rate),
        "num_trades": int(len(trade_pnl)),
        "final_equity": float(equity[-1]),
    }
    
    return equity, metrics


def compare_backtests(prices, signals, confidence_scores, indicators):
    """Run backtests with and without regime filter.
    
    This shows the performance impact of Opcja B.
    """
    
    logger.info("=" * 70)
    logger.info("REGIME FILTER COMPARISON TEST")
    logger.info("=" * 70)
    
    # Backtest WITHOUT regime filter (baseline)
    logger.info("\n[1] Running BASELINE backtest (WITHOUT regime filter)...")
    equity_baseline, metrics_baseline = backtest_without_regime_filter(
        prices=prices,
        signals=signals,
        confidence_scores=confidence_scores
    )
    
    logger.info(f"    Trades executed: {metrics_baseline['num_trades']}")
    logger.info(f"    Win rate: {metrics_baseline['win_rate_pct']:.2f}%")
    logger.info(f"    Total return: {metrics_baseline['total_return_pct']:.2f}%")
    logger.info(f"    Final equity: ${metrics_baseline['final_equity']:,.0f}")
    
    # Backtest WITH regime filter (Opcja B)
    logger.info("\n[2] Running REGIME FILTER backtest (WITH regime filter)...")
    equity_filtered, metrics_filtered = backtest_with_regime_filter(
        prices=prices,
        signals=signals,
        confidence_scores=confidence_scores,
        indicators=indicators
    )
    
    logger.info(f"    Signals generated: {metrics_filtered['num_signals_generated']}")
    logger.info(f"    Signals suppressed: {metrics_filtered['num_signals_suppressed']}")
    logger.info(f"    Suppression rate: {metrics_filtered['suppression_rate_pct']:.1f}%")
    logger.info(f"    Trades executed: {metrics_filtered['num_trades']}")
    logger.info(f"    Win rate: {metrics_filtered['win_rate_pct']:.2f}%")
    logger.info(f"    Total return: {metrics_filtered['total_return_pct']:.2f}%")
    logger.info(f"    Final equity: ${metrics_filtered['final_equity']:,.0f}")
    
    # Compare
    logger.info("\n[3] COMPARISON RESULTS:")
    logger.info("=" * 70)
    
    win_rate_improvement = metrics_filtered['win_rate_pct'] - metrics_baseline['win_rate_pct']
    return_improvement = metrics_filtered['total_return_pct'] - metrics_baseline['total_return_pct']
    
    logger.info(f"    Win rate: {metrics_baseline['win_rate_pct']:.2f}% → {metrics_filtered['win_rate_pct']:.2f}%")
    logger.info(f"              {'✅ IMPROVED' if win_rate_improvement > 0 else '❌ DECREASED'} by {abs(win_rate_improvement):+.2f} pp")
    
    logger.info(f"\n    Return:   {metrics_baseline['total_return_pct']:.2f}% → {metrics_filtered['total_return_pct']:.2f}%")
    logger.info(f"              {'✅ IMPROVED' if return_improvement > 0 else '❌ DECREASED'} by {abs(return_improvement):+.2f}%")
    
    logger.info(f"\n    Trades:   {metrics_baseline['num_trades']} → {metrics_filtered['num_trades']}")
    logger.info(f"              {metrics_filtered['num_trades'] - metrics_baseline['num_trades']:+d} trades (regime filtering effect)")
    
    logger.info("=" * 70)
    
    # Summary
    if win_rate_improvement >= 13.4:
        logger.info(f"✅ TARGET ACHIEVED: Win rate improved by {win_rate_improvement:.2f} pp")
        logger.info(f"   (Target: +13.4 to +18.4 pp)")
    else:
        logger.info(f"⚠️  Below target: Win rate improved by {win_rate_improvement:.2f} pp")
        logger.info(f"   (Target: +13.4 to +18.4 pp)")
    
    return {
        'baseline': metrics_baseline,
        'filtered': metrics_filtered,
        'improvement': {
            'win_rate_pp': win_rate_improvement,
            'return_pp': return_improvement
        }
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    logger.info("This is a reference implementation.")
    logger.info("See ml/PRODUCTION_INTEGRATION_GUIDE.md for integration steps.")
