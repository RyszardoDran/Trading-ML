#!/usr/bin/env python3
"""Run backtest on trained model predictions.

Usage:
    # Backtest with default settings
    python run_backtest.py
    
    # Backtest specific years with custom capital
    python run_backtest.py --years 2024 --capital 50000
    
    # Walk-forward analysis
    python run_backtest.py --walk-forward --train-days 180 --test-days 30
    
    # Custom position sizing
    python run_backtest.py --position-sizing risk_based --risk-per-trade 0.02

Output:
    - Detailed performance metrics
    - Equity curve plot
    - Trade log CSV
    - Summary statistics
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
_script_dir = Path(__file__).parent
_ml_dir = _script_dir.parent
sys.path.insert(0, str(_ml_dir.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.src.backtesting import BacktestEngine
from ml.src.backtesting.config import BacktestConfig
from ml.src.backtesting.position_sizer import PositionSizingMethod
from ml.src.data_loading import load_all_years
from ml.src.scripts.predict_sequence import load_model_artifacts
from ml.src.features.engineer_m5 import aggregate_to_m5, engineer_m5_candle_features
from ml.src.filters import should_trade
from ml.src.utils import PipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Backtest trading strategy with trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data selection
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=None,
        help='Years to backtest (e.g., 2023 2024). If None, use all available data.'
    )
    
    # Capital and position sizing
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital in USD'
    )
    
    parser.add_argument(
        '--position-sizing',
        type=str,
        default='fixed',
        choices=['fixed', 'risk_based', 'kelly', 'volatility_based'],
        help='Position sizing method'
    )
    
    parser.add_argument(
        '--position-size',
        type=float,
        default=0.01,
        help='Fixed position size in lots (for fixed method)'
    )
    
    parser.add_argument(
        '--risk-per-trade',
        type=float,
        default=0.01,
        help='Risk percentage per trade (for risk_based method)'
    )
    
    # Transaction costs
    parser.add_argument(
        '--spread',
        type=float,
        default=0.5,
        help='Spread in pips (XAU/USD typical: 0.5)'
    )
    
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.3,
        help='Slippage in pips'
    )
    
    parser.add_argument(
        '--commission',
        type=float,
        default=0.0,
        help='Commission per trade in USD'
    )
    
    # Trading rules
    parser.add_argument(
        '--min-probability',
        type=float,
        default=0.5,
        help='Minimum model probability to take trade'
    )
    
    parser.add_argument(
        '--threshold-override',
        type=float,
        default=None,
        help='Override model threshold (e.g., 0.3 to allow more trades)'
    )
    
    parser.add_argument(
        '--max-trades-per-day',
        type=int,
        default=None,
        help='Maximum trades per day (None = unlimited)'
    )
    
    # Risk management
    parser.add_argument(
        '--max-drawdown',
        type=float,
        default=0.20,
        help='Maximum drawdown percentage (e.g., 0.20 = 20%%)'
    )
    
    # Walk-forward analysis
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Enable walk-forward analysis'
    )
    
    parser.add_argument(
        '--train-days',
        type=int,
        default=180,
        help='Training window size in days (for walk-forward)'
    )
    
    parser.add_argument(
        '--test-days',
        type=int,
        default=30,
        help='Testing window size in days (for walk-forward)'
    )
    
    parser.add_argument(
        '--step-days',
        type=int,
        default=30,
        help='Step size in days (for walk-forward)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('ml/outputs/backtests'),
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save-plot',
        action='store_true',
        help='Save equity curve plot'
    )
    
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=None,
        help='Directory with model artifacts (default: ml/src/models)'
    )
    
    return parser.parse_args()


def load_data_and_predict(
    data_dir: Path,
    models_dir: Path,
    year_filter: Optional[list[int]] = None,
    threshold_override: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and generate predictions using optimized M5 approach.
    
    Optimization: Aggregate M1→M5 once at start, then slide window over M5.
    This is O(n) instead of O(n × window_size).
    
    Args:
        data_dir: Directory with OHLCV data
        models_dir: Directory with model artifacts
        year_filter: Optional list of years to load
        
    Returns:
        Tuple of (predictions DataFrame, prices DataFrame)
    """
    logger.info("=" * 80)
    logger.info("OPTIMIZED M5 BACKTEST - LOAD DATA AND PREDICT")
    logger.info("=" * 80)
    
    # Load M1 price data
    logger.info(f"Loading M1 price data from: {data_dir}")
    if year_filter:
        logger.info(f"  Years: {year_filter}")
    
    m1_prices = load_all_years(data_dir, year_filter=year_filter)
    logger.info(f"Loaded {len(m1_prices):,} M1 candles")
    logger.info(f"Period: {m1_prices.index[0]} to {m1_prices.index[-1]}")
    
    # Load model artifacts
    logger.info(f"\nLoading model from: {models_dir}")
    artifacts = load_model_artifacts(models_dir)
    model = artifacts['model']
    scaler = artifacts['scaler']
    window_size_m5 = artifacts['window_size']
    threshold = artifacts['threshold']
    
    logger.info(f"Model window size: {window_size_m5} M5 candles")
    logger.info(f"Decision threshold: {threshold:.4f}")
    
    if threshold_override is not None:
        threshold = threshold_override
        logger.info(f"Threshold overridden to {threshold:.4f}")
    
    # OPTIMIZATION: Aggregate M1→M5 ONCE at the start
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATING M1 → M5 (ONE-TIME OPERATION)")
    logger.info(f"{'='*80}")
    m5_data = aggregate_to_m5(m1_prices)
    logger.info(f"✅ Aggregated to {len(m5_data):,} M5 candles ({len(m1_prices)/len(m5_data):.1f}x compression)")
    
    # OPTIMIZATION: Engineer M5 features ONCE at the start
    logger.info(f"\n{'='*80}")
    logger.info("ENGINEERING M5 FEATURES (ONE-TIME OPERATION)")
    logger.info(f"{'='*80}")
    m5_features = engineer_m5_candle_features(m5_data)
    logger.info(f"✅ Engineered {len(m5_features):,} M5 rows × {m5_features.shape[1]} features")
    
    # Verify alignment
    if len(m5_features) != len(m5_data):
        raise ValueError(f"M5 features/data length mismatch: {len(m5_features)} vs {len(m5_data)}")
    
    # Calculate minimum M5 candles needed (window + warmup for indicators)
    min_m5_candles = window_size_m5 + 200  # 60 + 200 for SMA200
    logger.info(f"\nMinimum M5 candles: {min_m5_candles} ({window_size_m5} window + 200 warmup)")
    
    if len(m5_features) < min_m5_candles:
        raise ValueError(f"Insufficient M5 data: {len(m5_features)} < {min_m5_candles}")
    
    # Generate predictions by sliding window over M5 sequences
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING PREDICTIONS (SLIDING WINDOW OVER M5)")
    logger.info(f"{'='*80}")
    
    predictions_list = []
    
    # Slide window over M5 data
    suppressed_count = 0
    for i in range(min_m5_candles, len(m5_features)):
        # Extract window of M5 features
        feature_window = m5_features.iloc[i-window_size_m5:i]
        
        # Flatten to 1D (model expects flattened sequence: 60 M5 × 15 features = 900)
        X = feature_window.values.flatten().reshape(1, -1)
        
        # Scale features
        X = scaler.transform(X)
        
        # Predict
        proba = model.predict_proba(X)[0, 1]
        prediction = 1 if proba >= threshold else 0
        
        # Apply regime filter if prediction is BUY
        regime = None
        reason = None
        if prediction == 1:
            # Get features from the last M5 candle
            last_features = feature_window.iloc[-1]
            atr_m5 = last_features['atr']
            adx = last_features['adx']
            entry_price = m5_data.loc[m5_timestamp, 'Close']
            sma200 = last_features['sma200']
            
            allowed, regime, reason = should_trade(atr_m5, adx, entry_price, sma200)
            if not allowed:
                prediction = 0
                suppressed_count += 1
        
        # Map back to M1 timestamp (last M1 candle of this M5 candle)
        m5_timestamp = m5_data.index[i]
        
        # Find corresponding M1 timestamp
        m1_timestamp = m1_prices[m1_prices.index <= m5_timestamp].index[-1]
        
        predictions_list.append({
            'timestamp': m1_timestamp,
            'probability': proba,
            'prediction': prediction,
            'threshold': threshold,
            'regime': regime,
            'reason': reason,
        })
        
        # Progress logging
        if (i - min_m5_candles + 1) % 10000 == 0:
            progress = (i - min_m5_candles + 1) / (len(m5_features) - min_m5_candles) * 100
            logger.info(f"  Progress: {progress:.1f}% ({i - min_m5_candles + 1:,} / {len(m5_features) - min_m5_candles:,})")
    
    predictions_df = pd.DataFrame(predictions_list)
    predictions_df.set_index('timestamp', inplace=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("PREDICTION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total predictions: {len(predictions_df):,}")
    logger.info(f"  BUY signals: {(predictions_df['prediction']==1).sum():,} ({(predictions_df['prediction']==1).sum()/len(predictions_df)*100:.2f}%)")
    logger.info(f"  HOLD signals: {(predictions_df['prediction']==0).sum():,} ({(predictions_df['prediction']==0).sum()/len(predictions_df)*100:.2f}%)")
    logger.info(f"  Avg probability: {predictions_df['probability'].mean():.4f}")
    logger.info(f"  Min probability: {predictions_df['probability'].min():.4f}")
    logger.info(f"  Max probability: {predictions_df['probability'].max():.4f}")
    
    # Regime filter summary
    regime_signals = predictions_df[predictions_df['regime'].notna()]
    if len(regime_signals) > 0:
        logger.info(f"\nRegime filter applied to {len(regime_signals):,} signals:")
        logger.info(f"  Allowed: {(regime_signals['prediction']==1).sum():,} ({(regime_signals['prediction']==1).sum()/len(regime_signals)*100:.2f}%)")
        logger.info(f"  Suppressed: {(regime_signals['prediction']==0).sum():,} ({(regime_signals['prediction']==0).sum()/len(regime_signals)*100:.2f}%)")
        reason_counts = regime_signals['reason'].value_counts()
        for reason, count in reason_counts.items():
            logger.info(f"    {reason}: {count:,} ({count/len(regime_signals)*100:.2f}%)")
    
    return predictions_df, m1_prices


def plot_equity_curve(
    equity: pd.Series,
    trades: pd.DataFrame,
    config: BacktestConfig,
    save_path: Optional[Path] = None,
) -> None:
    """Plot equity curve with trade markers.
    
    Args:
        equity: Equity curve time series
        trades: Trades DataFrame
        config: Backtest configuration
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot equity curve
    ax1.plot(equity.index, equity.values, linewidth=2, label='Equity')
    ax1.axhline(
        config.initial_capital,
        color='gray',
        linestyle='--',
        alpha=0.5,
        label='Initial Capital'
    )
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    ax2.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color='red',
        alpha=0.3,
        label='Drawdown'
    )
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Equity curve plot saved to: {save_path}")
    
    plt.show()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    try:
        # Setup paths
        config_obj = PipelineConfig()
        data_dir = config_obj.data_dir
        
        if args.models_dir is None:
            models_dir = config_obj.outputs_models_dir
        else:
            models_dir = args.models_dir
        
        # Load data and generate predictions
        predictions, prices = load_data_and_predict(
            data_dir=data_dir,
            models_dir=models_dir,
            year_filter=args.years,
            threshold_override=args.threshold_override,
        )
        
        # Create backtest configuration
        logger.info("\n" + "=" * 80)
        logger.info("CONFIGURING BACKTEST")
        logger.info("=" * 80)
        
        position_sizing = PositionSizingMethod(args.position_sizing)
        
        config = BacktestConfig(
            initial_capital=args.capital,
            position_sizing=position_sizing,
            fixed_position_size=args.position_size,
            risk_per_trade=args.risk_per_trade,
            spread_pips=args.spread,
            slippage_pips=args.slippage,
            commission=args.commission,
            min_probability=args.min_probability,
            max_trades_per_day=args.max_trades_per_day,
            max_drawdown_pct=args.max_drawdown,
            walk_forward_enabled=args.walk_forward,
            walk_forward_train_days=args.train_days,
            walk_forward_test_days=args.test_days,
            walk_forward_step_days=args.step_days,
            output_dir=args.output_dir,
            save_trades=True,
            save_equity_curve=True,
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  Initial capital: ${config.initial_capital:,.2f}")
        logger.info(f"  Position sizing: {config.position_sizing.value}")
        logger.info(f"  Transaction costs: ${config.total_transaction_cost():.2f} per trade")
        logger.info(f"  Min probability: {config.min_probability:.2f}")
        logger.info(f"  Walk-forward: {config.walk_forward_enabled}")
        
        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run(predictions, prices)
        
        # Plot results
        if args.save_plot or True:  # Always show plot
            plot_path = args.output_dir / f"equity_curve_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_equity_curve(
                results['equity_curve'],
                results['trades'],
                config,
                save_path=plot_path if args.save_plot else None,
            )
        
        logger.info("\n✅ Backtest completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
