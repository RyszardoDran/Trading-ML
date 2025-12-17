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
from ml.src.scripts.predict_sequence import load_model_artifacts, predict
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and generate predictions.
    
    Args:
        data_dir: Directory with OHLCV data
        models_dir: Directory with model artifacts
        year_filter: Optional list of years to load
        
    Returns:
        Tuple of (predictions DataFrame, prices DataFrame)
    """
    logger.info("=" * 80)
    logger.info("LOADING DATA AND GENERATING PREDICTIONS")
    logger.info("=" * 80)
    
    # Load price data
    logger.info(f"Loading price data from: {data_dir}")
    if year_filter:
        logger.info(f"  Years: {year_filter}")
    
    prices = load_all_years(data_dir, year_filter=year_filter)
    logger.info(f"Loaded {len(prices):,} candles")
    logger.info(f"Period: {prices.index[0]} to {prices.index[-1]}")
    
    # Load model
    logger.info(f"\nLoading model from: {models_dir}")
    artifacts = load_model_artifacts(models_dir)
    window_size = artifacts['window_size']
    logger.info(f"Model window size: {window_size}")
    
    # Generate predictions for all data
    logger.info(f"\nGenerating predictions...")
    
    predictions_list = []
    
    # Process in chunks to avoid memory issues
    chunk_size = 10000
    for start_idx in range(window_size, len(prices), chunk_size):
        end_idx = min(start_idx + chunk_size, len(prices))
        
        for i in range(start_idx, end_idx):
            # Get last window_size candles
            window_data = prices.iloc[i-window_size:i]
            
            if len(window_data) < window_size:
                continue
            
            # Predict
            try:
                result = predict(window_data, models_dir)
                
                predictions_list.append({
                    'timestamp': prices.index[i],
                    'probability': result['probability'],
                    'prediction': result['prediction'],
                    'threshold': result['threshold'],
                })
            except Exception as e:
                logger.warning(f"Prediction failed at {prices.index[i]}: {e}")
                continue
        
        if (end_idx - window_size) % 50000 == 0:
            logger.info(f"  Processed {end_idx - window_size:,} / {len(prices) - window_size:,} predictions")
    
    predictions_df = pd.DataFrame(predictions_list)
    predictions_df.set_index('timestamp', inplace=True)
    
    logger.info(f"Generated {len(predictions_df):,} predictions")
    logger.info(f"  BUY signals: {(predictions_df['prediction']==1).sum():,}")
    logger.info(f"  HOLD signals: {(predictions_df['prediction']==0).sum():,}")
    
    return predictions_df, prices


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
