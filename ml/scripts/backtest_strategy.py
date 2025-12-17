"""CLI script for backtesting trading strategy with trained XGBoost model.

This script loads a trained model and historical price data, then simulates
trading strategy performance under various scenarios (nominal, stress tests).

Features:
- Loads pickled model and scaler from ml/outputs/models/
- Loads historical OHLCV data
- Generates trading signals using the trained model
- Simulates portfolio performance with realistic constraints:
  * Transaction costs (spread, commission)
  * Daily trade limits
  * Risk metrics (drawdown, Sharpe ratio, win rate)
- Backtests multiple scenarios (nominal, drawdown stress, liquidity stress)
- Saves detailed results to ml/outputs/backtest/

Example:
    # Backtest with default paths
    python ml/scripts/backtest_strategy.py
    
    # Backtest with custom data path
    python ml/scripts/backtest_strategy.py --data-path data/historical.pkl
    
    # Backtest with custom initial capital
    python ml/scripts/backtest_strategy.py --initial-capital 100000
    
    # Backtest with custom trade limits
    python ml/scripts/backtest_strategy.py --max-trades-per-day 3
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("ml/outputs/logs/backtest_strategy.log"),
            mode="a",
        ),
    ],
)
logger = logging.getLogger(__name__)


def load_model_artifacts(
    models_dir: Path,
) -> Tuple[CalibratedClassifierCV, RobustScaler, dict]:
    """Load trained model, scaler, and metadata from disk.

    Args:
        models_dir: Directory containing model artifacts

    Returns:
        Tuple of (model, scaler, metadata_dict)

    Raises:
        FileNotFoundError: If required artifact files are missing
    """
    logger.info(f"Loading artifacts from {models_dir}")

    # Load model
    model_path = models_dir / "sequence_xgb_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"✅ Loaded model from {model_path}")

    # Load scaler
    scaler_path = models_dir / "sequence_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"✅ Loaded scaler from {scaler_path}")

    # Load metadata
    metadata_path = models_dir / "sequence_threshold.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"✅ Loaded metadata: threshold={metadata.get('threshold'):.4f}")

    return model, scaler, metadata


def load_historical_data(data_path: Path) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load historical OHLCV data for backtesting.

    Expects pickle file containing OHLCV DataFrame with columns:
    ['Open', 'High', 'Low', 'Close', 'Volume'] and DatetimeIndex

    Args:
        data_path: Path to historical OHLCV data pickle file

    Returns:
        Tuple of (OHLCV DataFrame, DatetimeIndex)

    Raises:
        FileNotFoundError: If data file not found
        ValueError: If data structure is invalid
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading historical data from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict) and "data" in data:
        df = data["data"]
    else:
        raise ValueError("Data must be DataFrame or dict with 'data' key")

    # Validate required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index)
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")

    logger.info(
        f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}"
    )

    return df, df.index


def simulate_trades(
    prices: np.ndarray,
    timestamps: pd.DatetimeIndex,
    signals: np.ndarray,
    initial_capital: float = 100000.0,
    spread: float = 0.0001,
    commission: float = 0.0005,
    max_trades_per_day: int = 5,
) -> Dict[str, float]:
    """Simulate trading strategy and calculate performance metrics.

    Args:
        prices: Close prices array
        timestamps: DatetimeIndex of price timestamps
        signals: Binary trading signals (1=trade, 0=no trade)
        initial_capital: Starting capital
        spread: Bid-ask spread as fraction of price
        commission: Commission per trade as fraction of trade value
        max_trades_per_day: Maximum trades per calendar day

    Returns:
        Dictionary with performance metrics:
        - total_return: Cumulative return (%)
        - sharpe_ratio: Annualized Sharpe ratio
        - win_rate: Percentage of winning trades
        - max_drawdown: Maximum drawdown (%)
        - num_trades: Number of executed trades
        - final_equity: Final portfolio value
    """
    logger.info("Simulating trading strategy...")

    # Apply daily cap
    signals = signals.copy()
    dates = pd.DatetimeIndex(timestamps).date
    unique_days = np.unique(dates)
    for d in unique_days:
        day_idx = np.where(dates == d)[0]
        day_trades = day_idx[signals[day_idx] == 1]
        if len(day_trades) > max_trades_per_day:
            # Keep only top-k signals (would use probability ranking)
            signals[day_trades[max_trades_per_day:]] = 0

    # Simulate P&L
    trades = np.where(signals == 1)[0]
    logger.info(f"Total signals: {signals.sum()}, Executed trades: {len(trades)}")

    equity = np.full(len(prices), initial_capital, dtype=np.float64)
    returns = np.zeros(len(prices))
    trade_pnl = []

    for i in range(1, len(prices)):
        # Check if trade signal at previous candle
        if signals[i - 1] == 1:
            # Entry at close of signal candle
            entry_price = prices[i - 1] * (1 + spread / 2)
            # Exit at close of next candle
            exit_price = prices[i] * (1 - spread / 2)
            # Account for commission
            entry_cost = entry_price * (1 + commission)
            exit_value = exit_price * (1 - commission)
            # P&L per unit
            pnl_pct = (exit_value - entry_cost) / entry_cost
            trade_pnl.append(pnl_pct)
            # Apply to equity
            equity[i] = equity[i - 1] * (1 + pnl_pct)
        else:
            # No trade, carry forward equity
            equity[i] = equity[i - 1]

        # Daily returns
        if i > 0:
            returns[i] = equity[i] / equity[i - 1] - 1

    # Calculate metrics
    total_return = (equity[-1] / initial_capital - 1) * 100
    annual_returns = np.mean(returns) * 252
    annual_volatility = np.std(returns) * np.sqrt(252)
    sharpe = annual_returns / annual_volatility if annual_volatility > 0 else 0
    max_dd = (np.max(np.maximum.accumulate(equity) - equity) / np.max(equity)) * 100

    winning_trades = sum(1 for pnl in trade_pnl if pnl > 0)
    win_rate = (winning_trades / len(trade_pnl) * 100) if trade_pnl else 0

    metrics = {
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe),
        "win_rate_pct": float(win_rate),
        "max_drawdown_pct": float(max_dd),
        "num_trades": int(len(trade_pnl)),
        "final_equity": float(equity[-1]),
        "annual_return_pct": float(annual_returns * 100),
        "annual_volatility_pct": float(annual_volatility * 100),
    }

    logger.info(f"✅ Simulation complete!")
    logger.info(f"   Trades: {len(trade_pnl)}, Win rate: {win_rate:.2f}%")
    logger.info(f"   Return: {total_return:.2f}%, Sharpe: {sharpe:.4f}")

    return metrics


def backtest_scenarios(
    model: CalibratedClassifierCV,
    scaler: RobustScaler,
    metadata: Dict,
    prices: np.ndarray,
    timestamps: pd.DatetimeIndex,
    features: np.ndarray,
    initial_capital: float = 100000.0,
    max_trades_per_day: int = 5,
) -> Dict:
    """Run backtest across multiple scenarios.

    Args:
        model: Trained classifier
        scaler: Fitted scaler
        metadata: Model metadata with threshold
        prices: Close prices
        timestamps: DatetimeIndex
        features: Feature matrix (pre-computed)
        initial_capital: Starting capital
        max_trades_per_day: Daily trade limit

    Returns:
        Dictionary with results for each scenario
    """
    logger.info("Running backtest scenarios...")

    # Scale features
    features_scaled = scaler.transform(features)

    # Generate signals
    proba = model.predict_proba(features_scaled)[:, 1]
    threshold = metadata.get("threshold", 0.65)
    signals = (proba >= threshold).astype(int)
    logger.info(f"Signal generation: {signals.sum()} signals with threshold={threshold:.4f}")

    # Nominal scenario
    logger.info("\n" + "=" * 70)
    logger.info("SCENARIO 1: Nominal (base case)")
    logger.info("=" * 70)
    nominal = simulate_trades(
        prices=prices,
        timestamps=timestamps,
        signals=signals,
        initial_capital=initial_capital,
        spread=0.0001,
        commission=0.0005,
        max_trades_per_day=max_trades_per_day,
    )

    # Stress test 1: Wider spreads (liquidity crisis)
    logger.info("\n" + "=" * 70)
    logger.info("SCENARIO 2: Stress (wider spreads)")
    logger.info("=" * 70)
    stress_wide = simulate_trades(
        prices=prices,
        timestamps=timestamps,
        signals=signals,
        initial_capital=initial_capital,
        spread=0.001,  # 10x wider
        commission=0.002,  # 4x higher commission
        max_trades_per_day=max_trades_per_day,
    )

    # Stress test 2: Threshold increase (more conservative)
    logger.info("\n" + "=" * 70)
    logger.info("SCENARIO 3: Conservative (higher threshold)")
    logger.info("=" * 70)
    conservative_threshold = threshold + 0.10
    conservative_signals = (proba >= conservative_threshold).astype(int)
    logger.info(f"Conservative threshold: {conservative_threshold:.4f}")
    stress_conservative = simulate_trades(
        prices=prices,
        timestamps=timestamps,
        signals=conservative_signals,
        initial_capital=initial_capital,
        spread=0.0001,
        commission=0.0005,
        max_trades_per_day=max_trades_per_day,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "threshold": threshold,
            "window_size": metadata.get("window_size"),
        },
        "parameters": {
            "initial_capital": initial_capital,
            "max_trades_per_day": max_trades_per_day,
            "backtest_period": {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
                "candles": len(timestamps),
            },
        },
        "scenarios": {
            "nominal": nominal,
            "stress_wide_spreads": stress_wide,
            "conservative_threshold": stress_conservative,
        },
    }

    return results


def save_backtest_results(
    results: Dict,
    output_dir: Path,
) -> Path:
    """Save backtest results to JSON file.

    Args:
        results: Backtest results dictionary
        output_dir: Directory to save results

    Returns:
        Path to saved results file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"backtest_results_{timestamp}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Backtest results saved to {results_path}")
    return results_path


def print_backtest_summary(results: Dict):
    """Print summary of backtest results.

    Args:
        results: Backtest results dictionary
    """
    logger.info("=" * 70)
    logger.info("BACKTEST RESULTS SUMMARY")
    logger.info("=" * 70)

    for scenario_name, metrics in results["scenarios"].items():
        logger.info(f"\n{scenario_name.upper()}:")
        logger.info(f"  Total Return:     {metrics['total_return_pct']:7.2f}%")
        logger.info(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:7.4f}")
        logger.info(f"  Win Rate:         {metrics['win_rate_pct']:7.2f}%")
        logger.info(f"  Max Drawdown:     {metrics['max_drawdown_pct']:7.2f}%")
        logger.info(f"  Num Trades:       {metrics['num_trades']:7d}")
        logger.info(f"  Final Equity:     {metrics['final_equity']:12,.0f}")

    logger.info("=" * 70)


def main():
    """Main entry point for backtest_strategy CLI."""
    parser = argparse.ArgumentParser(
        description="Backtest trading strategy with trained XGBoost model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest with default paths
  python ml/scripts/backtest_strategy.py
  
  # Backtest with custom data path
  python ml/scripts/backtest_strategy.py --data-path data/historical_xauusd.pkl
  
  # Backtest with custom capital
  python ml/scripts/backtest_strategy.py --initial-capital 50000
  
  # Backtest with stricter daily limits
  python ml/scripts/backtest_strategy.py --max-trades-per-day 3
        """,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/outputs/models"),
        help="Directory containing model artifacts (default: ml/outputs/models)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/xauusd_20years.pkl"),
        help="Path to historical OHLCV data (default: data/xauusd_20years.pkl)",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("ml/outputs/backtest_features.pkl"),
        help="Path to pre-computed features (default: ml/outputs/backtest_features.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/outputs/backtest"),
        help="Directory to save results (default: ml/outputs/backtest)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial trading capital (default: 100000)",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=5,
        help="Maximum trades per day (default: 5)",
    )

    args = parser.parse_args()

    try:
        # Load artifacts
        model, scaler, metadata = load_model_artifacts(args.model_path)

        # Load historical data
        df, timestamps = load_historical_data(args.data_path)
        prices = df["Close"].values

        # Load or generate features
        if args.features_path.exists():
            logger.info(f"Loading pre-computed features from {args.features_path}")
            with open(args.features_path, "rb") as f:
                features = pickle.load(f)
            logger.info(f"✅ Loaded features: shape={features.shape}")
        else:
            logger.warning(
                f"Features not found at {args.features_path}. "
                "Please provide pre-computed features for backtesting."
            )
            raise FileNotFoundError(f"Features file not found: {args.features_path}")

        # Validate shapes
        if len(prices) != len(features):
            raise ValueError(
                f"Price/feature length mismatch: {len(prices)} vs {len(features)}"
            )

        # Run backtest
        results = backtest_scenarios(
            model=model,
            scaler=scaler,
            metadata=metadata,
            prices=prices,
            timestamps=timestamps,
            features=features,
            initial_capital=args.initial_capital,
            max_trades_per_day=args.max_trades_per_day,
        )

        # Save and display results
        results_path = save_backtest_results(results, args.output_dir)
        print_backtest_summary(results)

        logger.info(f"✅ Backtest completed successfully!")
        logger.info(f"📊 Results saved to: {results_path}")

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise SystemExit(1) from e
    except ValueError as e:
        logger.error(f"❌ Invalid data: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
