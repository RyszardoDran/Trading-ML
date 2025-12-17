"""Backtesting framework for trading strategies.

Modules:
    - metrics: Risk and performance metrics (Sharpe, Sortino, Max DD, VaR, etc.)
    - position_sizer: Position sizing strategies (fixed, risk-based, Kelly)
    - backtest_engine: Core backtesting engine with walk-forward analysis
    - config: Configuration dataclass for backtesting parameters

Example:
    >>> from ml.src.backtesting import BacktestEngine, BacktestConfig
    >>> config = BacktestConfig(initial_capital=100000)
    >>> engine = BacktestEngine(config)
    >>> results = engine.run(predictions, prices)
"""

from ml.src.backtesting.backtest_engine import BacktestEngine
from ml.src.backtesting.config import BacktestConfig
from ml.src.backtesting.position_sizer import PositionSizer, PositionSizingMethod

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "PositionSizer",
    "PositionSizingMethod",
]
