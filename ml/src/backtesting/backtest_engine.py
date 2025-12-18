"""Core backtesting engine for trading strategies.

Implements comprehensive backtesting with:
- Walk-forward analysis with rolling windows
- Realistic transaction costs (spread, slippage, commission)
- Position sizing strategies
- Risk management (max drawdown limits)
- Data quality validation
- Comprehensive logging and error handling

Follows life-critical system standards with 100% validation and reproducibility.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.backtesting.config import BacktestConfig
from ml.src.backtesting.metrics import calculate_all_metrics
from ml.src.backtesting.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Execute backtests on trading strategies.
    
    Attributes:
        config: Backtesting configuration
        position_sizer: Position sizing calculator
        equity_curve: Time series of portfolio equity
        trades: DataFrame of executed trades
        current_capital: Current portfolio capital
        
    Examples:
        >>> config = BacktestConfig(initial_capital=100000)
        >>> engine = BacktestEngine(config)
        >>> results = engine.run(predictions, prices, targets)
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize backtest engine.
        
        Args:
            config: Backtesting configuration
            
        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self.config.validate()
        
        # Initialize position sizer
        self.position_sizer = PositionSizer(
            method=config.position_sizing,
            fixed_size=config.fixed_position_size,
            risk_per_trade=config.risk_per_trade,
            kelly_fraction=config.kelly_fraction,
            min_size=config.min_position_size,
            max_size=config.max_position_size,
        )
        
        # State
        self.equity_curve: Optional[pd.Series] = None
        self.trades: Optional[pd.DataFrame] = None
        self.current_capital = config.initial_capital
        
        logger.info("BacktestEngine initialized")
        logger.info(f"  Initial capital: ${config.initial_capital:,.2f}")
        logger.info(f"  Position sizing: {config.position_sizing.value}")
        logger.info(f"  Transaction costs: ${config.total_transaction_cost():.2f} per trade")
    
    def run(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        targets: Optional[pd.Series] = None,
    ) -> dict:
        """Run backtest on predictions.
        
        Args:
            predictions: DataFrame with columns ['probability', 'prediction', 'threshold']
                        Index must be DatetimeIndex aligned with prices
            prices: DataFrame with OHLCV columns, DatetimeIndex
            targets: Optional Series of actual outcomes (1=win, 0=loss) for validation
            
        Returns:
            Dictionary with:
                - metrics: Performance metrics
                - equity_curve: Portfolio equity over time
                - trades: DataFrame of executed trades
                - config: Backtest configuration used
                
        Raises:
            ValueError: If inputs are invalid
            
        Examples:
            >>> predictions = pd.DataFrame({
            ...     'probability': [0.7, 0.6, 0.8],
            ...     'prediction': [1, 1, 1],
            ... }, index=pd.date_range('2024-01-01', periods=3, freq='1min'))
            >>> prices = pd.DataFrame({
            ...     'Open': [2000, 2005, 2010],
            ...     'High': [2010, 2015, 2020],
            ...     'Low': [1995, 2000, 2005],
            ...     'Close': [2005, 2010, 2015],
            ...     'Volume': [100, 110, 120],
            ... }, index=predictions.index)
            >>> engine = BacktestEngine(BacktestConfig())
            >>> results = engine.run(predictions, prices)
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)
        
        # Validate inputs
        self._validate_inputs(predictions, prices, targets)
        
        # Store full M1 prices for accurate SL/TP detection
        self.full_prices = prices.copy()
        
        # Align predictions with prices (predictions may be M5, prices M1)
        # We only take price rows that match prediction timestamps for iteration
        predictions_aligned = predictions.copy()
        prices_aligned = prices.loc[predictions.index] if not prices.index.equals(predictions.index) else prices.copy()
        
        logger.info(f"Backtest period: {prices.index[0]} to {prices.index[-1]}")
        logger.info(f"Total M1 candles: {len(prices):,}")
        logger.info(f"Prediction points (M5): {len(predictions):,}")
        logger.info(f"Signals with prediction=1: {(predictions['prediction']==1).sum():,}")
        
        # Run backtest
        if self.config.walk_forward_enabled:
            logger.info("Running walk-forward analysis...")
            trades, equity = self._run_walk_forward(predictions_aligned, prices_aligned, targets)
        else:
            logger.info("Running simple backtest...")
            trades, equity = self._run_simple_backtest(predictions_aligned, prices_aligned, targets)
        
        # Store results
        self.trades = trades
        self.equity_curve = equity
        
        # Calculate metrics
        logger.info("\nCalculating performance metrics...")
        metrics = calculate_all_metrics(
            equity_curve=equity,
            initial_capital=self.config.initial_capital,
            trades=trades,
            periods_per_year=self.config.periods_per_year,
            risk_free_rate=self.config.risk_free_rate,
        )
        
        # Log summary
        self._log_summary(metrics, trades)
        
        # Save results if configured
        if self.config.save_trades or self.config.save_equity_curve:
            self._save_results(trades, equity, metrics)
        
        return {
            'metrics': metrics,
            'equity_curve': equity,
            'trades': trades,
            'config': self.config.to_dict(),
        }
    
    def _validate_inputs(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        targets: Optional[pd.Series],
    ) -> None:
        """Validate input data.
        
        Raises:
            TypeError: If inputs have wrong type
            ValueError: If inputs have invalid values or structure
        """
        # Type checks
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError(f"predictions must be pd.DataFrame, got {type(predictions)}")
        
        if not isinstance(prices, pd.DataFrame):
            raise TypeError(f"prices must be pd.DataFrame, got {type(prices)}")
        
        if targets is not None and not isinstance(targets, pd.Series):
            raise TypeError(f"targets must be pd.Series or None, got {type(targets)}")
        
        # Required columns
        required_pred_cols = ['probability', 'prediction']
        missing_pred = set(required_pred_cols) - set(predictions.columns)
        if missing_pred:
            raise ValueError(f"predictions missing columns: {missing_pred}")
        
        required_price_cols = ['Open', 'High', 'Low', 'Close']
        missing_price = set(required_price_cols) - set(prices.columns)
        if missing_price:
            raise ValueError(f"prices missing columns: {missing_price}")
        
        # Index checks
        if not isinstance(predictions.index, pd.DatetimeIndex):
            raise ValueError("predictions index must be DatetimeIndex")
        
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices index must be DatetimeIndex")
        
        # Value validation
        if len(predictions) == 0:
            raise ValueError("predictions cannot be empty")
        
        if len(prices) == 0:
            raise ValueError("prices cannot be empty")
        
        if not predictions['probability'].between(0, 1).all():
            raise ValueError("probability values must be in [0, 1]")
        
        if not predictions['prediction'].isin([0, 1]).all():
            raise ValueError("prediction values must be 0 or 1")
        
        if (prices[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            raise ValueError("prices must be positive")
        
        if not np.isfinite(prices[['Open', 'High', 'Low', 'Close']]).all().all():
            raise ValueError("prices contain non-finite values")
        
        logger.info("✅ Input validation passed")
    
    def _align_data(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        targets: Optional[pd.Series],
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """Align predictions, prices, and targets on common index.
        
        Args:
            predictions: Predictions DataFrame
            prices: Prices DataFrame
            targets: Optional targets Series
            
        Returns:
            Tuple of (aligned_predictions, aligned_prices, aligned_targets)
        """
        # Find common index
        common_idx = predictions.index.intersection(prices.index)
        
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between predictions and prices")
        
        predictions = predictions.loc[common_idx]
        prices = prices.loc[common_idx]
        
        if targets is not None:
            common_idx = common_idx.intersection(targets.index)
            if len(common_idx) == 0:
                raise ValueError("No overlapping dates between predictions and targets")
            predictions = predictions.loc[common_idx]
            prices = prices.loc[common_idx]
            targets = targets.loc[common_idx]
        
        logger.info(f"Data aligned: {len(common_idx):,} common data points")
        
        return predictions, prices, targets
    
    def _run_simple_backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        targets: Optional[pd.Series],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Run simple backtest without walk-forward analysis.
        
        Args:
            predictions: Aligned predictions
            prices: Aligned prices
            targets: Aligned targets (optional)
            
        Returns:
            Tuple of (trades DataFrame, equity curve Series)
        """
        trades_list = []
        equity = [self.config.initial_capital]
        equity_index = [prices.index[0]]
        
        capital = self.config.initial_capital
        trades_today = 0
        current_date = None
        
        # Track active trade (realistic trading - one position at a time)
        active_trade_exit_time = None
        
        for i in range(len(predictions)):
            timestamp = predictions.index[i]
            pred_row = predictions.iloc[i]
            price_row = prices.iloc[i]
            
            # Reset daily trade counter
            if current_date is None or timestamp.date() != current_date:
                trades_today = 0
                current_date = timestamp.date()
            
            # Skip if we have an active trade (realistic trading)
            if active_trade_exit_time is not None and timestamp < active_trade_exit_time:
                continue
            
            # Check if we should take this trade
            should_trade = (
                pred_row['prediction'] == 1 and
                pred_row['probability'] >= self.config.min_probability
            )
            
            # Check daily trade limit
            if self.config.max_trades_per_day is not None:
                if trades_today >= self.config.max_trades_per_day:
                    should_trade = False
            
            # Check max drawdown
            current_dd = (max(equity) - capital) / max(equity) if equity else 0
            if current_dd >= self.config.max_drawdown_pct:
                logger.warning(f"Max drawdown reached ({current_dd:.2%}), stopping backtest")
                break
            
            if should_trade:
                # Calculate position size
                position_size = self.position_sizer.calculate_size(
                    capital=capital,
                    signal_prob=pred_row['probability'],
                )
                
                # Get ATR value for SL/TP calculation
                atr_value = pred_row.get('atr', None)
                
                # Get ALL M1 candles from this point forward (accurate SL/TP detection)
                future_prices = self.full_prices.loc[timestamp:]
                
                # Execute trade
                trade_result = self._execute_trade(
                    timestamp=timestamp,
                    position_size=position_size,
                    entry_price=price_row['Close'],
                    probability=pred_row['probability'],
                    atr=atr_value,
                    future_prices=future_prices,
                    actual_outcome=targets.iloc[i] if targets is not None else None,
                )
                
                # Skip trades that didn't hit SL/TP in available data
                if trade_result is not None:
                    trades_list.append(trade_result)
                    capital += trade_result['pnl']
                    trades_today += 1
                    
                    # Mark this trade as active until its exit time
                    active_trade_exit_time = trade_result['exit_time']
            
            # Record equity
            equity.append(capital)
            equity_index.append(timestamp)
        
        # Create DataFrames
        trades_df = pd.DataFrame(trades_list)
        equity_series = pd.Series(equity, index=equity_index)
        
        return trades_df, equity_series
    
    def _execute_trade(
        self,
        timestamp: pd.Timestamp,
        position_size: float,
        entry_price: float,
        probability: float,
        atr: Optional[float] = None,
        future_prices: Optional[pd.DataFrame] = None,
        actual_outcome: Optional[int] = None,
    ) -> dict:
        """Execute a single trade and calculate PnL.
        
        Args:
            timestamp: Trade timestamp (entry time)
            position_size: Position size in lots
            entry_price: Entry price
            probability: Model probability
            atr: ATR value for SL/TP calculation
            future_prices: DataFrame with future prices for exit simulation
            actual_outcome: Actual trade outcome (1=win, 0=loss) if known
            
        Returns:
            Dictionary with trade details including SL, TP, and exit time
        """
        # Transaction costs
        transaction_cost = self.config.total_transaction_cost()
        
        # Calculate SL and TP levels
        if atr is not None and not np.isnan(atr):
            sl_distance = self.config.atr_sl_multiplier * atr
            tp_distance = self.config.atr_tp_multiplier * atr
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            # Fallback to percentage-based levels if ATR not available
            sl_price = entry_price * 0.999  # 0.1% stop loss
            tp_price = entry_price * 1.002  # 0.2% take profit
        
        # Simulate exit by checking future prices
        exit_time = None
        exit_price = None
        is_win = None
        
        if future_prices is not None and len(future_prices) > 1:
            for idx in range(1, len(future_prices)):
                candle = future_prices.iloc[idx]
                
                sl_hit = candle['Low'] <= sl_price
                tp_hit = candle['High'] >= tp_price
                
                # If both hit in same candle, determine which was hit first
                if sl_hit and tp_hit:
                    # Check candle direction: if opened closer to TP, assume TP hit first
                    open_price = candle['Open']
                    dist_to_tp = abs(open_price - tp_price)
                    dist_to_sl = abs(open_price - sl_price)
                    
                    if dist_to_tp < dist_to_sl:
                        # TP closer to open, likely hit first
                        exit_time = future_prices.index[idx]
                        exit_price = tp_price
                        is_win = True
                    else:
                        # SL closer to open, likely hit first
                        exit_time = future_prices.index[idx]
                        exit_price = sl_price
                        is_win = False
                    break
                
                # Only SL hit
                elif sl_hit:
                    exit_time = future_prices.index[idx]
                    exit_price = sl_price
                    is_win = False
                    break
                
                # Only TP hit
                elif tp_hit:
                    exit_time = future_prices.index[idx]
                    exit_price = tp_price
                    is_win = True
                    break
        
        # If no exit found in available data, skip this trade (incomplete data)
        if exit_time is None:
            # Return None to signal this trade should be skipped
            return None
        
        # Calculate PnL based on actual exit
        gross_pnl = position_size * (exit_price - entry_price)
        net_pnl = gross_pnl - transaction_cost
        
        # Calculate trade duration in minutes
        duration_minutes = (exit_time - timestamp).total_seconds() / 60 if exit_time else None
        
        return {
            'timestamp': timestamp,
            'exit_time': exit_time,
            'duration_minutes': duration_minutes,
            'position_size': position_size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'probability': probability,
            'is_win': is_win,
            'gross_pnl': gross_pnl,
            'transaction_cost': transaction_cost,
            'pnl': net_pnl,
        }
    
    def _run_walk_forward(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        targets: Optional[pd.Series],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Run walk-forward analysis.
        
        Splits data into overlapping train/test windows and backtests each window.
        
        Args:
            predictions: Predictions DataFrame
            prices: Prices DataFrame
            targets: Targets Series (optional)
            
        Returns:
            Tuple of (trades DataFrame, equity curve Series)
        """
        logger.info("Walk-forward configuration:")
        logger.info(f"  Train window: {self.config.walk_forward_train_days} days")
        logger.info(f"  Test window: {self.config.walk_forward_test_days} days")
        logger.info(f"  Step size: {self.config.walk_forward_step_days} days")
        
        all_trades = []
        equity = [self.config.initial_capital]
        equity_index = [prices.index[0]]
        capital = self.config.initial_capital
        
        # Calculate windows
        start_date = prices.index[0]
        end_date = prices.index[-1]
        current_date = start_date
        
        window_num = 0
        
        while current_date < end_date:
            window_num += 1
            
            # Define train and test periods
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.walk_forward_train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.walk_forward_test_days)
            
            # Get test data for this window
            test_mask = (prices.index >= test_start) & (prices.index < test_end)
            
            if test_mask.sum() == 0:
                break
            
            window_preds = predictions[test_mask]
            window_prices = prices[test_mask]
            window_targets = targets[test_mask] if targets is not None else None
            
            logger.info(f"\nWindow {window_num}: {test_start.date()} to {test_end.date()}")
            logger.info(f"  Test samples: {len(window_preds):,}")
            
            # Run backtest on this window
            window_trades, window_equity = self._run_simple_backtest(
                window_preds,
                window_prices,
                window_targets,
            )
            
            # Append results
            if len(window_trades) > 0:
                all_trades.append(window_trades)
            
            # Update capital from last trade
            if len(window_trades) > 0:
                capital = window_equity.iloc[-1]
            
            # Step forward
            current_date += timedelta(days=self.config.walk_forward_step_days)
        
        # Combine all trades
        if len(all_trades) > 0:
            trades_df = pd.concat(all_trades, ignore_index=True)
        else:
            trades_df = pd.DataFrame()
        
        # Reconstruct equity curve
        if len(trades_df) > 0:
            equity_series = pd.Series(
                self.config.initial_capital,
                index=[prices.index[0]]
            ).append(
                (self.config.initial_capital + trades_df['pnl'].cumsum()).reset_index(drop=True)
            )
            equity_series.index = pd.concat([
                pd.Series([prices.index[0]]),
                trades_df['timestamp']
            ]).reset_index(drop=True)
        else:
            equity_series = pd.Series([self.config.initial_capital], index=[prices.index[0]])
        
        logger.info(f"\nWalk-forward complete: {window_num} windows processed")
        
        return trades_df, equity_series
    
    def _log_summary(self, metrics: dict, trades: pd.DataFrame) -> None:
        """Log backtest summary."""
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\nReturns:")
        logger.info(f"  Total Return:      {metrics['total_return']:>10.2%}")
        logger.info(f"  Annualized Return: {metrics['annualized_return']:>10.2%}")
        
        logger.info(f"\nRisk-Adjusted:")
        logger.info(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
        logger.info(f"  Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}")
        logger.info(f"  Calmar Ratio:      {metrics['calmar_ratio']:>10.2f}")
        
        logger.info(f"\nDrawdown:")
        logger.info(f"  Max Drawdown:      {metrics['max_drawdown']:>10.2%}")
        
        logger.info(f"\nRisk:")
        logger.info(f"  Volatility:        {metrics['volatility']:>10.2%}")
        logger.info(f"  VaR 95%:           {metrics['var_95']:>10.2%}")
        logger.info(f"  CVaR 95%:          {metrics['cvar_95']:>10.2%}")
        
        logger.info(f"\nTrades:")
        logger.info(f"  Total Trades:      {metrics['total_trades']:>10,}")
        logger.info(f"  Win Rate:          {metrics['win_rate']:>10.2%}")
        logger.info(f"  Profit Factor:     {metrics['profit_factor']:>10.2f}")
        logger.info(f"  Avg Win/Loss:      {metrics['avg_win_loss_ratio']:>10.2f}")
        
        logger.info("=" * 80)
    
    def _save_results(
        self,
        trades: pd.DataFrame,
        equity: pd.Series,
        metrics: dict,
    ) -> None:
        """Save backtest results to files."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.save_trades and len(trades) > 0:
            trades_path = output_dir / f"trades_{timestamp}.csv"
            trades.to_csv(trades_path, index=False)
            logger.info(f"Trades saved to: {trades_path}")
        
        if self.config.save_equity_curve:
            equity_path = output_dir / f"equity_{timestamp}.csv"
            equity.to_csv(equity_path, header=['equity'])
            logger.info(f"Equity curve saved to: {equity_path}")
        
        # Save metrics
        metrics_path = output_dir / f"metrics_{timestamp}.json"
        import json
        
        # Convert non-serializable values
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                metrics_serializable[k] = str(v)
            elif isinstance(v, (np.integer, np.floating)):
                metrics_serializable[k] = float(v)
            elif pd.isna(v):
                metrics_serializable[k] = None
            else:
                metrics_serializable[k] = v
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")
