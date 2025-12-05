---
applyTo: "**/*.py"
---

# Python Machine Learning & Data Analysis Guidelines

## Project Overview

This Python module specializes in:
- **Stock Chart Analysis**: Analyzing historical stock price charts and patterns
- **Time-Series Forecasting**: Predicting future price movements using forecasting models
- **Machine Learning Models**: Building and training ML models for stock price prediction
- **Data Processing**: Cleaning, preprocessing, and feature engineering financial data
- **Backtesting**: Validating trading strategies against historical data

## General Guidelines

Follow PEP 8 standards and emphasize code clarity, reproducibility, and scientific rigor.

- **Code Style**: Use Black for code formatting; use Flake8 for linting; use isort for import organization.
- **Type Hints**: Use type hints extensively for better code documentation and IDE support.
- **Virtual Environments**: Always use virtual environments (venv, Poetry, Conda).
- **Reproducibility**: Use random seeds, version pinning, and clear documentation.
- **Scientific Rigor**: Validate assumptions, document data sources, and report model metrics transparently.
- **Error Handling**: Use logging (Python's logging module) for debugging and monitoring.

## Project Structure

```
ml/
├── src/
│   ├── analysis/              # Chart and technical analysis modules
│   ├── models/                # ML models and training logic
│   ├── data/                  # Data loading, preprocessing, feature engineering
│   ├── forecasting/           # Time-series forecasting models
│   ├── backtesting/           # Strategy backtesting framework
│   ├── utils/                 # Utility functions
│   └── config/                # Configuration files
├── notebooks/                 # Jupyter notebooks for exploration (not production code)
├── tests/                     # Unit and integration tests
├── requirements.txt           # Dependencies
├── .env.example               # Environment variables template
└── README.md                  # Project documentation
```

## Dependencies & Tools

### Core Libraries

```
pandas>=2.0.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
scikit-learn>=1.3.0         # ML algorithms
matplotlib>=3.7.0           # Plotting
seaborn>=0.12.0             # Statistical visualization
```

### Time-Series & Forecasting

```
statsmodels>=0.14.0         # Statistical models (ARIMA, etc.)
keras>=2.13.0               # Deep learning
tensorflow>=2.13.0          # Neural networks (if needed)
pmdarima>=2.0.0             # Auto ARIMA
```

### Data Sources

```
yfinance>=0.2.0             # Yahoo Finance data
pandas-datareader>=0.10.0   # Alternative data sources
```

### Development & Testing

```
pytest>=7.4.0               # Testing framework
pytest-cov>=4.1.0           # Coverage reporting
black>=23.0.0               # Code formatter
flake8>=6.0.0               # Linter
isort>=5.12.0               # Import sorting
```

## Data Processing & Feature Engineering

### Data Loading

```python
import pandas as pd
from pathlib import Path
from src.data.loaders import load_stock_data

# Load historical stock data
data = load_stock_data(symbol='AAPL', start_date='2020-01-01', end_date='2023-12-31')
print(data.head())
print(data.info())
```

### Data Cleaning

```python
# Remove duplicates and NaN values
data = data.drop_duplicates()
data = data.dropna()

# Handle missing data (forward fill for time-series)
data['Close'] = data['Close'].fillna(method='ffill')

# Remove outliers (example: 3-sigma rule)
mean = data['Close'].mean()
std = data['Close'].std()
data = data[(data['Close'] > mean - 3*std) & (data['Close'] < mean + 3*std)]
```

### Feature Engineering

```python
from src.data.features import (
    create_technical_indicators,
    create_lag_features,
    create_rolling_features
)

# Add technical indicators
data = create_technical_indicators(data)  # SMA, EMA, MACD, RSI, Bollinger Bands

# Add lag features (previous days' prices)
data = create_lag_features(data, lags=[1, 2, 5, 10])

# Add rolling statistics
data = create_rolling_features(data, windows=[5, 20, 50])

# Forward shift target variable (next day's return)
data['target'] = data['Close'].pct_change().shift(-1)
data = data.dropna()
```

## Chart Analysis & Technical Analysis

### Supported Indicators

```python
from src.analysis.indicators import (
    SMA, EMA, RSI, MACD, BollingerBands,
    ATR, StochasticOscillator, OBV
)

# Moving Averages
sma_20 = SMA(data['Close'], window=20)
ema_12 = EMA(data['Close'], span=12)

# Momentum Indicators
rsi = RSI(data['Close'], period=14)
macd_line, signal_line, histogram = MACD(data['Close'])

# Volatility Indicators
bb_upper, bb_middle, bb_lower = BollingerBands(data['Close'], window=20, num_std=2)
atr = ATR(data[['High', 'Low', 'Close']], period=14)

# Volume Indicators
obv = OBV(data['Close'], data['Volume'])
```

### Chart Pattern Recognition

```python
from src.analysis.patterns import (
    detect_support_resistance,
    detect_head_and_shoulders,
    detect_double_top_bottom
)

# Detect support and resistance levels
support_levels, resistance_levels = detect_support_resistance(data['Close'])

# Detect chart patterns
h_and_s = detect_head_and_shoulders(data['Close'])
double_top = detect_double_top_bottom(data['Close'], pattern='top')
```

## Time-Series Forecasting

### ARIMA Models

```python
from statsmodels.tsa.arima.model import ARIMA
from src.forecasting.evaluation import calculate_metrics

# Fit ARIMA model
model = ARIMA(data['Close'], order=(1, 1, 1))
results = model.fit()

# Forecast next 10 days
forecast = results.get_forecast(steps=10)
forecast_ci = forecast.conf_int()

print(forecast.summary_table())
```

### Machine Learning Models

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.models.train import train_test_split, cross_validate

# Prepare features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data (temporal split for time-series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = model.predict(X_test_scaled)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Deep Learning (Neural Networks)

```python
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

# Prepare sequences for LSTM
from src.models.preprocessing import create_sequences

X_train_seq, y_train_seq = create_sequences(X_train_scaled, window=20)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, window=20)

# Build LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(20, X_train_seq.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)

# Evaluate
loss, mae = model.evaluate(X_test_seq, y_test_seq)
print(f"Loss: {loss:.4f}, MAE: {mae:.4f}")
```

## Backtesting

### Strategy Definition

```python
from src.backtesting.strategy import Strategy

class SimpleMovingAverageCrossover(Strategy):
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        data['SMA_short'] = data['Close'].rolling(self.short_window).mean()
        data['SMA_long'] = data['Close'].rolling(self.long_window).mean()
        data['signal'] = 0
        data.loc[data['SMA_short'] > data['SMA_long'], 'signal'] = 1
        data.loc[data['SMA_short'] <= data['SMA_long'], 'signal'] = -1
        return data
```

### Backtest Execution

```python
from src.backtesting.backtester import Backtester

strategy = SimpleMovingAverageCrossover()
backtester = Backtester(
    strategy=strategy,
    initial_capital=100000,
    commission=0.001  # 0.1% commission
)

results = backtester.run(data)
print(results.summary())

# Performance metrics
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
```

## Validation & Model Evaluation

### Cross-Validation (Time-Series)

```python
from src.models.validation import TimeSeriesSplit

# Time-series aware cross-validation
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold Score: {score:.4f}")
```

### Metrics & Reporting

```python
from src.models.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
    calculate_directional_accuracy
)

# Regression metrics
rmse = calculate_rmse(y_test, y_pred)
mae = calculate_mae(y_test, y_pred)
r2 = calculate_r2(y_test, y_pred)

# Trading-specific metrics
direction_accuracy = calculate_directional_accuracy(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"Direction Accuracy: {direction_accuracy:.2%}")
```

## Logging & Monitoring

```python
import logging
from src.config.logging import setup_logging

# Setup logging
logger = setup_logging('trading_ml', log_level=logging.INFO)

logger.info("Starting model training...")
logger.debug(f"Data shape: {data.shape}")
logger.warning("Low sample size: less than 1000 observations")
logger.error("Failed to fetch data from API")
```

## Testing Standards

### Unit Tests

```python
import pytest
from src.analysis.indicators import SMA

class TestSMA:
    def test_sma_calculation(self):
        """Test SMA calculation with known values"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = SMA(data, window=3)
        expected = [None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        assert result.tolist() == expected
    
    def test_sma_with_nans(self):
        """Test SMA handles NaN values correctly"""
        import numpy as np
        data = [1, np.nan, 3, 4, 5]
        result = SMA(data, window=2)
        assert np.isnan(result[0])
```

### Integration Tests

```python
@pytest.fixture
def sample_data():
    """Fixture providing sample stock data"""
    return load_stock_data('AAPL', '2023-01-01', '2023-12-31')

def test_full_pipeline(sample_data):
    """Test complete data pipeline"""
    cleaned = clean_data(sample_data)
    featured = create_features(cleaned)
    model = train_model(featured)
    predictions = model.predict(featured)
    
    assert predictions is not None
    assert len(predictions) == len(featured)
```

## Documentation

- Document data sources and data quality issues
- Explain model assumptions and limitations
- Include methodology and hyperparameter choices
- Report all performance metrics and confidence intervals
- Use Jupyter notebooks for exploratory analysis (not production code)
- Add docstrings to all functions with parameter descriptions

## Configuration

Use environment variables and config files:

```python
# .env
STOCK_SYMBOLS=AAPL,MSFT,GOOGL
DATA_START_DATE=2020-01-01
MODEL_HYPERPARAMS_FILE=configs/model_config.yaml
RANDOM_SEED=42
```

```python
# src/config/settings.py
import os
from pathlib import Path

DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', 0.8))
```

<!-- © Capgemini 2025 -->
