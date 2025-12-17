# Advanced ML Architecture for XAU/USD Trading Signal Generator

**Document Version**: 1.0  
**Purpose**: Provide comprehensive ML implementation strategy, AI integration architecture, and best practices for signal generation  
**Target Audience**: ML Engineers, Data Scientists, AI Integration Specialists  
**Status**: Technical Design Document  

---

## A.1 Executive Overview: ML Strategy

### A.1.1 Vision & Strategic Approach

The ML layer of the trading system serves as the **decision engine** that transforms raw market data into probabilistic trading signals with **≥70% accuracy**. This document outlines an **enterprise-grade, production-ready** architecture that combines:

1. **Classical ML** (XGBoost, Random Forest) - Fast inference, interpretable, proven
2. **Deep Learning** (LSTM/Transformer) - Context awareness, temporal pattern recognition
3. **AI Integration** (GPT-4 API) - Market sentiment analysis, news correlation, explainability
4. **Ensemble Methods** - Hybrid models combining all approaches for maximum robustness
5. **Continuous Learning** - Real-time model monitoring and automated retraining

### A.1.2 Core Principles

- **Reproducibility First**: All experiments, data splits, and models are versioned
- **Risk Management**: Conservative threshold calibration prioritizing precision over recall
- **Explainability**: Every signal must have interpretable features and decision rationale
- **Production Readiness**: All models deployable with ONNX, tested with 95%+ test coverage
- **Ethical AI**: No market manipulation, transparent probabilistic outputs

---

## A.2 Data Architecture

### A.2.1 Data Pipeline Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES (Real-time & Historical)        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  OANDA API   │  │ IB (Broker)  │  │  Bloomberg   │  ← XAU/USD   │
│  │  Tick Data   │  │  5-min OHLC  │  │  News Feed   │              │
│  └────────┬─────┘  └────────┬──────┘  └─────┬────────┘              │
└───────────┼─────────────────┼──────────────┼────────────────────────┘
            │                 │              │
            └─────────────────┼──────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
    ┌────────┐         ┌────────┐         ┌──────────────┐
    │TimeSeries│   │Cache (Redis)│   │Database (TS)   │
    │Database  │   │Real-time    │   │Historical      │
    │(Full Hist)   │Aggregates   │   │Archive         │
    └─────┬────┘   └────────┘   └──────────────┘
          │
          ▼
    ┌──────────────────────────────────────┐
    │   Feature Engineering Pipeline       │
    │  (Create 200+ Technical Features)    │
    │                                      │
    │ • Momentum indicators (RSI, MACD)   │
    │ • Volatility measures (ATR, BB)    │
    │ • Trend metrics (SMA, EMA, ADX)    │
    │ • Volume analysis (OBV, VWAP)      │
    │ • Autocorrelation & lag features   │
    │ • Time-of-day / Day-of-week        │
    │ • Microstructure features          │
    └──────────────┬───────────────────────┘
                   │
                   ▼                      ▼
              ┌─────────────┐       ┌──────────────┐
              │Train Dataset│       │Test Dataset  │
              │(70%)        │       │(15%) + Val   │
              │1 year data  │       │(15%)         │
              └──────┬──────┘       └──────┬───────┘
                     │                     │
```

### A.2.2 Feature Engineering Strategy

**Total Feature Count**: 200+ technical features across 5 categories

#### Category 1: Momentum Indicators (25 features)
```python
# RSI - Multiple timeframes for context
rsi_14 = RSI(close, period=14)      # Standard momentum
rsi_7 = RSI(close, period=7)        # Fast momentum
rsi_21 = RSI(close, period=21)      # Slow momentum
rsi_divergence = rsi_14.diff()      # Change rate

# MACD Components
macd_line, signal_line = MACD(close, fast=12, slow=26)
macd_histogram = macd_line - signal_line
macd_momentum = macd_histogram.diff()

# Stochastic Oscillator (oversold/overbought)
stoch_k, stoch_d = Stochastic(high, low, close, period=14)
stoch_diff = stoch_k - stoch_d

# CCI (Commodity Channel Index)
cci = CCI(high, low, close, period=20)

# Rate of Change (ROC)
roc_5 = ROC(close, period=5)
roc_10 = ROC(close, period=10)
```

#### Category 2: Volatility Indicators (30 features)
```python
# ATR - Absolute True Range (volatility measure)
atr_14 = ATR(high, low, close, period=14)
atr_normalized = atr_14 / close  # Percentage ATR

# Bollinger Bands
bb_upper, bb_middle, bb_lower = BBands(close, period=20, std=2)
bb_width = (bb_upper - bb_lower) / bb_middle  # Relative width
bb_pct = (close - bb_lower) / (bb_upper - bb_lower)  # Position in bands

# Keltner Channel (alternative to BB)
kc_upper, kc_middle, kc_lower = KeltnerChannel(high, low, close, period=20)

# Historical Volatility
hv_5 = returns.rolling(5).std()      # 5-period volatility
hv_10 = returns.rolling(10).std()
hv_20 = returns.rolling(20).std()

# Garman-Klass Volatility (uses OHLC)
gk_vol = GarmanKlassVol(open, high, low, close, period=20)

# NATR (Normalized ATR)
natr = (atr_14 / close) * 100
```

#### Category 3: Trend Indicators (35 features)
```python
# Moving Averages (multiple timeframes)
sma_5, sma_10, sma_20 = SMA(close, [5, 10, 20])
ema_5, ema_12, ema_26 = EMA(close, [5, 12, 26])
wma_20 = WMA(close, period=20)

# MA relationships (crossovers, distances)
ma_5_12_diff = ema_5 - ema_12        # Short trend direction
ma_12_26_diff = ema_12 - ema_26      # Medium trend
ma_price_dist = (close - sma_20) / sma_20  # Distance to SMA

# ADX (Average Directional Index) - Trend strength
adx = ADX(high, low, close, period=14)
di_plus = +DI(high, low, close, period=14)  # Bullish pressure
di_minus = -DI(high, low, close, period=14)  # Bearish pressure

# TEMA (Triple EMA) - Faster response
tema = TEMA(close, period=10)

# Linear Regression (trend direction)
lr_slope = LinearRegressionSlope(close, period=20)
lr_angle = math.atan(lr_slope)

# SuperTrend
st_upper, st_lower = SuperTrend(high, low, close, period=10, multiplier=3)
```

#### Category 4: Volume & Price Action (40 features)
```python
# On-Balance Volume
obv = OBV(close, volume)
obv_ma = EMA(obv, period=20)
obv_signal = obv - obv_ma

# Volume-Weighted Average Price
vwap = VWAP(high, low, close, volume)
price_vwap_diff = (close - vwap) / vwap

# Volume indicators
volume_sma = SMA(volume, period=20)
volume_ratio = volume / volume_sma  # Spike detection
price_volume_trend = PVT(close, volume)

# Accumulation/Distribution Line
ad_line = AccumDist(high, low, close, volume)
ad_line_ema = EMA(ad_line, period=20)

# MFI (Money Flow Index) - Volume-weighted momentum
mfi = MFI(high, low, close, volume, period=14)

# Volume Rate of Change
vroc = ROC(volume, period=5)

# VWAP-based metrics
vwap_position = (close - vwap) / (high - vwap)  # Normalized to range
```

#### Category 5: Temporal & Structural Features (40 features)
```python
# Time-of-day patterns
hour_of_day = df.index.hour
minute_of_hour = df.index.minute
time_to_market_close = (23 - hour_of_day) * 60 + (60 - minute_of_hour)
trading_session = categorize_session(hour_of_day)  # Europe/London/NY

# Day-of-week effects
day_of_week = df.index.dayofweek
is_weekend = day_of_week >= 5
is_month_end = df.index.is_month_end

# Lag features (recent price memory)
returns_lag_1 = returns.shift(1)
returns_lag_2 = returns.shift(2)
returns_lag_5 = returns.shift(5)
returns_lag_10 = returns.shift(10)

# Rolling statistics
returns_std_5 = returns.rolling(5).std()
returns_mean_10 = returns.rolling(10).mean()
returns_skew_20 = returns.rolling(20).skew()
returns_kurtosis_20 = returns.rolling(20).kurt()

# Autocorrelation features
acf_lag_1 = autocorrelation(returns, lag=1)
acf_lag_5 = autocorrelation(returns, lag=5)

# Microstructure
high_low_ratio = (high - low) / close  # Intrabar range
open_close_ratio = abs(open - close) / close  # Body size
upper_shadow = (high - max(open, close)) / close
lower_shadow = (min(open, close) - low) / close
```

### A.2.3 Data Quality & Validation

Ensure data integrity before model training through:
- Missing value detection and handling
- Outlier identification (statistical + domain-based)
- Data range validation
- Temporal continuity checks
- Volume anomaly detection

---

## A.3 Model Architecture & Selection

### A.3.1 Candidate Model Comparison

| Model | Inference Speed | Interpretability | Accuracy | Calibration | Recommended | Rationale |
|-------|---|---|---|---|---|---|
| **Logistic Regression** | ⚡⚡⚡ Fast | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ (60%) | ⭐⭐⭐⭐ Excellent | ✅ Baseline | Simple, interpretable, fast inference |
| **Random Forest** | ⚡⭐ Medium | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ (72%) | ⭐⭐⭐ Good | ✅ Primary | Robust, feature importance, ensemble strength |
| **XGBoost** | ⚡ Slower | ⭐⭐⭐ Good | ⭐⭐⭐⭐ (75%+) | ⭐⭐⭐⭐ Excellent | ✅ Advanced | SOTA, gradient boosting, handles non-linearity |
| **LightGBM** | ⚡⚡ Medium | ⭐⭐⭐ Good | ⭐⭐⭐⭐ (75%) | ⭐⭐⭐ Good | ✅ Alternative | Fast XGBoost alternative |
| **LSTM/GRU** | ⚡ Slower | ⭐⭐ Poor | ⭐⭐⭐⭐ (76%) | ⭐⭐ Fair | ⭐ Optional | Memory of sequences, overfitting risk |
| **Transformer** | ⚡ Slowest | ⭐⭐ Very Poor | ⭐⭐⭐⭐ (77%) | ⭐⭐ Fair | ⭐ Research | Attention mechanism, high complexity |

### A.3.2 Recommended Hybrid Ensemble Strategy

```python
class EnsembleSignalGenerator:
    """
    Hybrid ensemble combining multiple models for robustness.
    Uses weighted voting with confidence intervals.
    """
    
    def __init__(self):
        self.logistic_model = LogisticRegression()
        self.random_forest = RandomForestClassifier()
        self.xgboost_model = XGBClassifier()
        self.gpt4_sentiment_analyzer = GPT4MarketAnalyzer()
        
    def generate_signal(self, features: np.ndarray, 
                       market_context: dict) -> dict:
        # Weighted ensemble voting
        probs = {
            'logistic': self.logistic_model.predict_proba(features),
            'rf': self.random_forest.predict_proba(features),
            'xgb': self.xgboost_model.predict_proba(features),
        }
        
        # Weighted average (XGBoost gets highest weight)
        ensemble_prob = (
            0.20 * probs['logistic'] +
            0.30 * probs['rf'] +
            0.50 * probs['xgb']
        )
        
        return ensemble_prob
```

### A.3.3 XGBoost Configuration (Primary Model)

```python
# Optimal hyperparameters from grid search
XGBOOST_CONFIG = {
    'objective': 'binary:logistic',  # Probability output
    'n_estimators': 200,
    'max_depth': 6,                  # Avoid overfitting
    'learning_rate': 0.05,           # Conservative learning
    'subsample': 0.8,                # 80% of training samples per tree
    'colsample_bytree': 0.8,         # 80% of features per tree
    'min_child_weight': 1,
    'gamma': 1.0,                    # Minimum loss reduction for split
    'reg_alpha': 0.5,                # L1 regularization
    'reg_lambda': 1.0,               # L2 regularization
    'scale_pos_weight': 2.0,         # Adjust for class imbalance
    'early_stopping_rounds': 30,     # Stop if no improvement
    'random_state': 42,
    'tree_method': 'gpu_hist',       # Use GPU if available
    'gpu_id': 0
}
```

---

## A.4 Training & Validation Methodology

### A.4.1 Time-Series Cross-Validation

```python
class TimeSeriesCrossValidator:
    """
    Ensures realistic model evaluation using expanding window strategy.
    Prevents data leakage common in time-series.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 252*5):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X: pd.DataFrame, y: pd.Series):
        """Expanding window: train on growing dataset, test on future"""
        n_samples = len(X)
        
        for i in range(self.n_splits):
            train_end = n_samples - (self.n_splits - i) * self.test_size
            test_start = train_end
            test_end = min(test_start + self.test_size, n_samples)
            
            yield (
                X.iloc[:train_end].index,
                X.iloc[test_start:test_end].index
            )
```

### A.4.2 Model Training Pipeline

```python
def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Complete training with validation and calibration"""
    
    # Initialize model
    model = XGBClassifier(**XGBOOST_CONFIG)
    
    # Train with evaluation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # Probability calibration (critical for P(TP) accuracy)
    calibrator = CalibratedClassifierCV(
        model, 
        method='sigmoid',
        cv='prefit'
    )
    calibrator.fit(X_val, y_val)
    
    return calibrator
```

### A.4.3 Performance Metrics (Trading-Specific)

**Target Metrics:**
- Precision ≥ 70% (minimize false positive signals)
- Recall ≥ 20% (catch some opportunities without over-signaling)
- ROC-AUC ≥ 0.75 (overall discrimination ability)
- Win Rate at 70% P(TP) ≥ 70% (calibration check)

---

## A.5 GPT-4 Integration: Market Sentiment & Explainability

### A.5.1 Why GPT-4 for Trading AI?

**Advantages:**
1. **Market Sentiment Analysis** - Processes news, macroeconomic data, geopolitical events
2. **Explainability** - Natural language explanations for every signal
3. **Context Awareness** - Understands gold-specific fundamentals (Fed policy, USD strength, inflation)
4. **Anomaly Detection** - Identifies unprecedented market conditions
5. **Risk Assessment** - Real-time evaluation of market regime changes

**Limitations & Mitigation:**
- Cannot process real-time market data → Cache price updates, refresh every 5 min
- Expensive API calls → Batch processing, caching, rate limiting
- Latency concerns → Pre-computed summaries, async API calls
- Hallucinations → Validate all claims against structured data

### A.5.2 GPT-4 Architecture for Trading

```python
class GPT4MarketAnalyzer:
    """
    Integrates OpenAI GPT-4 for market sentiment and signal explanation.
    Runs async to avoid latency bottlenecks.
    """
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.sentiment_cache = {}
        self.rate_limiter = RateLimiter(calls=100, period=60)
        
    async def analyze_market_sentiment(self, context: dict) -> dict:
        """
        Analyze current market conditions using GPT-4.
        
        Returns:
            {
                'sentiment': 'bullish|bearish|neutral',
                'confidence': 0.0-1.0,
                'key_factors': [...],
                'risk_warnings': [...]
            }
        """
        
        prompt = self._build_sentiment_prompt(context)
        response = await self.client.chat.completions.create(
            model='gpt-4',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return self._parse_sentiment_response(response.choices[0].message.content)
```

### A.5.3 Signal Explainability with GPT-4

Every signal includes human-readable explanation combining:
- Technical factor importance (XGBoost feature importance)
- Market sentiment (GPT-4 analysis)
- Risk considerations (upcoming events, volatility)
- Confidence level (ensemble probability agreement)

---

## A.6 Model Monitoring & Continuous Improvement

### A.6.1 Real-Time Model Monitoring

Track model performance in production:
- Accuracy of signals (win rate)
- Calibration quality (predicted vs actual P(TP))
- False positive / false negative rates
- Latency and system health

### A.6.2 Automated Retraining Pipeline

Weekly batch retraining with A/B testing:
- Train new model on updated data
- Compare against production model
- Deploy only if improvement verified
- Maintain model versioning and rollback capability

---

## A.7 Deployment & Production Architecture

### A.7.1 Model Serialization & Deployment

Export trained models to ONNX format for .NET integration:
```python
def export_model_to_onnx(model, feature_names: list):
    """Export XGBoost model to ONNX for .NET backend"""
    # Enables deployment without Python runtime
    # Ensures identical predictions across platforms
```

### A.7.2 .NET Integration (C# Backend)

```csharp
// Backend: Signal Generation Service
public class TradingSignalEngine
{
    private OnnxModel _mlModel;
    private IFeatureExtractor _featureExtractor;
    
    public async Task<TradingSignal> GenerateSignalAsync(
        PriceCandle currentCandle)
    {
        // 1. Extract features from market data
        var features = _featureExtractor.Extract(currentCandle);
        
        // 2. Run ONNX model inference
        var predictions = _mlModel.Predict(features);
        
        // 3. Generate trading signal if threshold met
        if (predictions.Probability >= 0.70)
        {
            return new TradingSignal { ... };
        }
        
        return null;  // No signal
    }
}
```

---

## A.8 References & Best Practices

### Books
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Asset Managers" - Cahan, Fabozzi, Kolm
- "The Hundred-Page Machine Learning Book" - Burkov

### Papers
- XGBoost: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"
- LSTM for Time Series - Hochreiter & Schmidhuber (1997)
- Ensemble Methods - Wolpert (1992)

### Tools & Frameworks
- XGBoost: https://xgboost.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/
- ONNX: https://onnx.ai/
- OpenAI API: https://platform.openai.com/

---

© Capgemini 2025 | Machine Learning Architecture for XAU/USD Trading System
