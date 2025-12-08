# ML Architecture & Library Selection Plan
**XAU/USD Trading Signal System - Phase 1**

**Document Version**: 1.0  
**Date Created**: December 5, 2025  
**Focus**: Model Selection, Library Stack, Training Strategy  
**Target Accuracy**: P(TP) ≥ 70%  

---

## 1. Executive Summary

This document provides **comprehensive ML architecture** tailored for:
- **Data Characteristics**: 20+ years XAU/USD 1-minute OHLCV (highly liquid forex data)
- **Problem Type**: Binary classification (signal/no-signal)
- **Data Size**: ~20+ million rows (manageable with batch processing)
- **Goal**: Signal accuracy ≥70% with interpretability

**Three-Tier Approach:**
1. **Tier 1 (Classical ML)**: XGBoost, LightGBM - Fast, interpretable, production-ready
2. **Tier 2 (Ensemble)**: Blend models for robustness
3. **Tier 3 (Deep Learning)**: LSTM/Transformer for temporal patterns (optional Phase 2)

---

## 2. Data Characteristics & Implications

### 2.1 Data Profile (Verified - December 2025)

**Storage Location**: `ml/src/data/XAU_1m_data_*.csv` (22 files)

| Characteristic | Value | Implication |
|---|---|---|
| **Time Period** | 22 years (2004-2025) | Complete market cycles; multiple regimes |
| **Granularity** | 1-minute OHLCV (5-minute planned) | High frequency; XAU/USD tick data |
| **Total Samples** | ~5.3M+ 1-min candles | Manageable; batch processing recommended |
| **Data Storage** | 327.2 MB (22 CSV files) | Fits in memory; fast with Polars |
| **Price Range** | $384 (2004) → $2625 (2025) | High volatility across regimes |
| **Liquidity** | Very High (OTC forex 23/5) | Minimal gaps; consistent 1-min bars |
| **Seasonality** | Yes (trading hours: 07:00-23:00 CET) | Include time-of-day features |
| **Trend Regimes** | Multiple (bull/bear/range) | Need regime-aware features |
| **Volatility** | Variable (observed ATR ~20-40 pips) | Include volatility normalization |

**Data Format**:
```
Date;Open;High;Low;Close;Volume
2004.06.11 07:18;384;384.1;384;384;3
2004.06.11 07:23;384.1;384.1;384;384;2
...
2025.01.02 01:00;2624.48;2625.06;2624.41;2624.83;49
2025.01.02 01:01;2624.86;2624.99;2624.78;2624.84;56
```

### 2.2 Key Considerations for ML

✅ **Advantages of Your Data:**
- **22 years of history** → sufficient for training/testing across multiple market cycles (2008 crisis, COVID, etc.)
- **Very liquid XAU/USD** → minimal gaps and outliers; consistent 1-minute bars
- **327.2 MB total** → fits in memory; fast loading with Polars
- **Clear trading hours** (07:00-23:00 CET) → predictable seasonality patterns
- **Known data quality** → verified structure: Date;Open;High;Low;Close;Volume

⚠️ **Challenges & Mitigations:**
- **Class Imbalance**: Only ~30% of candles expected as valid signals → use stratified sampling + scale_pos_weight
- **Non-Stationary**: Market regime changes (2008 vs 2025) → walk-forward validation (re-train yearly)
- **Overfitting Risk**: 5.3M samples but fewer true signals → L1/L2 regularization + early stopping
- **Data Leakage**: Future prices cannot influence past features → strict chronological splits (no shuffling)
- **Feature Stationarity**: XAU/USD price changed 6.8x over 22 years → normalize features within rolling windows
- **Volume Sparsity**: Low volume (30-60 contracts/min) → use price-based features primarily

---

## 3. Recommended Library Stack

### 3.1 Core Data & Preprocessing
```
pandas>=2.0.0           # Data manipulation (essential)
numpy>=1.24.0           # Numerical operations
polars>=0.18.0          # Optional: faster alternative for large datasets
pyarrow>=10.0.0         # Fast I/O, Arrow format for large files
```

**Why This Stack:**
- `pandas`: Industry standard, proven for financial data
- `numpy`: Fast operations, integrates with sklearn/XGBoost
- `polars`: 10-100x faster than pandas for large datasets (20M rows)
- `pyarrow`: Efficient storage format, reduces memory by 50%

### 3.2 Feature Engineering & Technical Analysis
```
ta-lib>=0.4.28          # Technical Analysis Library (C-wrapped, fastest)
pandas-ta>=0.3.14b      # Pure Python alternative (more flexible)
statsmodels>=0.14.0     # Time-series features (autocorrelation, ARIMA)
scikit-learn>=1.3.0     # Preprocessing (scaling, encoding)
scipy>=1.10.0           # Scientific computing (wavelets, filtering)
```

**Why This Stack:**
- `ta-lib`: Industry standard, optimized C code (10-50x faster than Python)
- `pandas-ta`: Backup/flexibility; better maintainability
- `statsmodels`: ACF/PACF, decomposition, seasonal patterns
- `scikit-learn`: Scalers, preprocessors, feature selection
- `scipy`: Advanced signal processing (wavelets for trend decomposition)

### 3.3 Model Training & Hyperparameter Optimization
```
xgboost>=1.7.0          # Primary model (fast, interpretable, production-ready)
lightgbm>=3.3.0         # Fast alternative; handles missing values well
catboost>=1.1.0         # Categorical features + overfitting protection
scikit-learn>=1.3.0     # Ensemble methods (voting, stacking)
optuna>=3.0.0           # Bayesian hyperparameter optimization
ray-tune>=2.5.0         # Distributed hyperparameter tuning (optional)
```

**Why This Stack:**
- **XGBoost**: Best for mixed feature types; proven in trading; fast inference
- **LightGBM**: 2-10x faster training; lower memory; good for large datasets
- **CatBoost**: Built-in overfitting protection; handles categorical features natively
- **Optuna**: Efficient Bayesian search; better than grid search for high-dimensional spaces
- **Ray Tune**: Distributed tuning; useful for parallel training across multiple models

### 3.4 Model Evaluation & Validation
```
scikit-learn>=1.3.0     # Metrics, cross-validation, GridSearch
imbalanced-learn>=0.10.0 # Handle class imbalance (SMOTE, stratified splits)
shap>=0.42.0            # Model interpretability (SHAP values)
eli5>=0.11.0            # Feature importance visualization
```

**Why This Stack:**
- `scikit-learn`: Standard metrics (precision, recall, F1, ROC-AUC)
- `imbalanced-learn`: Stratified K-Fold, SMOTE (only for train, not for CV)
- `shap`: Explainable AI; understand model decisions
- `eli5`: Feature importance; debug model behavior

### 3.5 Time-Series Specific
```
tsfresh>=0.19.0         # Automated feature extraction from time-series
tslearn>=1.0.0          # Time-series specific models (optional)
pmdarima>=2.0.0         # Auto ARIMA (for baseline comparison)
```

**Why This Stack:**
- `tsfresh`: Auto-extract 100+ time-series features (complementary to manual engineering)
- `tslearn`: Dynamic Time Warping, time-series clustering (Phase 2)
- `pmdarima`: ARIMA baseline (to compare with ML models)

### 3.6 Deep Learning (Phase 2)
```
tensorflow>=2.13.0      # Full-featured deep learning
keras>=2.13.0           # High-level API
pytorch>=2.0.0          # Research-friendly alternative (optional)
pytorch-lightning>=2.0.0 # Pytorch trainer abstraction
```

**Note**: Start with classical ML (XGBoost). Add deep learning only if needed for Phase 2.

### 3.7 Development & Monitoring
```
jupyter>=1.0.0          # Notebooks for EDA
matplotlib>=3.7.0       # Visualization
seaborn>=0.12.0         # Statistical plots
plotly>=5.13.0          # Interactive plots
wandb>=0.14.0           # Experiment tracking (optional)
mlflow>=2.5.0           # Model versioning & serving
```

---

## 4. Recommended Model Architecture

### 4.1 Model Selection Rationale

**Primary Model: XGBoost**

| Aspect | Rating | Justification |
|--------|--------|---|
| **Speed** | ⭐⭐⭐⭐⭐ | 1-5 million samples train in minutes |
| **Accuracy** | ⭐⭐⭐⭐⭐ | Proven for trading signals; handles mixed features |
| **Interpretability** | ⭐⭐⭐⭐ | Feature importance, SHAP values available |
| **Scalability** | ⭐⭐⭐⭐⭐ | Handles 20M rows with proper batching |
| **Production Ready** | ⭐⭐⭐⭐⭐ | ONNX export, fast inference, low latency |
| **Hyperparameter Tuning** | ⭐⭐⭐⭐ | Well-understood; many guides available |
| **Class Imbalance** | ⭐⭐⭐⭐ | Good with `scale_pos_weight` parameter |

**Secondary Model: LightGBM (Ensemble)**

- 2-10x faster training than XGBoost
- Better for large datasets (20M rows)
- Lower memory footprint
- Handles missing values naturally

**Ensemble Strategy: Voting Classifier**
- Combine XGBoost + LightGBM predictions
- Average probabilities for robustness
- Reduces overfitting by 5-10%

### 4.2 Model Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│               INPUT: Feature Matrix (200+ features)      │
│  From ml/src/data/feature_engineer.py (Phase 1 output)  │
└────────────────────────┬─────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐      ┌─────────┐      ┌──────────┐
   │ XGBoost │      │LightGBM │      │ Logistic │
   │ Clf-1   │      │ Clf-2   │      │ Regr     │
   │ (trees) │      │(leaves) │      │(baseline)│
   └────┬────┘      └────┬────┘      └────┬─────┘
        │                │                 │
        │ Proba          │ Proba           │ Proba
        │                │                 │
        └────────────────┼─────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Voting Classifier     │
            │  avg(predictions)      │
            │  threshold = 0.50      │
            └────────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Signal Generated?   │
              │  (if proba ≥ 0.50)  │
              │  + Validation Logic  │
              └──────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Trading Signal Output │
            │  - Entry level         │
            │  - SL / TP targets     │
            │  - Confidence score    │
            │  - Feature importance  │
            └────────────────────────┘
```

### 4.3 Model Parameters (Recommended Starting Points)

**XGBoost Classifier:**
```python
xgb_params = {
    'max_depth': 6,              # Shallow trees prevent overfitting
    'learning_rate': 0.1,        # Conservative learning
    'n_estimators': 500,         # Will use early stopping
    'subsample': 0.8,            # Stochastic gradient boosting
    'colsample_bytree': 0.8,     # Feature subsampling
    'colsample_bylevel': 0.8,    # Level-wise subsampling
    'min_child_weight': 1,       # Prevent shallow splits
    'reg_alpha': 0.1,            # L1 regularization (feature selection)
    'reg_lambda': 1.0,           # L2 regularization
    'scale_pos_weight': 2.5,     # Adjust for class imbalance (~30% signals)
    'random_state': 42,          # Reproducibility
    'n_jobs': -1,                # Parallel processing
}
```

**LightGBM Classifier:**
```python
lgb_params = {
    'max_depth': 5,              # Even shallower than XGB
    'learning_rate': 0.1,
    'n_estimators': 500,
    'num_leaves': 31,            # Controls tree complexity
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_data_in_leaf': 20,      # Prevent overfitting
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'scale_pos_weight': 2.5,     # Class weight
    'random_state': 42,
    'n_jobs': -1,
}
```

**Voting Classifier:**
```python
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
    ],
    voting='soft',               # Use predicted probabilities
    weights=[1, 1],              # Equal weight
)
```

---

## 5. Training & Validation Strategy

### 5.1 Walk-Forward Validation (Recommended for Time-Series)

**Why Not Random Split?**
- ❌ Random K-Fold causes **data leakage** (future prices influence past predictions)
- ❌ Market regime changes → model trained on 2004 may not work for 2025
- ✅ Walk-Forward preserves **temporal order** and detects model drift

**Walk-Forward Process:**

```
Train Period: 2004-2016 (12 years) → Test: 2017 (1 year)
Train Period: 2004-2017 (13 years) → Test: 2018 (1 year)
Train Period: 2004-2018 (14 years) → Test: 2019 (1 year)
...
Train Period: 2004-2023 (19 years) → Test: 2024 (1 year)
Train Period: 2004-2024 (20 years) → Test: 2025 (1 year) ← Latest
```

**Validation Within Train Period:**
- Use **Time-Series Split** from scikit-learn
- 3-5 folds with expanding window
- Early stopping based on fold average metrics

### 5.2 Class Imbalance Handling

**Problem**: Only ~30% of candles are valid trading signals

**Solution Strategy (In Order):**

1. **Stratified K-Fold** (Primary)
   ```python
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
   ```
   - Ensures each fold has ~30% signals
   - No data leakage with shuffle=False

2. **Scale Pos Weight** (In Model)
   ```python
   scale_pos_weight = (neg_samples / pos_samples)  # ~3.33 for 30% signals
   ```
   - Built-in to XGBoost/LightGBM
   - Penalizes misclassification of signals

3. **Threshold Tuning** (Post-Training)
   ```python
   # Instead of default 0.50, find optimal threshold
   thresholds = np.arange(0.40, 0.70, 0.05)
   for threshold in thresholds:
       predictions = (y_proba >= threshold).astype(int)
       f1 = f1_score(y_true, predictions)
       # Plot F1 vs threshold, select best
   ```
   - Typically optimal threshold is 0.45-0.55 for trading

4. **SMOTE** (Optional, Use Carefully)
   ```python
   from imblearn.over_sampling import SMOTE
   # Only apply to TRAIN set, NOT to validation/test
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
   ```
   - Oversample minority class synthetically
   - Can help but may introduce artifacts
   - Use with caution for financial data

**Recommended: Use (1) + (2) + (3). Avoid SMOTE for trading data.**

### 5.3 Train/Val/Test Split with Walk-Forward

```python
# Phase 1: Develop model (2004-2023)
# - Train: 2004-2022 (70% = 14 years)
# - Val:   2023 (15% = 1 year, for early stopping)
# - Test:  2024 (15% = 1 year, final evaluation)

# Phase 2: Production (2025+)
# - Retrain monthly on 2004-2024
# - Deploy on live 2025 data
# - Monitor for drift
```

**Implementation:**
```python
train_end = pd.Timestamp('2022-12-31')
val_end = pd.Timestamp('2023-12-31')
test_end = pd.Timestamp('2024-12-31')

X_train = X[X.index <= train_end]
X_val = X[(X.index > train_end) & (X.index <= val_end)]
X_test = X[(X.index > val_end) & (X.index <= test_end)]

y_train = y[y.index <= train_end]
y_val = y[(y.index > train_end) & (y.index <= val_end)]
y_test = y[(y.index > val_end) & (y.index <= test_end)]
```

---

## 6. Feature Selection & Engineering

### 6.1 Feature Count Strategy

**Challenge**: 200+ features → risk of overfitting

**Solution: Hierarchical Feature Selection**

```
Phase 1: Domain Expert Selection
  ├─ Momentum (5 features): RSI, MACD, Stochastic, Momentum, ROC
  ├─ Volatility (4 features): ATR, Bollinger Bands std, Realized Vol
  ├─ Trend (4 features): SMA, EMA, ADX, Aroon
  ├─ Volume (3 features): OBV, VWAP, MFI
  ├─ Time (4 features): Hour, Day, Trading Hours, Session
  └─ Lags (10 features): Close[t-1], Close[t-5], Close[t-20], etc.
     TOTAL: ~30-40 core features

Phase 2: Automated Feature Selection
  ├─ Mutual Information: Keep top 50 features by MI score
  ├─ Permutation Importance: Remove features with low importance
  ├─ Correlation Matrix: Remove redundant features (r > 0.95)
  └─ SHAP Analysis: Understand feature contributions
     RESULT: ~50-80 features

Phase 3: Model-Based Selection
  ├─ XGBoost feature importance
  ├─ Recursive Feature Elimination (RFE)
  ├─ L1 regularization (Lasso)
  └─ Keep only top 50 features for final model
```

**Recommended**: Start with 40 core features, expand to 80-100 based on validation metrics.

### 6.2 Feature Importance Analysis

**Post-Training Insights:**

```python
# XGBoost feature importance
feature_importance = model.get_booster().get_score()

# SHAP values (explains each prediction)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Permutation importance
from sklearn.inspection import permutation_importance
importance = permutation_importance(model, X_test, y_test, n_repeats=10)

# Identify which features matter most for signals
top_features = feature_importance.head(20)
print(f"Top 20 features explain ~80% of model decisions")
```

---

## 7. Performance Metrics & Success Criteria

### 7.1 Primary Metrics

| Metric | Target | Why | Formula |
|--------|--------|-----|---------|
| **Precision** | ≥ 70% | Minimize false signals | TP / (TP + FP) |
| **Recall** | ≥ 50% | Don't miss opportunities | TP / (TP + FN) |
| **F1 Score** | ≥ 60% | Balance precision & recall | 2 × (P×R)/(P+R) |
| **ROC-AUC** | ≥ 0.75 | Discrimination ability | Area under ROC |
| **PR-AUC** | ≥ 0.65 | For imbalanced data | Area under PR curve |
| **MCC** | ≥ 0.40 | Matthews Correlation Coeff | Balanced metric |

### 7.2 Business Metrics

| Metric | Target | Calculation |
|--------|--------|---|
| **Win Rate (WR)** | ≥ 70% | (TP) / (TP + FP) |
| **Sharpe Ratio** | ≥ 2.0 | (Return - Rf) / StdDev |
| **Maximum Drawdown** | ≤ 15% | Max peak-to-trough decline |
| **Profit Factor** | ≥ 2.0 | (Gross Profit) / (Gross Loss) |
| **Expectancy** | > 0 | (WR × AvgWin) - ((1-WR) × AvgLoss) |

### 7.3 Monitoring Metrics (Production)

```python
# Track in real-time
daily_metrics = {
    'signals_generated': count,
    'precision_today': TP / (TP + FP),
    'precision_7day': rolling_precision,
    'precision_30day': rolling_precision,
    'model_drift': compare_feature_distributions(),
    'data_quality': check_nan_ratio(),
}

# Alert if:
if precision_7day < 0.65:  # Degradation alert
    send_alert("Model precision dropping")
if model_drift > threshold:
    send_alert("Market regime change detected")
```

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Classical ML (4-6 weeks)

**Week 1-2: EDA & Feature Engineering**
- Load 22 years of data (~5.3M rows, 327MB) using Polars for speed
- Parse Date format: `YYYY.MM.DD HH:MM` → datetime
- Analyze OHLCV distributions across time periods
- Check for gaps/anomalies during trading hours
- Create 40-50 core features from raw OHLCV
- Aggregate 1-minute data to 5-minute candles (required for signals)
- Data quality validation

**Week 3: Model Training**
- Train XGBoost baseline on 2004-2023 data
- Train LightGBM alternative for comparison
- Hyperparameter tuning (Optuna) on 2004-2022 train + 2023 val
- Walk-forward validation on multiple years
- Early stopping on 2023 validation set

**Week 4: Model Evaluation**
- Test on 2024 (latest full year)
- Analyze feature importance
- SHAP value analysis
- Error analysis by market regime
- Threshold tuning for optimal precision/recall

**Week 5-6: Production Prep**
- Model serialization (pickle, ONNX)
- Inference optimization (< 100ms latency)
- Documentation of data pipeline
- Deployment scripts
- Setup monitoring for 2025 live data

### 8.2 Phase 2: Ensemble & Monitoring (Weeks 7-10)

- Combine XGBoost + LightGBM
- Add drift detection
- Implement retraining pipeline
- Real-time monitoring dashboard

### 8.3 Phase 3: Deep Learning (Optional, Weeks 11-16)

- LSTM for temporal patterns
- Transformer model
- Ensemble with classical ML
- Production deployment

---

## 9. Complete Requirements.txt for ML

```
# Core Data Processing
pandas>=2.0.0
numpy>=1.24.0
polars>=0.18.0           # Fast alternative to pandas
pyarrow>=10.0.0          # Efficient I/O

# Feature Engineering
ta-lib>=0.4.28           # Technical Analysis (fastest)
pandas-ta>=0.3.14b       # Backup TA library
statsmodels>=0.14.0      # Time-series analysis
scikit-learn>=1.3.0      # ML utilities
scipy>=1.10.0            # Scientific computing
tsfresh>=0.19.0          # Auto TS feature extraction

# Model Training
xgboost>=1.7.0           # Primary model
lightgbm>=3.3.0          # Secondary model
catboost>=1.1.0          # Optional ensemble
optuna>=3.0.0            # Hyperparameter optimization

# Class Imbalance
imbalanced-learn>=0.10.0 # SMOTE, stratified splits

# Model Interpretability
shap>=0.42.0             # SHAP values
eli5>=0.11.0             # Feature importance

# Visualization & Monitoring
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.13.0
wandb>=0.14.0            # Experiment tracking (optional)
mlflow>=2.5.0            # Model registry

# Development
jupyter>=1.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0

# Production
joblib>=1.3.0            # Model serialization
onnx>=1.14.0             # ONNX export for inference
onnxruntime>=1.15.0      # Fast ONNX inference
```

---

## 10. ML Project Structure

```
ml/
├── src/
│   ├── data/
│   │   ├── loader.py              # Load CSV (Phase 1 output)
│   │   ├── preprocessor.py        # Clean & resample (Phase 1)
│   │   ├── feature_engineer.py    # 200+ features (Phase 1)
│   │   ├── splitter.py            # Train/val/test (Phase 1)
│   │   └── validators.py          # Feature quality (Phase 1)
│   │
│   ├── models/
│   │   ├── xgboost_model.py       # XGBoost classifier
│   │   ├── lightgbm_model.py      # LightGBM classifier
│   │   ├── ensemble.py            # Voting ensemble
│   │   ├── trainer.py             # Training loop
│   │   └── validator.py           # Cross-validation
│   │
│   ├── analysis/
│   │   ├── feature_importance.py  # Feature analysis
│   │   ├── shap_analysis.py       # SHAP interpretability
│   │   ├── drift_detection.py     # Model monitoring
│   │   └── backtest.py            # Trading simulation
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py   # End-to-end training
│   │   ├── inference_pipeline.py  # Real-time inference
│   │   └── retraining_pipeline.py # Automated retraining
│   │
│   ├── config/
│   │   ├── defaults.yaml
│   │   ├── feature_config.yaml
│   │   ├── model_config.yaml      # Model hyperparameters
│   │   └── training_config.yaml   # Training strategy
│   │
│   └── utils/
│       ├── metrics.py             # Custom metrics
│       ├── logger.py              # Logging setup
│       └── helpers.py             # Utility functions
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_production_prep.ipynb
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_inference.py
│   └── fixtures/
│
├── models/
│   ├── xgboost_v1.pkl             # Saved models
│   ├── lgb_v1.pkl
│   ├── ensemble_v1.pkl
│   └── metadata.json              # Model versions
│
├── logs/
│   ├── training.log
│   ├── inference.log
│   └── monitoring.log
│
└── main.py                        # Entry point
```

---

## 11. Key Design Decisions Explained

### Why XGBoost?
✅ Fast training (millions of samples in minutes)  
✅ Excellent accuracy (often beats neural nets)  
✅ Interpretable (feature importance, SHAP)  
✅ Production-ready (low latency inference)  
✅ Handles mixed features naturally  
✅ Proven in trading (used by most quant firms)  

### Why Not Deep Learning First?
❌ Requires 10x more data (we have enough but simpler is better)  
❌ Slower inference (not ideal for real-time signals)  
❌ Black-box (less interpretable for trading)  
❌ More hyperparameters (harder to tune)  

**Recommendation**: Start with XGBoost, add LSTM only if XGB doesn't hit 70% target.

### Why Walk-Forward Validation?
❌ Random K-Fold causes data leakage (future prices used to predict past)  
❌ Market changes over time (model trained on 2004 won't work for 2025)  
✅ Walk-forward respects temporal order  
✅ Detects model drift naturally  
✅ Simulates real-world deployment  

### Why Ensemble (XGB + LGB)?
✅ Reduces overfitting by 5-10%  
✅ More robust to market changes  
✅ Combines strengths of both models  
✅ Simple averaging → no additional complexity  

### Why Stratified K-Fold?
❌ Random split → imbalanced folds (some have 25% signals, others 35%)  
✅ Stratified → all folds have ~30% signals  
✅ More stable cross-validation scores  
✅ Better hyperparameter tuning  

---

## 12. Success Checklist

**Before Training:**
- [ ] 200+ features engineered and validated
- [ ] Class imbalance strategy defined (scale_pos_weight calculated)
- [ ] Train/val/test splits created (temporal, no leakage)
- [ ] Stratified K-Fold configured for CV
- [ ] Feature correlation checked (remove r > 0.95)

**During Training:**
- [ ] Early stopping implemented (monitor val metric)
- [ ] Hyperparameter tuning (Optuna) completed
- [ ] Feature importance analyzed
- [ ] SHAP values computed for explainability
- [ ] Model drift detection setup

**After Training:**
- [ ] Test set performance ≥ 70% precision
- [ ] Feature importance makes business sense
- [ ] Walk-forward validation shows consistent performance
- [ ] Threshold tuning optimizes for F1 or custom metric
- [ ] Model serialized and inference time < 100ms

**Production Ready:**
- [ ] Model versioning in place
- [ ] Retraining pipeline automated (monthly)
- [ ] Monitoring dashboard live
- [ ] Drift detection alerts configured
- [ ] Fallback model in case of degradation

---

## 13. Next Steps (Immediate Actions)

1. **Prepare Data** (Week 1)
   - Move CSV files to `ml/data/`
   - Run Phase 1 data pipeline
   - Generate 200+ features
   - Create train/val/test splits

2. **Setup Environment** (Day 1)
   - Create `requirements_ml.txt` with libraries above
   - Install `ta-lib` (may need C compiler)
   - Test imports

3. **EDA Notebook** (Week 1)
   - Load sample data
   - Analyze OHLCV distributions
   - Check feature correlations
   - Plan feature engineering strategy

4. **XGBoost Baseline** (Week 2)
   - Train simple XGBoost model
   - Get baseline metrics
   - Analyze feature importance
   - Plan next improvements

5. **Hyperparameter Tuning** (Week 3)
   - Use Optuna for optimization
   - Walk-forward validation
   - Threshold tuning
   - Documentation

---

## 14. References & Resources

**XGBoost:**
- Official Docs: https://xgboost.readthedocs.io/
- Parameter Tuning Guide: https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
- Trading Applications: Case studies from Kaggle

**LightGBM:**
- Official Docs: https://lightgbm.readthedocs.io/
- Parameter Reference: https://lightgbm.readthedocs.io/en/latest/Parameters.html

**Feature Engineering:**
- "Feature Engineering for Machine Learning" by Alice Zheng
- SHAP docs: https://shap.readthedocs.io/
- Time-series features: Statsmodels documentation

**Time-Series Validation:**
- "Advances in Financial Machine Learning" by Marcos López de Prado
- Walk-forward analysis: Chapter 4
- Position-aware metrics for trading

**Interpretability:**
- SHAP Paper: https://arxiv.org/abs/1705.07874
- Molnar's "Interpretable Machine Learning": https://christophmolnar.com/books/iml/

---

## Approval & Sign-off

**Plan Author**: Python/ML Planner Agent  
**Date Created**: December 5, 2025  
**Status**: 🔄 Ready for Review  

**Required Approvals:**
- [ ] ML/Data Science Lead
- [ ] Data Engineer
- [ ] DevOps (for infrastructure)

**Approval Signature**: ________________________ Date: __________

---

**Next Document**: After approval, create detailed `ML Model Training Plan` with:
- Exact Optuna search space
- XGBoost parameter tuning steps
- Walk-forward validation implementation
- SHAP analysis workflow
- Production inference optimization

