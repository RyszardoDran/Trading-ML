# Sequence-Based Trading Model Pipeline

## ⚠️ CRITICAL: CONSTANT PARAMETERS

**DO NOT CHANGE THE FOLLOWING PARAMETERS:**
- **Take Profit (TP)**: 2.0 ATR
- **Stop Loss (SL)**: 1.0 ATR
- **Risk:Reward Ratio**: 1:2

These parameters define the "ground truth" of the strategy. Changing them to "improve" the model is cheating (data snooping) and leads to overfitting. The model must learn to win with these constraints.

## Overview

This pipeline implements a **sequence-based approach** for XAU/USD (gold) trading predictions. Unlike traditional point-in-time models, this approach uses **100 previous candles** to capture temporal patterns and market context before making a prediction.

### Key Concept: Win Rate

**Win Rate = Precision** - When the model predicts "BUY", what percentage of those predictions are correct?

- **High Win Rate (>70%)**: Model is conservative, fewer trades but higher accuracy
- **Medium Win Rate (50-70%)**: Balanced approach
- **Low Win Rate (<50%)**: Model needs improvement or market conditions changed

## Architecture

```
Input: 100 previous 1-minute candles (OHLCV)
   ↓
Per-candle feature engineering (13 features per candle)
   ↓
Flatten to 1300-dimensional vector (100 × 13)
   ↓
XGBoost Classifier (calibrated probabilities)
   ↓
Output: Probability + Win Rate estimate
```

### Why This Approach?

1. **Temporal Context**: Model sees market structure over last 100 minutes (~1.5 hours)
2. **Pattern Recognition**: Captures support/resistance, trends, volatility regimes
3. **Realistic**: Mimics how human traders analyze charts
4. **Robust**: Less sensitive to single-candle noise
5. **Win Rate Validation**: Direct measure of prediction accuracy on test data

## Files

```
ml/src/pipelines/
├── sequence_training_pipeline.py    # Main training pipeline
ml/src/scripts/
├── predict_sequence.py              # Prediction script for new data
ml/tests/
├── test_sequence_pipeline.py        # Comprehensive test suite
```

## Installation

Ensure dependencies are installed:

```bash
pip install -r ml/requirements_ml.txt
```

Required packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 1.7.0
- pytest >= 7.4.0 (for testing)

## Usage

### 1. Train the Model

Train with default parameters (100 candles, 5-minute horizon, 5bp threshold):

```bash
cd ml/src/pipelines
python sequence_training_pipeline.py
```

Train with custom parameters:

```bash
python sequence_training_pipeline.py \
    --window-size 50 \
    --horizon 10 \
    --min-return-bp 10.0 \
    --session london
```

**Parameters:**
- `--window-size`: Number of previous candles (default: 100)
- `--horizon`: Forward prediction horizon in minutes (default: 5)
- `--min-return-bp`: Minimum return threshold in basis points (default: 5.0)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--session`: Trading session filter (default: `london_ny`). Options: `london`, `ny`, `asian`, `london_ny`, `all`, `custom`.
- `--custom-start-hour`: Start hour (0-23) for `custom` session.
- `--custom-end-hour`: End hour (0-23) for `custom` session.
- `--max-windows`: Maximum number of windows to keep to avoid OOM (default: 200,000).

### Session Filtering

You can filter training data to specific trading sessions to specialize the model for certain market conditions.

**Available Sessions (UTC approx):**
- `london`: 08:00 - 16:00
- `ny`: 13:00 - 22:00
- `asian`: 00:00 - 09:00
- `london_ny`: 08:00 - 22:00 (Default - High liquidity)
- `all`: No filtering (Use all 24h data)
- `custom`: User-defined range

**Example: Train only on Asian session**
```bash
python sequence_training_pipeline.py --session asian
```

**Example: Train on custom hours (e.g., 10:00 - 14:00)**
```bash
python sequence_training_pipeline.py --session custom --custom-start-hour 10 --custom-end-hour 14
```

**Output:**
```
=== TRAINING COMPLETE - SEQUENCE PIPELINE ===
Window Size:       100 candles
Threshold:         0.45
WIN RATE:          0.6823 (68.23%)
Precision:         0.6823
Recall:            0.5421
F1 Score:          0.6042
ROC-AUC:           0.7891
PR-AUC:            0.7234
```

**Artifacts saved to `ml/src/models/`:**
- `sequence_xgb_model.pkl` - Trained calibrated model
- `sequence_feature_columns.json` - Feature names
- `sequence_threshold.json` - Threshold + win rate metadata

### 2. Make Predictions

Predict from latest data:

```bash
cd ml/src/scripts
python predict_sequence.py --data-dir ../data
```

Predict from specific CSV:

```bash
python predict_sequence.py --input-csv my_100_candles.csv
```

**Output:**
```
=== PREDICTION RESULT - SEQUENCE MODEL ===
Probability:       0.7345 (73.45%)
Prediction:        BUY (1)
Threshold:         0.45
Expected Win Rate: 0.6823 (68.23%)
Confidence:        HIGH

✅ MODEL RECOMMENDS: BUY
   Based on the last 100 candles, the model predicts a 73.45% chance
   of achieving the target return. Historical win rate: 68.23%
```

**JSON Output (for programmatic use):**
```json
{
  "probability": 0.7345,
  "prediction": 1,
  "threshold": 0.45,
  "expected_win_rate": 0.6823,
  "confidence": "high"
}
```

### 3. Run Tests

```bash
cd ml/tests
pytest test_sequence_pipeline.py -v
```

Expected output:
```
test_sequence_pipeline.py::TestSchemaValidation::test_valid_schema PASSED
test_sequence_pipeline.py::TestFeatureEngineering::test_feature_engineering_shape PASSED
test_sequence_pipeline.py::TestSequenceCreation::test_sequence_creation_shape PASSED
...
======================== 20 passed in 5.23s ========================
```

## Feature Engineering

Each candle generates **13 features**:

### Price Action Features
1. **ret_1**: 1-minute log return
2. **range_n**: Normalized range (High-Low)/Close
3. **body_ratio**: Candle body size relative to range
4. **upper_shadow**: Upper wick relative to range
5. **lower_shadow**: Lower wick relative to range

### Volume Features
6. **vol_change**: Log change in volume

### Trend Features
7. **ema_spread_n**: EMA(12) - EMA(26) spread normalized

### Momentum Features
8. **rsi_14**: 14-period Relative Strength Index

### Volatility Features
9. **vol_20**: 20-period rolling volatility

### Time Features
10. **hour_sin**: Sine encoding of hour (0-23)
11. **hour_cos**: Cosine encoding of hour
12. **minute_sin**: Sine encoding of minute (0-59)
13. **minute_cos**: Cosine encoding of minute

**Total input features**: 100 candles × 13 features = **1,300 dimensions**

## Data Requirements

### Input Format

CSV files with semicolon separator:
```
Date;Open;High;Low;Close;Volume
2023-01-01 00:00:00;1800.50;1801.20;1799.80;1800.90;5432
2023-01-01 00:01:00;1800.90;1802.10;1800.50;1801.80;6123
...
```

### Data Quality Checks

The pipeline validates:
- ✅ All required columns present (Date, Open, High, Low, Close, Volume)
- ✅ Prices are positive
- ✅ High >= Low (no inconsistencies)
- ✅ No NaN or infinite values
- ✅ Timestamps are unique and sorted

### Minimum Data Requirements

- **Training**: At least 100 candles + horizon (e.g., 105 for 5-min horizon)
- **Prediction**: Exactly 100 candles
- **Recommended**: 1+ years of historical data for robust training

## Model Details

### XGBoost Hyperparameters

```python
XGBClassifier(
    n_estimators=400,           # Number of trees
    max_depth=6,                # Tree depth
    learning_rate=0.05,         # Learning rate
    subsample=0.8,              # Row sampling
    colsample_bytree=0.8,       # Column sampling
    reg_lambda=1.0,             # L2 regularization
    scale_pos_weight=auto,      # Auto-calculated for class imbalance
    tree_method="hist",         # Fast histogram-based algorithm
)
```

### Calibration

Model uses **sigmoid calibration** on validation set to ensure:
- Predicted probabilities are well-calibrated
- P(success) = actual success rate for that probability bin
- Enables reliable confidence estimates

### Train/Validation/Test Split

**Chronological split** (no data leakage):
```
Train:      2004-2022  (all data up to 2022-12-31)
Validation: 2023       (2023-01-01 to 2023-12-31)
Test:       2024       (2024-01-01 to 2024-12-31)
```

## Metrics Explained

### Win Rate (Precision)
**Most important for trading!**

```
Win Rate = TP / (TP + FP)
         = Correct BUY signals / All BUY signals
```

Example: If model predicts BUY 100 times and 68 are profitable → **68% win rate**

### Recall
```
Recall = TP / (TP + FN)
       = Correct BUY signals / All profitable opportunities
```

High recall = catches most opportunities (but may have false positives)

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Balanced metric combining precision and recall.

### ROC-AUC
Area under ROC curve. Measures model's ability to discriminate between classes.
- 0.5 = Random guessing
- 1.0 = Perfect classification

### PR-AUC
Area under Precision-Recall curve. Better for imbalanced datasets.

## Interpretation Guide

### High Confidence Prediction
```
Probability:       0.82 (82%)
Win Rate:          0.68 (68%)
Confidence:        HIGH
```

**Interpretation**: Model is very confident (82%) this trade will succeed. Historical win rate shows that when model is this confident, it's correct 68% of the time.

**Action**: Strong BUY signal - high probability with proven track record.

### Medium Confidence Prediction
```
Probability:       0.52 (52%)
Win Rate:          0.68 (68%)
Confidence:        MEDIUM
```

**Interpretation**: Model is only slightly above threshold. Win rate is still 68%, but this specific prediction is marginal.

**Action**: Borderline - consider other factors (market conditions, risk tolerance).

### Low Confidence (No Trade)
```
Probability:       0.38 (38%)
Win Rate:          0.68 (68%)
Confidence:        LOW
```

**Interpretation**: Model doesn't see a clear pattern. Even though historical win rate is 68%, this specific case doesn't meet criteria.

**Action**: NO TRADE - wait for better setup.

## Performance Benchmarks

Based on XAU/USD 1-minute data (2004-2024):

### Expected Performance
- **Win Rate**: 60-70% (precision on test set)
- **ROC-AUC**: 0.70-0.85
- **F1 Score**: 0.55-0.70

### Red Flags
⚠️ **Win rate < 55%**: Model may need retraining or market regime changed  
⚠️ **ROC-AUC < 0.65**: Poor discriminative ability  
⚠️ **Huge train/test gap**: Overfitting - reduce model complexity

## Troubleshooting

### Issue: "Insufficient data for window"
**Cause**: Less than 100 candles provided  
**Solution**: Ensure input has at least 100 rows

### Issue: "Feature mismatch"
**Cause**: Model trained with different features than prediction input  
**Solution**: Retrain model or verify feature engineering matches

### Issue: "Model not found"
**Cause**: Artifacts not saved or wrong directory  
**Solution**: Run training pipeline first to generate artifacts

### Issue: Low win rate (<50%)
**Cause**: Market regime changed, overfitting, or bad hyperparameters  
**Solution**: 
1. Retrain with recent data
2. Check class balance
3. Tune hyperparameters (reduce `max_depth`, increase `reg_lambda`)
4. Collect more diverse training data

### Issue: High train accuracy, low test accuracy
**Cause**: Overfitting  
**Solution**:
1. Reduce `max_depth` (try 4-5)
2. Increase `reg_lambda` (try 2.0-5.0)
3. Increase `min_child_weight`
4. Use more dropout (`subsample=0.6`)

## Advanced Usage

### Custom Feature Engineering

To add new features, modify `engineer_candle_features()`:

```python
# Add your custom feature
custom_feature = df["Close"].rolling(50).mean() / df["Close"]

features["custom_feature"] = custom_feature
```

**Important**: Update both `sequence_training_pipeline.py` AND `predict_sequence.py`

### Hyperparameter Tuning

Use cross-validation to find optimal parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
}

# Run grid search...
```

### Live Trading Integration

```python
from predict_sequence import predict, load_latest_candles_from_dir

# In your trading bot
candles = load_latest_candles_from_dir(data_dir)
result = predict(candles, models_dir)

if result['prediction'] == 1 and result['confidence'] == 'high':
    # Execute BUY order
    execute_trade(
        symbol='XAUUSD',
        direction='BUY',
        confidence=result['probability'],
        expected_win_rate=result['expected_win_rate']
    )
```

## FAQ

**Q: Why 100 candles specifically?**  
A: 100 minutes (~1.5 hours) captures intraday patterns without excessive dimensionality. You can experiment with 50-200.

**Q: Can I use this for other instruments (EUR/USD, BTC)?**  
A: Yes! Just replace the CSV data. Feature engineering is instrument-agnostic.

**Q: How often should I retrain?**  
A: Monthly or when win rate drops below acceptable threshold (e.g., <55%).

**Q: What's the difference from original `training_pipeline.py`?**  
A: Original uses single-point features. This uses 100-candle sequences for better temporal context.

**Q: Can I use LSTM instead of XGBoost?**  
A: Yes! Replace `train_xgb()` with LSTM training. XGBoost is faster and easier to debug.

**Q: How do I interpret "expected_win_rate"?**  
A: It's the model's precision on test data. If 68%, then historically 68% of BUY predictions were profitable.

## References

- **XGBoost**: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"
- **Probability Calibration**: Platt (1999) - "Probabilistic Outputs for Support Vector Machines"
- **Time-Series Cross-Validation**: Bergmeir & Benítez (2012)

## License

© Capgemini 2025

---

**Last Updated**: December 10, 2025  
**Version**: 1.0.0  
**Author**: Trading ML Team
