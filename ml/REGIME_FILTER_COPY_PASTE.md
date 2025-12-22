# Regime Filter: Copy-Paste Integration Code

## Option A: Backtest Integration (Simplest)

### Before (Without Regime Filter)
```python
# File: ml/scripts/backtest_strategy.py
# Around line 210

for i in range(1, len(prices)):
    signal = signals[i - 1]  # Original signal
    
    if signal == 1:
        # Execute trade
        entry_price = prices[i - 1] * (1 + spread / 2)
        exit_price = prices[i] * (1 - spread / 2)
        # ... rest of trade logic
```

### After (With Regime Filter - Copy This)
```python
# File: ml/scripts/backtest_strategy.py
# Around line 210

# ← ADD THIS IMPORT AT TOP OF FILE
from ml.src.filters.regime_filter import RegimeFilter
from ml.src.utils.risk_config import ENABLE_REGIME_FILTER
import numpy as np

# ← ADD THIS IN SIMULATION FUNCTION (once, before loop)
regime_filter = RegimeFilter() if ENABLE_REGIME_FILTER else None

# ← REPLACE THE LOOP WITH THIS:
for i in range(1, len(prices)):
    signal = signals[i - 1]  # Original signal
    confidence = confidence_scores[i - 1]  # Model confidence
    
    # ← ADD THIS BLOCK: Regime filter gating
    if regime_filter is not None and signal == 1:
        filtered_signal = regime_filter.filter_predictions_by_regime(
            signals=np.array([signal]),
            confidence=np.array([confidence]),
            indicators={
                'atr': atr_values[i - 1],
                'adx': adx_values[i - 1],
                'close': prices[i - 1],
                'sma200': sma200_values[i - 1]
            }
        )
        signal = filtered_signal[0]
    
    if signal == 1:
        # Execute trade
        entry_price = prices[i - 1] * (1 + spread / 2)
        exit_price = prices[i] * (1 - spread / 2)
        # ... rest of trade logic
```

**Total addition: 15 lines of code**

---

## Option B: Real-Time Trading Integration

### Before (Without Regime Filter)
```python
# Your trading bot / inference service

def predict_and_trade(market_data):
    features = engineer_features(market_data)
    prediction = model.predict_proba(features)[1]
    
    if prediction > 0.5:
        place_order()  # Execute immediately
```

### After (With Regime Filter - Copy This)
```python
# Your trading bot / inference service

# ← ADD THESE IMPORTS AT TOP
from ml.src.filters.regime_filter import RegimeFilter
from ml.src.utils.risk_config import ENABLE_REGIME_FILTER
import numpy as np

# ← ADD THIS CLASS/FUNCTION
class TradingBot:
    def __init__(self):
        self.model = load_model()
        self.regime_filter = RegimeFilter() if ENABLE_REGIME_FILTER else None
    
    def predict_and_trade(self, market_data):
        """Predict with regime-aware gating."""
        
        # Get features and prediction
        features = engineer_features(market_data)
        prediction = self.model.predict_proba(features)[1]
        signal = 1 if prediction > 0.5 else 0
        
        # ← ADD THIS BLOCK: Regime filtering
        if self.regime_filter is not None and signal == 1:
            filtered_signal = self.regime_filter.filter_predictions_by_regime(
                signals=np.array([signal]),
                confidence=np.array([prediction]),
                indicators={
                    'atr': market_data['atr'],
                    'adx': market_data['adx'],
                    'close': market_data['close'],
                    'sma200': market_data['sma200']
                }
            )
            signal = filtered_signal[0]
        
        # Execute trade (only if signal = 1 after filtering)
        if signal == 1:
            place_order()
```

**Total addition: 20 lines of code**

---

## Option C: Existing Function Wrapper

### If you have existing prediction function:

```python
# Before: Your existing function
def make_prediction(features):
    return model.predict_proba(features)[1]

signal = 1 if make_prediction(features) > 0.5 else 0
if signal == 1:
    execute_trade()

# ────────────────────────────────────────────

# After: Wrapped with regime filter
from ml.src.filters.regime_filter import RegimeFilter
from ml.src.utils.risk_config import ENABLE_REGIME_FILTER

regime_filter = RegimeFilter() if ENABLE_REGIME_FILTER else None

def make_prediction_with_regime_filter(features, indicators):
    """Prediction with regime-aware gating."""
    confidence = model.predict_proba(features)[1]
    signal = 1 if confidence > 0.5 else 0
    
    # Apply regime filter
    if regime_filter is not None and signal == 1:
        signal = regime_filter.filter_predictions_by_regime(
            signals=np.array([signal]),
            confidence=np.array([confidence]),
            indicators=indicators
        )[0]
    
    return signal, confidence

# Usage:
signal, confidence = make_prediction_with_regime_filter(features, indicators)
if signal == 1:
    execute_trade()
```

---

## Quick Configuration Reference

### Step 0: Enable Filter (One Line Change)

**File**: `ml/src/utils/risk_config.py`

```python
# Change this line
ENABLE_REGIME_FILTER = True  # ← Set to True (was False)
```

### Step 1: Import (Top of File)

```python
from ml.src.filters.regime_filter import RegimeFilter
from ml.src.utils.risk_config import ENABLE_REGIME_FILTER
import numpy as np  # For np.array()
```

### Step 2: Initialize (Once)

```python
# In main function or class __init__
regime_filter = RegimeFilter() if ENABLE_REGIME_FILTER else None
```

### Step 3: Apply Filter (Before Trade)

```python
# When you have a signal
signal = 1  # From model prediction
confidence = 0.65  # Model confidence [0, 1]

if regime_filter is not None and signal == 1:
    signal = regime_filter.filter_predictions_by_regime(
        signals=np.array([signal]),
        confidence=np.array([confidence]),
        indicators={
            'atr': current_atr_value,
            'adx': current_adx_value,
            'close': current_price,
            'sma200': current_sma200
        }
    )[0]

# Now use filtered signal
if signal == 1:
    place_order()
```

---

## Testing After Integration

### Quick Test (5 minutes)
```bash
cd ml
python scripts/simple_regime_filter_test.py
```

Expected: Should complete successfully with XGBoost training initiated

### Full Validation (1-2 hours)
```bash
cd ml
python scripts/walk_forward_with_regime_filter.py
```

Expected output:
```
WITHOUT regime filter: WIN_RATE = 31.58%
WITH regime filter:    WIN_RATE = 45-50%
IMPROVEMENT:           +13.4 to +18.4 pp ✅
```

---

## Debugging Checklist

If integration doesn't work, check:

1. **Filter enabled?**
   ```python
   from ml.src.utils.risk_config import ENABLE_REGIME_FILTER
   print(ENABLE_REGIME_FILTER)  # Should be True
   ```

2. **Indicators calculated?**
   ```python
   print(f"ATR: {atr}")    # Should be > 0
   print(f"ADX: {adx}")    # Should be > 0
   print(f"SMA200: {sma}") # Should be > 0
   ```

3. **Signal being generated?**
   ```python
   print(f"Signal: {signal}")          # Should be 0 or 1
   print(f"Confidence: {confidence}")  # Should be [0, 1]
   ```

4. **Filter being applied?**
   ```python
   original = signal
   filtered = regime_filter.filter_predictions_by_regime(...)
   print(f"Signal: {original} → {filtered[0]}")  # Shows change
   ```

5. **Imports working?**
   ```bash
   cd ml
   python -c "from ml.src.filters.regime_filter import RegimeFilter; print('✅ Import works')"
   ```

---

## Expected Behavior

### Signal Before/After Filtering

```
Market Condition          Signal  Confidence  After Filter  Why
─────────────────────────────────────────────────────────────────
TIER 1 (ATR≥18)          1       70%         → 1 ✅         TRADE
                                              (conf > 0.35)

TIER 2 (ATR 12-17)       1       55%         → 1 ✅         TRADE
                                              (conf > 0.50)

TIER 3 (ATR 8-11)        1       45%         → 0 🚫         SUPPRESS
                                              (conf < 0.65)

TIER 4 (ATR < 8)         1       60%         → 0 🚫         SUPPRESS
                                              (ATR too low)

Low ATR + ADX            1       80%         → 0 🚫         SUPPRESS
                                              (No trend)

Price below SMA200       1       75%         → 0 🚫         SUPPRESS
                                              (Downtrend)

Model says "No"          0       30%         → 0 ✅         SKIP
                                              (Not suppressed)
```

---

## Integration Checklist

Before going live:

- [ ] Regime filter import added
- [ ] `ENABLE_REGIME_FILTER = True` in config
- [ ] Regime filter initialized
- [ ] Indicators available (ATR, ADX, SMA200)
- [ ] Filtering code added before trade execution
- [ ] Simple test passed (`simple_regime_filter_test.py`)
- [ ] Walk-forward validation passed (45-50% win rate)
- [ ] Code reviewed
- [ ] Monitoring metrics set up
- [ ] Alerts configured
- [ ] Rollback plan documented

---

## File References

| File | Purpose | Lines to Change |
|------|---------|-----------------|
| `ml/src/utils/risk_config.py` | Enable filter | 1 line |
| `ml/scripts/backtest_strategy.py` | Add to backtest | ~15 lines |
| Your inference bot | Add to real-time | ~10-15 lines |
| Your test script | Add to validation | ~10-15 lines |

---

## Support Files

For more help, see:
- [PRODUCTION_INTEGRATION_GUIDE.md](PRODUCTION_INTEGRATION_GUIDE.md) - Full guide
- [REGIME_FILTER_VISUAL_GUIDE.md](REGIME_FILTER_VISUAL_GUIDE.md) - Diagrams
- [backtest_with_regime_filter_example.py](scripts/backtest_with_regime_filter_example.py) - Backtest example
- [realtime_inference_example.py](scripts/realtime_inference_example.py) - Real-time example

---

**Ready to integrate? Copy a code block above and adapt for your use case!** 🚀
