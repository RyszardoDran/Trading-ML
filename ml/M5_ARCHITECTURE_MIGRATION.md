# Migration to M5 Timeframe Architecture

**Date**: 2025-12-18  
**Status**: ✅ COMPLETED

## Overview

Successfully migrated from M1-based feature engineering with forward-filled M5 indicators to **true M5 timeframe architecture** with proper OHLCV aggregation.

## Architecture Changes

### Before (M1 + MTF Indicators)
```
M1 raw data (10,080 candles)
    ↓
M5/M15/M60 indicators calculated via resample + forward-fill to M1
    ↓
Features on M1 resolution (10,080 features)
    ↓
Model: 100 M1 candles = 100 minutes context
```

### After (True M5 Bars)
```
M1 raw data (10,080 candles)
    ↓
Aggregate M1 → M5 (proper OHLCV: Open=first, High=max, Low=min, Close=last, Volume=sum)
    ↓
M5 bars (2,016 candles - 5x compression)
    ↓
Features on M5 timeframe with M15/M60 context
    ↓
Model: 100 M5 candles = 500 minutes context (8.3 hours)
```

## Key Benefits

1. **Proper Timeframe Context**: Model sees 100 true M5 candles (500 min) vs 100 M1 candles (100 min)
2. **Cleaner Signal**: M5 bars filter M1 noise while preserving price action
3. **Multi-Timeframe Integrity**: M5→M15→M60 relationships are mathematically correct
4. **Reduced Data Size**: 5x compression (10,080 M1 → 2,016 M5) improves performance
5. **Semantic Correctness**: Trading M5 strategy should use M5 data, not M1 with M5 indicators

## Files Modified

### Core Pipeline
1. **`ml/src/pipeline_stages.py`**
   - `engineer_features_stage()`: Now aggregates M1→M5 first, then engineers features
   - `create_targets_stage()`: Updated to work on M5 timeframe (parameters in M5 candle units)
   - `build_sequences_stage()`: Disabled M5 alignment filter (data already M5-aligned)
   - Imports: Changed from `engineer_candle_features` to `engineer_m5_candle_features`

2. **`ml/src/pipelines/sequence_training_pipeline.py`**
   - Stage 1: Loads M1 data
   - Stage 2: Calls `engineer_features_stage()` which internally aggregates M1→M5
   - Stage 3: Re-aggregates M1→M5 for `create_targets_stage()` (uses M5 OHLCV for SL/TP simulation)
   - Stage 4: Uses M5 datetime index for sequence building

3. **`ml/src/scripts/predict_sequence.py`**
   - Updated documentation to reflect M5 architecture
   - `predict()`: Aggregates M1→M5 before feature engineering
   - Returns: `m1_candles_analyzed`, `m5_candles_generated`, `m5_candles_used_for_prediction`
   - Imports: Changed from `engineer_candle_features` to `engineer_m5_candle_features`

### New Module
4. **`ml/src/features/engineer_m5.py`** (Created previously)
   - `aggregate_to_m5()`: Proper OHLCV aggregation M1→M5
   - `engineer_m5_candle_features()`: Engineers 15 features directly on M5 timeframe
   - Multi-timeframe: M5→M15 (3:1), M5→M60 (12:1) using resample

## Breaking Changes

### Training Parameters
- **`min_hold_minutes`**: Now in M5 candle units (1 candle = 5 minutes)
  - Before: `min_hold_minutes=5` = 5 M1 candles = 5 minutes
  - After: `min_hold_minutes=1` = 1 M5 candle = 5 minutes
  
- **`max_horizon`**: Now in M5 candle units
  - Before: `max_horizon=60` = 60 M1 candles = 60 minutes
  - After: `max_horizon=60` = 60 M5 candles = 300 minutes (5 hours)

### Model Input Shape
- Before: `(n_sequences, 15 features × 100 M1 candles) = (n, 1500)`
- After: `(n_sequences, 15 features × 100 M5 candles) = (n, 1500)` (same shape, different timeframe)

### Prediction Output
Changed dictionary keys:
- ~~`candles_analyzed`~~ → `m1_candles_analyzed` (input M1 count)
- ~~`candles_used_for_prediction`~~ → `m5_candles_used_for_prediction` (M5 window size)
- Added: `m5_candles_generated` (M5 candles after aggregation)

## Backwards Compatibility

### Existing Models
⚠️ **NOT COMPATIBLE** - Old models trained on M1 data cannot be used with new M5 pipeline.

**Required Action**: Retrain all models with new M5 architecture.

### Configuration Files
✅ **COMPATIBLE** - No config file changes required (same parameter names).

**Note**: Interpret time-based parameters differently:
- `window_size=100`: Now 100 M5 candles (500 min) instead of 100 M1 candles (100 min)
- `max_horizon=60`: Now 60 M5 candles (300 min) instead of 60 M1 candles (60 min)

## Migration Checklist

- [✅] Created `engineer_m5.py` module
- [✅] Updated `pipeline_stages.py` imports and functions
- [✅] Modified `sequence_training_pipeline.py` to use M5 aggregation
- [✅] Updated `predict_sequence.py` for M5 predictions
- [✅] Disabled M5 alignment filter (data already M5-aligned)
- [✅] Updated all docstrings to reflect M5 timeframe
- [✅] Verified no syntax errors in modified files
- [ ] Retrain London session model with M5 architecture
- [ ] Validate prediction script with new M5 model
- [ ] Update any integration tests referencing old architecture
- [ ] Update README.md with M5 architecture documentation

## Testing Strategy

### Unit Tests
1. Test `aggregate_to_m5()`: Verify OHLCV aggregation correctness
2. Test `engineer_m5_candle_features()`: Verify 15 features generated on M5
3. Test `predict()`: Verify M1→M5 aggregation + prediction pipeline

### Integration Tests
1. Run full training pipeline: `python sequence_training_pipeline.py`
2. Verify artifacts: Check `analysis_window_days=7` in metadata
3. Test prediction: `python predict_sequence.py --data-dir ml/src/data`
4. Compare M5 model vs old M1 model: Win rate, precision, ROC-AUC

### Validation Checks
- ✅ 7 days M1 (~10,080) → ~2,016 M5 candles (5x compression)
- ✅ M5 alignment filter disabled (data already M5-aligned)
- ✅ Feature count: 15 features on M5 timeframe
- ✅ Sequence shape: (n_sequences, 1500) = (n, 15 features × 100 M5 candles)

## Expected Results

### Training Metrics (Estimated)
- **Data Compression**: 10,080 M1 → 2,016 M5 candles (~80% reduction)
- **Sequences Generated**: Expect ~20-30% fewer sequences (M5 has less granularity)
- **Training Time**: Should decrease (5x fewer candles to process)
- **Model Performance**: Expect similar or better (cleaner signal, less noise)

### Prediction Changes
- **Context Window**: 100 M5 candles = 500 minutes (8.3 hours) vs 100 minutes before
- **Feature Quality**: Indicators calculated on proper M5 bars (not forward-filled M1)
- **Response Time**: Should decrease (fewer candles to aggregate and process)

## Rollback Plan

If M5 architecture shows worse performance:

1. **Revert Files**:
   ```bash
   git checkout HEAD~1 ml/src/pipeline_stages.py
   git checkout HEAD~1 ml/src/pipelines/sequence_training_pipeline.py
   git checkout HEAD~1 ml/src/scripts/predict_sequence.py
   ```

2. **Keep Old Models**: Store old M1-based models separately before training M5 versions

3. **A/B Test**: Run both architectures in parallel and compare:
   - Win rate on validation set
   - Precision/Recall balance
   - ROC-AUC and PR-AUC
   - Real-world backtesting results

## Next Steps

1. **Retrain London Session Model**:
   ```bash
   python ml/src/pipelines/sequence_training_pipeline.py
   ```

2. **Validate Prediction**:
   ```bash
   python ml/src/scripts/predict_sequence.py --data-dir ml/src/data
   ```

3. **Compare Performance**:
   - Old M1 model: 70.46% win rate, 0.7965 ROC-AUC
   - New M5 model: TBD (expect similar or better)

4. **Update Documentation**:
   - README.md: Add M5 architecture section
   - SEQUENCE_PIPELINE_README.md: Update with M5 details
   - .github/instructions/python-ml.instructions.md: Add M5 best practices

## Technical Notes

### M5 Aggregation Details
```python
# Proper OHLCV aggregation M1 → M5
df_m5 = df_m1.resample('5T', label='right', closed='right').agg({
    'Open': 'first',    # First M1 open in 5-minute window
    'High': 'max',      # Highest high in 5-minute window
    'Low': 'min',       # Lowest low in 5-minute window
    'Close': 'last',    # Last M1 close in 5-minute window
    'Volume': 'sum'     # Total volume in 5-minute window
})
```

### Time Unit Conversion
- **1 M5 candle** = 5 minutes = 5 M1 candles
- **100 M5 candles** = 500 minutes = 8.3 hours
- **7 days M1** = 10,080 M1 candles = 2,016 M5 candles

### Feature Engineering Flow
```
M1 data → aggregate_to_m5() → M5 OHLCV
                                    ↓
                        engineer_m5_candle_features()
                                    ↓
                        M5 features (15 columns)
                                    ↓
                        M15/M60 context (resample from M5)
                                    ↓
                        Final M5 feature matrix
```

## References

- **Original Issue**: User identified M5 alignment only filtered timestamps, didn't aggregate
- **Solution**: Created `engineer_m5.py` for true M5 aggregation + integrated into pipeline
- **Design Philosophy**: "Trading M5 strategy should use M5 data, not M1 with M5 indicators"

---

**Author**: GitHub Copilot (Senior Python ML Engineer Mode)  
**Review Status**: ✅ Code changes complete, awaiting model retraining validation  
**Migration Date**: 2025-12-18
