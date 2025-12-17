# ML Scripts - CLI Entry Points

CLI (Command-Line Interface) entry points for the Trading-ML XAU/USD system.

## Scripts

### `train_sequence_model.py` - Main Training Script

Train a sequence-based XGBoost model for XAU/USD 1-minute trading signals.

#### Quick Start

```bash
# Default parameters (60-candle window, all available data)
python ml/scripts/train_sequence_model.py

# Show all options
python ml/scripts/train_sequence_model.py --help

# Custom window size
python ml/scripts/train_sequence_model.py --window-size 50 --max-horizon 120

# Test on specific years
python ml/scripts/train_sequence_model.py --years 2023,2024

# Disable some filters
python ml/scripts/train_sequence_model.py --disable-trend-filter --disable-pullback-filter
```

#### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--window-size N` | Number of previous candles for input | 60 |
| `--max-horizon N` | Maximum forward candles for target | 60 |
| `--atr-multiplier-sl X` | ATR multiplier for stop-loss | 1.0 |
| `--atr-multiplier-tp X` | ATR multiplier for take-profit | 2.0 |
| `--min-hold-minutes N` | Minimum holding time in minutes | 5 |
| `--years YEARS` | Comma-separated years (e.g., '2023,2024') | All years |
| `--session {london,ny,asian,london_ny,all,custom}` | Trading session filter | london_ny |
| `--min-precision P` | Minimum precision (win rate) floor | 0.85 |
| `--disable-trend-filter` | Disable SMA200 + ADX trend filter | Enabled |
| `--disable-pullback-filter` | Disable RSI_M5 pullback guard | Enabled |
| `--random-state SEED` | Random seed for reproducibility | 42 |
| `-v, --verbose` | Enable verbose output | False |

#### Output

The script creates:
1. **Model Artifacts** (in `ml/outputs/models/`):
   - `sequence_xgb_model.pkl` - Trained XGBoost classifier
   - `sequence_scaler.pkl` - RobustScaler for feature normalization
   - `sequence_feature_columns.json` - Ordered list of 57+ feature names
   - `sequence_metadata.json` - Training configuration and hyperparameters
   - `sequence_threshold.json` - Optimal decision threshold + expected win rate

2. **Training Log** (in `ml/outputs/logs/`):
   - `sequence_xgb_train_*.log` - Timestamped training log

#### Example Usage & Output

```bash
$ python ml/scripts/train_sequence_model.py --window-size 50 --years 2024

================================================================================
XAU/USD SEQUENCE XGBoost TRAINING - CLI ENTRY POINT
================================================================================
Training years: [2024]
Configuration:
  Window size: 50 candles
  Max horizon: 60 candles
  ATR SL multiplier: 1.0x
  ATR TP multiplier: 2.0x
  Min hold time: 5 minutes
  Trading session: london_ny
  M5 alignment: enabled
  Trend filter: enabled
  Pullback filter: enabled
  Min precision threshold: 0.85
  Random state: 42
================================================================================

Starting training pipeline...
Loading data...
Loaded 240000 rows from ml/src/data
Engineering per-candle features (window_size=50)...
Features shape: (240000, 57)
Creating target (SL=1.0×ATR, TP=2.0×ATR, min_hold=5min)...
Target shape: 240000, positive class: 18000 (7.50%)
Creating sequences (window_size=50)...
Filter configuration: m5=True, trend=True(dist_sma200=0.0, adx=15.0), pullback=True(rsi<=75.0)
Sequences: X.shape=(100000, 2850), y.shape=(100000,)
[... more logging ...]

================================================================================
TRAINING COMPLETED SUCCESSFULLY
================================================================================
Threshold: 0.6234
Win Rate (Precision): 0.8523 (85.23%)
Recall: 0.6145
F1 Score: 0.7234
ROC-AUC: 0.8912
PR-AUC: 0.8456
================================================================================

✅ Training completed successfully!
   Window Size: 50 candles
   Win Rate: 85.23%
   Threshold: 0.6234

📁 Artifacts saved to: ml/outputs/models/
📊 Logs saved to: ml/outputs/logs/
```

## Future Scripts (Etap 6+)

### `eval_model.py` - Model Evaluation (Planned)
Evaluate a trained model on new data.

```bash
python ml/scripts/eval_model.py --model-path ml/outputs/models/sequence_xgb_model.pkl --data-path ml/src/data/
```

### `analyze_features.py` - Feature Importance Analysis (Planned)
Analyze feature importance and create visualizations.

```bash
python ml/scripts/analyze_features.py --model-path ml/outputs/models/sequence_xgb_model.pkl
```

### `backtest_strategy.py` - Strategy Backtesting (Planned)
Backtest trading strategy with different scenarios.

```bash
python ml/scripts/backtest_strategy.py --model-path ml/outputs/models/sequence_xgb_model.pkl --strategy-config config.yaml
```

## Architecture

```
User Input
    ↓
CLI Script (train_sequence_model.py)
    ↓
ArgumentParser
    ↓
Validate Arguments
    ↓
run_pipeline() from ml.src.pipelines
    ↓
Complete ML Pipeline
    ↓
Save Artifacts
    ↓
Exit with Status Code (0=success, 1=failure)
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Invalid years format` | Wrong year format (not comma-separated) | Use `--years 2023,2024` format |
| `FileNotFoundError: Data files not found` | CSV files missing in `ml/src/data/` | Ensure `XAU_1m_data_*.csv` files exist |
| `Exception: Training failed` | Unexpected error in pipeline | Check logs in `ml/outputs/logs/` |

## Exit Codes

- **0**: Training completed successfully
- **1**: Invalid arguments or data not found
- **1**: Unexpected error during training

## Development

### Adding New Scripts

1. Create new script in `ml/scripts/` (e.g., `new_script.py`)
2. Implement `main()` function that returns exit code
3. Add argparse configuration
4. Document in this README
5. Test with `--help` and basic execution

### Testing

```bash
# Test help message
python ml/scripts/train_sequence_model.py --help

# Test with default parameters
python ml/scripts/train_sequence_model.py

# Test with custom parameters
python ml/scripts/train_sequence_model.py --window-size 50 --years 2023

# Check syntax
python -m py_compile ml/scripts/train_sequence_model.py
```

## Related Documentation

- [Etap 5 Completion](../refactoring_docs/ETAP_5_COMPLETE.md) - Full Etap 5 details
- [Pipeline API](ml/src/pipelines/README.md) - Pipeline module documentation
- [Feature Engineering](ml/src/features/README.md) - Feature module documentation
