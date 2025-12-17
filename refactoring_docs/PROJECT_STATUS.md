# Trading-ML: XAU/USD Day Trading System

**Status**: ✅ Etap 4 Complete - Ready for Etap 5

## Quick Links

- **Project Overview**: See [README.md](README.md)
- **Refactoring Documentation**: See [refactoring_docs/INDEX.md](refactoring_docs/INDEX.md)
- **Architecture & Design**: See [docs/](docs/)
- **Implementation**: See [ml/](ml/)

## Current Status

### Completed ✅
- **Etap 1**: Data loading structure and modularization
- **Etap 2**: Feature engineering module organization
- **Etap 3**: Target and sequence creation modules
- **Etap 4**: Training and evaluation modules extraction
  - 6 core functions extracted to `ml/src/training/`
  - 47% code reduction in main pipeline
  - All imports tested and working ✓

### In Progress
- **Etap 5**: Main API cleanup and CLI script creation (planned)

## Project Structure

```
Trading-ML/
├── ml/                          # Machine learning implementation
│   └── src/
│       ├── data_loading/        # Data loading & validation (Etap 1)
│       ├── features/            # Feature engineering (Etap 2)
│       ├── targets/             # Target creation (Etap 3)
│       ├── sequences/           # Sequence building (Etap 3)
│       ├── training/            # Model training (Etap 4) ← NEW
│       ├── pipelines/           # Pipeline orchestration
│       │   ├── sequence_training_pipeline.py  # Main API
│       │   └── sequence_split.py              # Data splitting
│       ├── config/              # Configuration files
│       └── utils/               # Utilities
├── docs/                        # Architecture & design documentation
├── refactoring_docs/            # All refactoring stage documentation
│   └── INDEX.md                 # Complete refactoring index
└── README.md                    # Project overview
```

## Etap 4 Achievements

### Extracted Functions (to `ml/src/training/`)
1. **sequence_xgb_trainer.py** - `train_xgb()`
   - XGBoost training with probability calibration
   - Class imbalance handling
   
2. **sequence_evaluation.py** - `evaluate()`
   - Comprehensive model metrics
   - Threshold optimization (`_pick_best_threshold()`)
   - Trade limiting (`_apply_daily_cap()`)

3. **sequence_feature_analysis.py** - `analyze_feature_importance()`
   - Feature importance extraction from trained model
   - Candle offset mapping

4. **sequence_artifacts.py** - `save_artifacts()`
   - Model and scaler serialization
   - Feature columns and metadata persistence

### Code Statistics
- **Lines reduced**: 816 → 433 lines in main pipeline (47% reduction)
- **Functions extracted**: 6 (all with tests)
- **New modules**: 1 (`ml/src/training/`)
- **Files renamed**: 5 (with `sequence_` prefix for clarity)

## Getting Started

### For Development
```bash
# Install dependencies
pip install -r ml/requirements_ml.txt

# Run tests
pytest ml/tests/

# Run main pipeline
python -c "from ml.src.pipelines.sequence_training_pipeline import run_pipeline; run_pipeline()"
```

### For Documentation
1. Start with [refactoring_docs/INDEX.md](refactoring_docs/INDEX.md)
2. Read [docs/02-ML-Architecture.md](docs/02-ML-Architecture.md) for system design
3. Check [ml/SEQUENCE_PIPELINE_README.md](ml/SEQUENCE_PIPELINE_README.md) for implementation details

## Next Phase (Etap 5)

**Objectives**:
- Refactor `run_pipeline()` for clarity and flexibility
- Create CLI scripts in `ml/src/scripts/`
- Establish public API interface
- Add production logging and monitoring

See [refactoring_docs/REFACTOR_PLAN.md](refactoring_docs/REFACTOR_PLAN.md) for detailed Etap 5 specifications.

---

**Last Updated**: After Etap 4 completion  
**Project**: Trading-ML - XAU/USD Day Trading System  
**Contact**: Arek
