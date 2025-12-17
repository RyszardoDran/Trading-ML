# Training Logging Configuration Changes

**Data**: 17 grudnia 2025  
**Zmiana**: Relokacja i ulepszenie systemu logowania treningowego

## Problem

Pliki logów treningowych trafiały do `ml/src/pipelines/train.log`, co powodowało:
1. ❌ Błędne miejsce przechowywania (mieszanie kodu ze wynikami)
2. ❌ Nadpisywanie logów (każdy trening kasował poprzednią historię)
3. ❌ Brak informacji co dokładnie dotyczy każdy log

## Rozwiązanie

### 1. **Zmiana lokalizacji logów**
- **Było**: `ml/src/pipelines/train.log`
- **Teraz**: `ml/outputs/logs/sequence_xgb_train_{years}_{YYYYMMDD_HHMMSS}.log`

### 2. **Unikalna nazwa z datą/czasem**
Każdy log ma unikalną nazwę zawierającą:
- Prefiks: `sequence_xgb_train`
- Lata treningu: `_all_years` lub `_years_2023_2024` itd.
- Timestamp startu: `_20251217_084530.log`

**Przykłady**:
```
sequence_xgb_train_all_years_20251217_084530.log
sequence_xgb_train_years_2023_2024_20251217_085101.log
```

### 3. **Opisowy zawartość logów**
Każdy log zawiera teraz:

```
================================================================================
SEQUENCE XGBoost TRAINING PIPELINE - XAU/USD 1-Minute Data
================================================================================
Log file: ml/outputs/logs/sequence_xgb_train_all_years_20251217_084530.log
Start time: 2025-12-17 08:45:30
Data years: All available years

Pipeline Configuration:
  Window size: 60 candles
  ATR SL multiplier: 1.0x
  ATR TP multiplier: 2.0x
  Min hold time: 5 minutes
  Max horizon: 60 candles
  Trading session: london_ny
  Min precision threshold: 0.85
  M5 alignment filter: enabled
  Trend filter: enabled
  Pullback filter: enabled
  Random state: 42
================================================================================
```

oraz na koniec:

```
================================================================================
TRAINING COMPLETE - SUMMARY
================================================================================
End time: 2025-12-17 08:50:15

Final Metrics:
  Window Size:       60 candles
  Threshold:         0.5234
  WIN RATE:          0.8652 (86.52%)
  Precision:         0.8652
  Recall:            0.7234
  F1 Score:          0.7898
  ROC-AUC:           0.8934
  PR-AUC:            0.8765

Artifacts saved to: ml/outputs/models
================================================================================
```

## Zmiany w kodzie

### `ml/src/utils/sequence_training_config.py`
```python
@property
def outputs_logs_dir(self) -> Path:
    """Directory for training logs."""
    return self.outputs_dir / "logs"

# Plus properties dla models, metrics, analysis directories
```

### `ml/src/pipelines/sequence_training_pipeline.py`

**Nowa funkcja**:
```python
def _setup_logging(config: PipelineConfig, year_filter: Optional[List[int]] = None) -> str:
    """Setup logging to file with unique timestamp-based name."""
    # Tworzy ml/outputs/logs/ jeśli nie istnieje
    # Generuje unikalną nazwę z datą/czasem
    # Konfiguruje file + console handlers
```

**W `main()` section**:
```python
# Setup logging przed run_pipeline()
config = PipelineConfig()
log_filepath = _setup_logging(config, year_filter)

# Loguj header z pełną konfiguracją
logger.info("SEQUENCE XGBoost TRAINING PIPELINE - XAU/USD 1-Minute Data")
logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# ... więcej detali
```

## Użytkowanie

Nic się nie zmienia w sposobie uruchamiania treningu:

```bash
# Standardowe uruchomienie
python ml/src/pipelines/sequence_training_pipeline.py

# Z parametrami
python ml/src/pipelines/sequence_training_pipeline.py --window-size 50 --years 2023,2024
```

Wynikiem będzie nowy log w `ml/outputs/logs/` z automatyczną unikalną nazwą.

## Korzyści

✅ **Separacja kodu od rezultatów** - Logi trafiają do `outputs/`, nie do `src/`  
✅ **Brak nadpisywania** - Każdy trening tworzy nowy plik z timestamp  
✅ **Pełna dokumentacja** - Log zawiera pełną konfigurację treningu  
✅ **Historia** - Wszystkie logi przechowywane chronologicznie  
✅ **Łatwy debugging** - Pełny kontekst treningowy w jednym pliku  

## Struktura katalogów

```
ml/
├── src/
│   ├── pipelines/
│   │   └── sequence_training_pipeline.py (kod, bez logów!)
│   ├── models/
│   └── data/
│
└── outputs/
    ├── logs/
    │   ├── sequence_xgb_train_all_years_20251217_084530.log
    │   ├── sequence_xgb_train_years_2023_2024_20251217_090145.log
    │   └── ... (więcej logów)
    ├── models/
    │   ├── sequence_xgb_model.pkl
    │   └── ...
    ├── metrics/
    └── analysis/
```

---

**Status**: ✅ **GOTOWE**
