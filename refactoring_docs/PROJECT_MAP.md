# 🗺️ MAPA PROJEKTU PO REFACTORYZACJI

## Struktura Katalogów (Kompleksowa)

```
Trading-ML/
│
├── ml/                                    [GŁÓWNY KATALOG PROJEKTU]
│
├── 📚 DOKUMENTACJA REFACTORYZACJI
│   ├── REFACTORING_SUMMARY.md             [📖 PODSUMOWANIE]
│   └── QUICK_REFERENCE.md                 [⚡ SZYBKA KARTA]
│
├── src/                                   [KOD - BIBLIOTEKA DO IMPORTU]
│   ├── __init__.py
│   │
│   ├── pipelines/                         [GŁÓWNY MODUŁ PIPELINE'U]
│   │   │
│   │   ├── 📚 DOKUMENTACJA (ETAPY)
│   │   │   ├── INDEX.md                   [📚 MAPA DOKUMENTACJI]
│   │   │   ├── ROADMAP.md                 [🗺️ DIAGRAM]
│   │   │   ├── REFACTOR_PLAN.md           [📋 PLAN 7 ETAPÓW]
│   │   │   ├── REFACTOR_ETAP_1.md         [🏗️ ETAP 1: Katalogi]
│   │   │   ├── REFACTOR_ETAP_2.md         [✨ ETAP 2: Features (planowy)]
│   │   │   ├── REFACTOR_ETAP_3.md         [🎯 ETAP 3: Targets/Sequences (planowy)]
│   │   │   ├── REFACTOR_ETAP_4.md         [🚀 ETAP 4: Training (planowy)]
│   │   │   ├── REFACTOR_ETAP_5.md         [🎬 ETAP 5: Main + CLI (planowy)]
│   │   │   ├── REFACTOR_ETAP_6.md         [📊 ETAP 6: Scripts (planowy)]
│   │   │   └── REFACTOR_ETAP_7.md         [✅ ETAP 7: Tests (planowy)]
│   │   │
│   │   ├── 🔧 KONFIGURACJA
│   │   │   ├── config.py                  [🆕 Centralna konfiguracja]
│   │   │   │   └── class PipelineConfig
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📥 ŁADOWANIE DANYCH [data_loading/]
│   │   │   ├── __init__.py
│   │   │   ├── loaders.py                 [load_all_years()]
│   │   │   ├── validators.py              [_validate_schema()]
│   │   │   └── [Etap 1]
│   │   │
│   │   ├── ✨ INŻYNIERIA CECH [features/]
│   │   │   ├── __init__.py
│   │   │   ├── engineer.py                [engineer_candle_features() - MAIN]
│   │   │   ├── indicators.py              [Indykatory: EMA, ADX, RSI, etc.]
│   │   │   ├── m5_context.py              [M5 context (resampling, ATR, RSI)]
│   │   │   ├── time_features.py           [Kodowanie godziny/minuty]
│   │   │   └── [Etap 2]
│   │   │
│   │   ├── 🎯 TWORZENIE CELU [targets/]
│   │   │   ├── __init__.py
│   │   │   ├── target_maker.py            [make_target() - backtest-based]
│   │   │   └── [Etap 3]
│   │   │
│   │   ├── 📊 TWORZENIE SEKWENCJI [sequences/]
│   │   │   ├── __init__.py
│   │   │   ├── config.py                  [SequenceFilterConfig dataclass]
│   │   │   ├── sequencer.py               [create_sequences() - main]
│   │   │   ├── filters.py                 [filter_by_session(), trend/pullback]
│   │   │   └── [Etap 1 (config), Etap 3 (rest)]
│   │   │
│   │   ├── 🚀 TRENING & EWALUACJA [training/]
│   │   │   ├── __init__.py
│   │   │   ├── xgb_trainer.py             [train_xgb()]
│   │   │   ├── evaluation.py              [evaluate(), _pick_best_threshold()]
│   │   │   ├── daily_cap.py               [_apply_daily_cap()]
│   │   │   ├── feature_analysis.py        [analyze_feature_importance()]
│   │   │   ├── artifacts.py               [save_artifacts()]
│   │   │   └── [Etap 4]
│   │   │
│   │   ├── 🛠️ UTYLITY [utils/]
│   │   │   ├── __init__.py
│   │   │   └── helpers.py                 [Funkcje pomocnicze]
│   │   │       └── [Etap 4]
│   │   │
│   │   ├── 📉 CHRONOLOGICZNY SPLIT
│   │   │   └── split.py                   [split_sequences() - train/val/test]
│   │   │       └── [Etap 1]
│   │   │
│   │   ├── 🎬 GŁÓWNY PLIK PIPELINE
│   │   │   ├── sequence_training_pipeline.py
│   │   │   │   ├── Importuje wszystkie moduły
│   │   │   │   └── run_pipeline() - PUBLICZNE API
│   │   │   └── __init__.py
│   │   │       └── [Refaktor Etap 5]
│   │   │
│   │   └── [Ostateczna struktura - Etap 1+]
│   │
│   ├── data/                              [DANE WEJŚCIOWE]
│   │   └── XAU_1m_data_*.csv              [Dane OHLCV]
│   │
│   ├── models/                            [ARTEFAKTY MODELI (OLD)]
│   │   └── [DEPRECATED - przenieść do ml/outputs/models/]
│   │
│   ├── config/                            [KONFIGURACJA PROJEKTU]
│   │   └── [Istniejące pliki]
│   │
│   ├── logs/                              [LOGI (OLD)]
│   │   └── [DEPRECATED - przenieść do ml/outputs/logs/]
│   │
│   └── [Istniejące moduły]
│       ├── analysis/
│       ├── backtesting/
│       ├── features/
│       ├── forecasting/
│       ├── notebooks/
│       ├── scripts/
│       ├── sequences/
│       ├── targets/
│       └── utils/
│
├── scripts/                                [🆕 SKRYPTY WYKONYWALNE]
│   ├── train_sequence_model.py             [🎬 Główny skrypt trenowania]
│   │   └── CLI + args + run_pipeline()
│   ├── eval_model.py                       [📊 Ewaluacja modelu]
│   ├── analyze_features.py                 [🔍 Analiza importance]
│   ├── backtest_strategy.py                [💹 Backtest (opcja)]
│   └── [Etap 5-6]
│
├── outputs/                                [🆕 WYNIKI (SEPARACJA!)]
│   │
│   ├── models/                             [Wytrenowane modele]
│   │   ├── sequence_xgb_model.pkl          [Wytrenowany model]
│   │   ├── sequence_scaler.pkl             [RobustScaler]
│   │   └── sequence_metadata.json          [Metadane]
│   │
│   ├── metrics/                            [Metryki ewaluacji]
│   │   ├── eval_metrics.json               [Precision, recall, F1, etc.]
│   │   └── [eval_*.json dla różnych run]
│   │
│   ├── analysis/                           [Analiza features]
│   │   ├── feature_importance.csv          [Importances]
│   │   ├── feature_importance.png          [Plot]
│   │   └── [Etap 6]
│   │
│   └── logs/                               [Logi z uruchomień]
│       ├── train_2025-12-16_14-30.log      [Log trenowania]
│       └── [eval_*.log dla różnych run]
│
├── tests/                                  [🆕 TESTY JEDNOSTKOWE]
│   ├── conftest.py                         [Pytest fixtures]
│   ├── test_data_loading.py                [Testy data_loading/]
│   ├── test_feature_engineering.py         [Testy features/]
│   ├── test_sequences.py                   [Testy sequences/]
│   ├── test_training.py                    [Testy training/]
│   └── [Etap 7]
│
└── [Inne katalogi projektu]
    ├── docs/
    ├── .github/
    ├── requirements.txt
    └── ...
```

---

## 🔄 Przepływ Danych

```
ml/data/
XAU_1m_data_*.csv
    ↓
[load_all_years()]
    ↓
DataFrame OHLCV
    ↓
[engineer_candle_features()]
    ↓
Features (35 columny)
    ↓
[make_target()]
    ↓
Target (0/1)
    ↓
[create_sequences()]
    ↓
X, y (sequential windows)
    ↓
[split_sequences()]
    ↓
X_train, X_val, X_test + y_train, y_val, y_test
    ↓
[train_xgb()]
    ↓
Trained Model
    ↓
[evaluate()]
    ↓
ml/outputs/models/ + ml/outputs/metrics/
```

---

## 📊 Rozmiary (Szacunkowe)

| Kategoria | Rozmiar | Notatka |
|-----------|---------|---------|
| **Oryginalny plik** | 1740 linii | Monolityczny |
| **Nowy kod (modułowy)** | ~2400 linii | +docstrings, logging, error handling |
| **Moduły** | 20+ | Każdy odpowiada funkcji |
| **Funkcje** | ~43 | Rozbite z oryginalnych 15+ |
| **Dokumentacja** | ~500 linii | Etapy 1-7 + README |
| **Testy** | ~500 linii | Pokrycie > 90% |

---

## ✅ Walidacja & Checklisty

### Etap 1 (Struktura)
- [ ] 12 katalogów stworzone
- [ ] 6 plików `__init__.py` stworzone
- [ ] `config.py` z `PipelineConfig`
- [ ] `sequences/config.py` z `SequenceFilterConfig`
- [ ] `split.py` (szkielet)
- [ ] `data_loading/validators.py` + `loaders.py`
- [ ] Importy działają: `from ml.src.pipelines.data_loading import load_all_years`

### Etapy 2-7
- [ ] Każdy etap ma dokumentację `REFACTOR_ETAP_N.md`
- [ ] Każdy etap ma listę kontrolną w pliku dokumentacji
- [ ] Każdy etap ma testy
- [ ] Każdy etap ma metryki sukcesu

---

## 🎯 Cele & Zalety

### Przed Refactoryzacją
```
❌ 1740 linii w jednym pliku
❌ Trudno znaleźć specyficzną funkcję
❌ Trudno testować poszczególne części
❌ Trudno używać w innych projektach
❌ Brak jasnej separacji wkładu/wyniku
```

### Po Refactoryzacji
```
✅ Kod zorganizowany w 20+ modułach
✅ Łatwo znaleźć, co się szuka
✅ Każdy moduł można testować osobno
✅ Łatwo importować w innych projektach
✅ Jasna separacja: src/ → outputs/
✅ Łatwo dodać nowe funkcje
✅ Łatwo zmienić logikę w konkretnym module
✅ 100% dokumentacji
```

---

## 🚀 Workflow Po Refactoryzacji

### Trening Modelu
```bash
python ml/scripts/train_sequence_model.py \
  --window-size 60 \
  --year-filter 2023 2024 \
  --session london_ny
```

**Wynik**:
```
ml/outputs/
├── models/
│   ├── sequence_xgb_model.pkl
│   ├── sequence_scaler.pkl
│   └── sequence_metadata.json
├── metrics/
│   └── eval_metrics.json
└── logs/
    └── train_2025-12-16_14-30.log
```

### Ewaluacja Modelu
```bash
python ml/scripts/eval_model.py \
  --model-path ml/outputs/models/sequence_xgb_model.pkl
```

### Analiza Features
```bash
python ml/scripts/analyze_features.py \
  --model-path ml/outputs/models/sequence_xgb_model.pkl
```

---

## 📖 Import API Po Refactoryzacji

```python
# W innym projekcie czy notebooku
from ml.src.pipelines import (
    load_all_years,           # z data_loading
    engineer_candle_features, # z features
    make_target,              # z targets
    create_sequences,         # z sequences
    train_xgb,               # z training
    evaluate,                # z training
    run_pipeline,            # główny API
)
from ml.src.pipelines.config import PipelineConfig

# Konfiguracja
config = PipelineConfig()
config.create_directories()

# Użycie
metrics = run_pipeline(
    window_size=60,
    year_filter=[2023, 2024],
    random_state=42
)
```

---

## 🎬 Timeline (Szacunkowy)

| Etap | Opis | Czas | Łącznie |
|------|------|------|---------|
| 0 | Plan & dokumentacja | 2-3h | 2-3h |
| 1 | Katalogi & importy | 1-2h | 3-5h |
| 2 | Features | 2-3h | 5-8h |
| 3 | Targets & sequences | 2-3h | 7-11h |
| 4 | Training & evaluation | 2-3h | 9-14h |
| 5 | Main + CLI | 2-3h | 11-17h |
| 6 | Dodatkowe skrypty | 1-2h | 12-19h |
| 7 | Testy & dokumentacja | 2-3h | 14-22h |

**Szacunkowy całkowity czas**: **2-3 tygodnie** (przy ~1-2h dziennie)

---

## 🎓 Nauka & Best Practices

### Zasady Refactoryzacji
1. ✅ Struktura katalogów PRZED kodem
2. ✅ Każdy moduł = jasna funkcja (SRP)
3. ✅ Importy działają zanim zmienisz logikę
4. ✅ Separacja: wkład → kod → wyjście
5. ✅ Dokumentacja towarzysząca zmianom

### Praktyki Python
1. ✅ Type hints na wszystkich funkcjach
2. ✅ Docstrings (Google style)
3. ✅ Logging zamiast print()
4. ✅ Error handling (nie `except:`)
5. ✅ Test-driven development (TDD)

---

**Status**: 📚 Plan Complete
**Zaznacz w Kalendarzu**: ~2-3 tygodnie na refactoryzację
**Zacznij**: INDEX.md → ROADMAP.md → REFACTOR_PLAN.md → REFACTOR_ETAP_1.md
