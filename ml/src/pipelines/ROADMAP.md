# 📋 Przegląd Refactoryzacji - PLAN WIZUALNY

## Monolityczny Kod → Architektura Modułowa

```
PRZED (1740 linii)                          PO (Rozłożone na moduły)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sequence_training_pipeline.py
├─ _validate_schema()              →    data_loading/validators.py
├─ load_all_years()                →    data_loading/loaders.py
├─ engineer_candle_features()      →    features/engineer.py
│  ├─ indicators (EMA, ADX, etc)   →    features/indicators.py
│  ├─ M5 context                   →    features/m5_context.py
│  └─ time features                →    features/time_features.py
├─ make_target()                   →    targets/target_maker.py
├─ create_sequences()              →    sequences/sequencer.py
├─ filter_by_session()             →    sequences/filters.py
├─ SequenceFilterConfig            →    sequences/config.py
├─ split_sequences()               →    split.py
├─ train_xgb()                     →    training/xgb_trainer.py
├─ evaluate()                      →    training/evaluation.py
├─ _pick_best_threshold()          →    training/evaluation.py
├─ _apply_daily_cap()              →    training/daily_cap.py
├─ analyze_feature_importance()    →    training/feature_analysis.py
├─ save_artifacts()                →    training/artifacts.py
└─ run_pipeline()                  →    sequence_training_pipeline.py
                                        [API publiczne, orchestracja]
```

---

## 📂 Struktura Katalogów - Separacja Wkładu/Wyniku

```
ml/
├── src/                           ← KOD (biblioteka do importu)
│   └── pipelines/
│       ├── data_loading/          ✨ [1] Ładowanie danych
│       ├── features/              ✨ [2] Inżynieria cech
│       ├── targets/               ✨ [3] Tworzenie celu
│       ├── sequences/             ✨ [3] Tworzenie sekwencji
│       ├── training/              ✨ [4] Training/ewaluacja
│       ├── utils/                 ✨ [4] Utylity
│       ├── config.py              ✨ [1] Konfiguracja
│       ├── split.py               ✨ [1] Split chronologiczny
│       └── sequence_training_pipeline.py (główny API)
│
├── data/                          ← DANE WEJŚCIOWE (XAU_1m_data_*.csv)
│
├── scripts/                       ✨ [5] SKRYPTY WYKONYWALNE
│   ├── train_sequence_model.py    ← main CLI do trenowania
│   ├── eval_model.py              ← ewaluacja
│   ├── analyze_features.py        ← analiza
│   └── ...
│
├── outputs/                       ✨ [1] WYNIKI (artefakty)
│   ├── models/                    ← Wytrenowane modele
│   │   ├── sequence_xgb_model.pkl
│   │   ├── sequence_scaler.pkl
│   │   └── sequence_metadata.json
│   ├── metrics/                   ← Metryki ewaluacji
│   │   └── eval_metrics.json
│   ├── analysis/                  ← Analiza features
│   │   └── feature_importance.csv
│   └── logs/                      ← Logi
│       └── train_*.log
│
└── tests/                         ← TESTY JEDNOSTKOWE
    └── test_*.py
```

---

## 🎯 7-Etapowy Plan Refactoryzacji

```
┌─────────────────────────────────────────────────────────────────┐
│ ETAP 1: Struktura Katalogów & Importy                           │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Katalogi: data_loading/, features/, targets/, sequences/     │
│ ✅ Pliki: __init__.py, config.py, split.py                      │
│ ✅ Funkcje: _validate_schema, load_all_years, SequenceFilterConfig │
│ ⏳ Rezultat: Szkielet gotowy, importy działają                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ETAP 2: Inżynieria Cech (features/)                             │
├─────────────────────────────────────────────────────────────────┤
│ ✅ features/indicators.py - wszystkie indykatory techniczne     │
│ ✅ features/m5_context.py - kontekst M5 (resampling)           │
│ ✅ features/time_features.py - kodowanie godziny/minuty         │
│ ✅ features/engineer.py - engineer_candle_features()           │
│ ⏳ Test: Porównanie output z oryginalnym                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ETAP 3: Cel & Sekwencje (targets/, sequences/)                  │
├─────────────────────────────────────────────────────────────────┤
│ ✅ targets/target_maker.py - make_target()                      │
│ ✅ sequences/sequencer.py - create_sequences()                  │
│ ✅ sequences/filters.py - filter_by_session(), trend filters   │
│ ✅ split.py - split_sequences()                                 │
│ ⏳ Test: Porównanie sekwencji z oryginalnym                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ETAP 4: Training & Ewaluacja (training/)                        │
├─────────────────────────────────────────────────────────────────┤
│ ✅ training/xgb_trainer.py - train_xgb()                        │
│ ✅ training/evaluation.py - evaluate(), _pick_best_threshold() │
│ ✅ training/daily_cap.py - _apply_daily_cap()                  │
│ ✅ training/feature_analysis.py - analyze_feature_importance() │
│ ✅ training/artifacts.py - save_artifacts()                    │
│ ⏳ Test: Trening modelu, porównanie metryk                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ETAP 5: Refaktor Główny Plik & Skrypty CLI                      │
├─────────────────────────────────────────────────────────────────┤
│ ✅ sequence_training_pipeline.py - refaktor run_pipeline()     │
│    (centralna orchestracja, importuje wszystkie moduły)         │
│ ✅ scripts/train_sequence_model.py - CLI do trenowania         │
│    (wołaj run_pipeline(), zapisz do outputs/)                  │
│ ⏳ Test: python scripts/train_sequence_model.py --help         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ETAP 6: Dodatkowe Skrypty                                        │
├─────────────────────────────────────────────────────────────────┤
│ ✅ scripts/eval_model.py - ewaluacja wytrenowanego modelu      │
│ ✅ scripts/analyze_features.py - analiza feature importance    │
│ ✅ scripts/backtest_strategy.py - backtest scenariuszy (opcja) │
│ ⏳ Test: Każdy skrypt uruchamia się bez błędów                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ETAP 7: Testy Jednostkowe & Dokumentacja                        │
├─────────────────────────────────────────────────────────────────┤
│ ✅ tests/test_data_loading.py - testy walidacji, obsługi błędów │
│ ✅ tests/test_feature_engineering.py - porównanie output        │
│ ✅ tests/test_sequences.py - tworzenie sekwencji, filtry       │
│ ✅ tests/test_training.py - trening, ewaluacja                 │
│ ⏳ Test: Pokrycie > 90%, wszystkie testy zielone               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Liczba Linii Kodu (Szacunkowa)

| Moduł | Linie | Funkcje | Status |
|-------|-------|---------|--------|
| `data_loading/validators.py` | ~50 | 1 | Etap 1 |
| `data_loading/loaders.py` | ~80 | 1 | Etap 1 |
| `features/engineer.py` | ~300 | 1 | Etap 2 |
| `features/indicators.py` | ~250 | ~15 | Etap 2 |
| `features/m5_context.py` | ~150 | ~5 | Etap 2 |
| `features/time_features.py` | ~50 | ~3 | Etap 2 |
| `targets/target_maker.py` | ~200 | 1 | Etap 3 |
| `sequences/sequencer.py` | ~300 | 1 | Etap 3 |
| `sequences/filters.py` | ~150 | ~5 | Etap 3 |
| `sequences/config.py` | ~30 | 0 | Etap 1 |
| `split.py` | ~80 | 1 | Etap 1 |
| `training/xgb_trainer.py` | ~100 | 1 | Etap 4 |
| `training/evaluation.py` | ~150 | 2 | Etap 4 |
| `training/daily_cap.py` | ~50 | 1 | Etap 4 |
| `training/feature_analysis.py` | ~100 | 1 | Etap 4 |
| `training/artifacts.py` | ~80 | 1 | Etap 4 |
| `sequence_training_pipeline.py` | ~150 | 1 | Etap 5 |
| `config.py` | ~60 | 1 | Etap 1 |
| `scripts/train_sequence_model.py` | ~100 | 1 | Etap 5 |
| `scripts/eval_model.py` | ~80 | 1 | Etap 6 |
| `scripts/analyze_features.py` | ~60 | 1 | Etap 6 |
| **RAZEM** | **~2400** | **~43** | - |

> **Uwaga**: Całość kodu będzie ~2400 linii (dodatkowe linie z docstrings, logowaniem, error handlingiem), lepiej sorganizowana na etap.

---

## ✅ Zasady Refactoryzacji (DO PAMIĘTANIA)

### 🚫 Zakazane
- ❌ Mieszać wyniki (`ml/outputs/`) ze skryptami (`ml/scripts/`)
- ❌ Hardkodować ścieżki plików w kodzie
- ❌ Umieszczać dane wejściowe w katalogach `scripts/` czy `outputs/`
- ❌ Pracować bez testów
- ❌ Zmieniać logikę w trakcie refactoryzacji (do etapu 7)

### ✅ Obowiązkowe
- ✅ Każdy moduł w `src/` musi mieć `__init__.py`
- ✅ Każda funkcja publiczna musi mieć docstring z typami
- ✅ Importy z `src/` muszą działać
- ✅ Katalogi tworzyć przed kodem
- ✅ Separacja: `src/` (kod) vs `outputs/` (wyniki)
- ✅ Skrypty w `scripts/` wołają funkcje z `src/`

---

## 🎬 Jak Zacząć

### Etap 1 (Teraz)
```bash
# 1. Przeczytaj REFACTOR_PLAN.md (przegląd całości)
# 2. Przeczytaj REFACTOR_ETAP_1.md (szczegóły Etapu 1)
# 3. Implementuj Etap 1:

# Katalogi
mkdir -p ml/src/pipelines/{data_loading,features,targets,sequences,training,utils}
mkdir -p ml/scripts
mkdir -p ml/outputs/{models,metrics,analysis,logs}

# Pliki __init__.py
touch ml/src/pipelines/data_loading/__init__.py
# ... itd
```

### Etapy 2-7
- Każdy etap ma dokumentację `REFACTOR_ETAP_N.md`
- Każdy etap można wykonać niezależnie po Etapie 1
- Rekomendacja: 1 etap na commit, 1-2 etapy na dzień

---

## 📌 Gdzie Znaleźć Informacje

| Dokument | Zawartość |
|----------|-----------|
| `REFACTOR_PLAN.md` | 📋 Przegląd całej refactoryzacji (7 etapów) |
| `REFACTOR_ETAP_1.md` | 🔍 Szczegóły Etapu 1 (struktura katalogów) |
| `REFACTOR_ETAP_2.md` | 🔍 Szczegóły Etapu 2 (inżynieria cech) |
| `REFACTOR_ETAP_3.md` | 🔍 Szczegóły Etapu 3 (cel & sekwencje) |
| `REFACTOR_ETAP_4.md` | 🔍 Szczegóły Etapu 4 (training & ewaluacja) |
| `REFACTOR_ETAP_5.md` | 🔍 Szczegóły Etapu 5 (refaktor główny + skrypty CLI) |
| `REFACTOR_ETAP_6.md` | 🔍 Szczegóły Etapu 6 (dodatkowe skrypty) |
| `REFACTOR_ETAP_7.md` | 🔍 Szczegóły Etapu 7 (testy + dokumentacja) |
| **ROADMAP.md** (ten plik) | 🗺️ Wizualny przegląd i roadmap |

---

## 🎯 Cele Refactoryzacji

✅ **Modularność** - Każdy moduł odpowiada jasnej funkcji (SOLID)
✅ **Testowalność** - Każdy moduł można testować niezależnie
✅ **Ponowne Użycie** - Funkcje z `src/` importowalne w innych projektach
✅ **Separacja Wkładu/Wyniku** - `data/` → `src/` → `outputs/`
✅ **Czytelność** - 1740 linii → 43 funkcje w 20 plikach
✅ **Konserwacja** - Zmiany w logice trafiają do konkretnych plików
✅ **Dokumentacja** - Każdy moduł ma jasny zakres

---

**Status**: ⏳ Gotowy do Implementacji
**Ostatnia Aktualizacja**: 2025-12-16
**Autor**: Refactoring Plan
