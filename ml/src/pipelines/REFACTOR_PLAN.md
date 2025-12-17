# Refactoryzacja `sequence_training_pipeline.py`

## Przegląd
Plik `sequence_training_pipeline.py` ma **1740 linii** i zawiera całą logikę przetwarzania danych, trenowania modelu i ewaluacji. Refactoryzacja ma na celu **rozbicie monolitu na moduły odpowiadające funkcjom domeny**, rozdzielając **skrypty obliczeniowe** od **modułów biblioteki**.

---

## Zasady Refactoryzacji

✅ **Obowiązkowe**:
- Rozdzielenie logiki biznesowej od kodu wykonywalnego (main)
- Wszystkie skrypty obliczeniowe trafiają do `scripts/`
- Wszystkie wyniki (modele, metryki, artefakty) trafiają do `outputs/`
- **Nigdy mieszać wyników ze skryptami**
- Funkcje mogą być importowane z modułów biblioteki
- Zachować kompatybilność API dla istniejących importów

❌ **Zakazane**:
- Hardkodowanie ścieżek plików w kodzie (użyć `Path`, konfiguracja)
- Mieszanie danych wejściowych (`data/`), kodu (`src/`) i wyników (`outputs/`)
- Funkcje w `scripts/` bez możliwości importu z `src/`

---

## Docelowa Struktura Katalogów

```
ml/
├── src/                           # BIBLIOTEKA - kod do importu
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── sequence_training_pipeline.py  # [ZA REFAKTOREM] - wyłącznie publiczne API
│   │   │
│   │   ├── data_loading/          # [NOWY] Moduł: ładowanie i walidacja danych
│   │   │   ├── __init__.py
│   │   │   ├── loaders.py         # load_all_years(), _validate_schema()
│   │   │   └── validators.py      # Schema validation, duplicate checks
│   │   │
│   │   ├── features/              # [NOWY] Moduł: inżynieria cech
│   │   │   ├── __init__.py
│   │   │   ├── engineer.py        # engineer_candle_features() - główna funkcja
│   │   │   ├── indicators.py      # Technical indicators (EMA, RSI, ADX, ATR...)
│   │   │   ├── m5_context.py      # M5 context features (resampling, context)
│   │   │   └── time_features.py   # Hour/minute encoding, temporal features
│   │   │
│   │   ├── targets/               # [NOWY] Moduł: tworzenie celu (target)
│   │   │   ├── __init__.py
│   │   │   └── target_maker.py    # make_target() - backtest-based target
│   │   │
│   │   ├── sequences/             # [NOWY] Moduł: tworzenie sekwencji
│   │   │   ├── __init__.py
│   │   │   ├── sequencer.py       # create_sequences() - słiding windows
│   │   │   ├── filters.py         # Session/trend/pullback filters
│   │   │   └── config.py          # SequenceFilterConfig dataclass
│   │   │
│   │   ├── training/              # [NOWY] Moduł: trening i ewaluacja
│   │   │   ├── __init__.py
│   │   │   ├── xgb_trainer.py     # train_xgb() - trening modelu
│   │   │   ├── evaluation.py      # evaluate(), _pick_best_threshold()
│   │   │   ├── daily_cap.py       # _apply_daily_cap() - limit na dzień
│   │   │   ├── feature_analysis.py # analyze_feature_importance()
│   │   │   └── artifacts.py       # save_artifacts(), model serialization
│   │   │
│   │   ├── utils/                 # [NOWY] Moduł: utylity
│   │   │   ├── __init__.py
│   │   │   └── helpers.py         # Inne funkcje pomocnicze
│   │   │
│   │   ├── config.py              # [NOWY] Centralna konfiguracja pipeline
│   │   │   # PipelineConfig dataclass (path defaults, thresholds, etc.)
│   │   │
│   │   └── split.py               # [NOWY] Moduł: split chronologiczny
│   │       # split_sequences() - chronological train/val/test split
│   │
│   ├── data/                      # DANE WEJŚCIOWE - XAU_1m_data_*.csv
│   │   └── [brak zmian]
│   │
│   ├── models/                    # MODELE - artefakty treningu
│   │   └── [brak zmian - mogą tu być domyślne]
│   │
│   └── config/                    # KONFIGURACJA - pliki yaml/json
│       └── [istniejące pliki]
│
├── scripts/                       # SKRYPTY OBLICZENIOWE - do uruchomienia
│   ├── train_sequence_model.py    # [NOWY] = główny skrypt (refaktor run_pipeline())
│   ├── eval_model.py              # [NOWY] Ewaluacja na danych testowych
│   ├── analyze_features.py        # [NOWY] Analiza feature importance
│   └── ...
│
├── outputs/                       # WYNIKI - artefakty, metryki, logi
│   ├── models/                    # Wytrenowane modele
│   │   ├── sequence_xgb_model.pkl
│   │   ├── sequence_scaler.pkl
│   │   └── sequence_metadata.json
│   │
│   ├── metrics/                   # Metryki ewaluacji
│   │   └── eval_metrics.json
│   │
│   ├── analysis/                  # Analiza features, importances
│   │   └── feature_importance.csv
│   │
│   └── logs/                      # Logi z uruchomień
│       └── train_*.log
│
└── tests/                         # TESTY JEDNOSTKOWE
    ├── test_data_loading.py
    ├── test_feature_engineering.py
    ├── test_sequences.py
    ├── test_training.py
    └── conftest.py
```

---

## Plan Refactoryzacji - Etapy

### **Etap 1: Struktura Katalogów i Moduły Bazowe**
- [x] Stworzenie tego planu
- [ ] Stworzyć katalogi: `data_loading/`, `features/`, `targets/`, `sequences/`, `training/`, `utils/`, `split.py`, `config.py`
- [ ] Stworzyć pliki `__init__.py` w każdym module
- [ ] Przepisać `_validate_schema()` → `data_loading/validators.py`
- [ ] Przepisać `load_all_years()` → `data_loading/loaders.py`
- [ ] Stworzyć `SequenceFilterConfig` w `sequences/config.py`
- [ ] **Test**: Sprawdzić, czy wszystkie importy się kompilują

**Wynik**: Gotowa struktura katalogów, modułowy import

---

### **Etap 2: Inżynieria Cech (`features/`)**
- [ ] Utworzyć `features/indicators.py` - wszystkie indykatory techniczne
  - EMA, ADX, +DI/-DI, MACD, RSI, Stochastic, CCI, Williams %R, ROC, Bollinger Bands, ATR, OBV
- [ ] Utworzyć `features/m5_context.py` - kontekst M5 (resampling, ATR_M5, RSI_M5, SMA200)
- [ ] Utworzyć `features/time_features.py` - kodowanie godziny/minuty
- [ ] Przepisać `engineer_candle_features()` → `features/engineer.py`
  - Refaktoryzować do funkcji zamiast monolitu: bardziej czytelne
- [ ] **Test**: Porównać output z oryginalnym

**Wynik**: Czysty moduł `features/` z testami

---

### **Etap 3: Tworzenie Celu i Sekwencji**
- [ ] Przepisać `make_target()` → `targets/target_maker.py`
- [ ] Przepisać `create_sequences()` → `sequences/sequencer.py`
- [ ] Przepisać filtry sesji/trendu/pullback → `sequences/filters.py`
- [ ] Przepisać `split_sequences()` → `split.py`
- [ ] **Test**: Porównać output sekwencji z oryginalnym

**Wynik**: Moduły do tworzenia celów i sekwencji

---

### **Etap 4: Training i Ewaluacja**
- [ ] Przepisać `train_xgb()` → `training/xgb_trainer.py`
- [ ] Przepisać `evaluate()` → `training/evaluation.py`
- [ ] Przepisać `_pick_best_threshold()` → `training/evaluation.py`
- [ ] Przepisać `_apply_daily_cap()` → `training/daily_cap.py`
- [ ] Przepisać `analyze_feature_importance()` → `training/feature_analysis.py`
- [ ] Przepisać `save_artifacts()` → `training/artifacts.py`
- [ ] **Test**: Trenować mały model i porównać metryki

**Wynik**: Moduły do trenowania i ewaluacji

---

### **Etap 5: Refaktor głównego pliku pipeline**
- [ ] Stworzyć `config.py` - centralna konfiguracja (ścieżki, domyślne wartości)
- [ ] Przepisać `run_pipeline()` w `sequence_training_pipeline.py` - to będzie **publiczne API**
  - Importować wszystkie moduły
  - Główna logika orchestracji
- [ ] Stworzyć `scripts/train_sequence_model.py` - skrypt CLI do uruchomienia
  - Sparsować argumenty
  - Wołać `run_pipeline()` z `src/pipelines`
  - Zapisywać wyniki do `outputs/`
- [ ] **Test**: Uruchomić `python scripts/train_sequence_model.py --help`

**Wynik**: Czysty API pipeline + skrypt CLI

---

### **Etap 6: Dodatkowe Skrypty**
- [ ] Stworzyć `scripts/eval_model.py` - ewaluacja wytrenowanego modelu
- [ ] Stworzyć `scripts/analyze_features.py` - analiza feature importance
- [ ] Stworzyć `scripts/backtest_strategy.py` - backtest ze scenariuszami (opcjonalnie)

**Wynik**: Pełny zestaw skryptów

---

### **Etap 7: Testy Jednostkowe**
- [ ] Testy dla `data_loading/` - walidacja schema, obsługa błędów
- [ ] Testy dla `features/` - porównanie output z oryginalnym
- [ ] Testy dla `targets/` - generowanie celów
- [ ] Testy dla `sequences/` - tworzenie sekwencji, filtry
- [ ] Testy dla `training/` - trening, ewaluacja

**Wynik**: Pokrycie testami > 90%

---

## Zalety Refactoryzacji

✅ **Modularność**: Każdy moduł odpowiada jasnej funkcji (SOLID Single Responsibility)
✅ **Testowalność**: Każdy moduł można testować niezależnie
✅ **Ponowne Użycie**: Funkcje z `src/` mogą być importowane w innych projektach
✅ **Separacja Wkładu/Wyniku**: `data/` (input) vs `outputs/` (results) vs `src/` (code)
✅ **Czytelność**: Kod ponad 1700 linii jest trudny do czytania
✅ **Konserwacja**: Zmiany w logice (np. nowe indykatory) trafiają do konkretnych plików
✅ **Dokumentacja**: Każdy moduł ma jasny zakres

---

## Przełącznik do Refaktoryzacji

| Wersja | Plik | Status | Notatki |
|--------|------|--------|---------|
| `PRZED` | `ml/src/pipelines/sequence_training_pipeline.py` (1740 linii) | Monolityczna | Do refactoryzacji |
| `PO` | Zmodularyzowana architektura | ✏️ W trakcie | Etap 1-7 |

---

## Ostateczny Katalog `src/`

```
src/
├── pipelines/
│   ├── data_loading/
│   ├── features/
│   ├── targets/
│   ├── sequences/
│   ├── training/
│   ├── utils/
│   ├── config.py
│   ├── split.py
│   ├── sequence_training_pipeline.py  # Publiczne API
│   └── __init__.py
├── data/
├── models/
├── config/
├── logs/
└── ...
```

Wyniki trafiają do **`ml/outputs/`**:
- `outputs/models/` - wytrenowane modele
- `outputs/metrics/` - metryki ewaluacji
- `outputs/analysis/` - analiza features
- `outputs/logs/` - logi uruchomień

---

## Następne Kroki

1. ✅ **Zatwierdzenie planu** (ten dokument)
2. ⏭️ **Etap 1**: Struktura katalogów
3. ⏭️ **Etap 2-7**: Stopniowa refactoryzacja

---

**Data**: 2025-12-16
**Autor**: Refactoring Plan
**Status**: Gotowy do Realizacji
