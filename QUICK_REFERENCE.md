# ⚡ QUICK REFERENCE - Refactoryzacja

## 🎯 Cel
Rozbić monolityczny plik `sequence_training_pipeline.py` (1740 linii) na **20+ modułów** w 7 etapach.

---

## 📂 Struktura (PRZED → PO)

### PRZED
```
ml/src/pipelines/
└── sequence_training_pipeline.py (1740 linii, wszystko w jednym pliku)
```

### PO
```
ml/src/pipelines/
├── data_loading/
│   ├── loaders.py (load_all_years)
│   └── validators.py (_validate_schema)
├── features/
│   ├── engineer.py (engineer_candle_features)
│   ├── indicators.py (techniczne indykatory)
│   ├── m5_context.py (kontekst M5)
│   └── time_features.py (kodowanie czasu)
├── targets/
│   └── target_maker.py (make_target)
├── sequences/
│   ├── sequencer.py (create_sequences)
│   ├── filters.py (filtry sesji/trendu)
│   └── config.py (SequenceFilterConfig)
├── training/
│   ├── xgb_trainer.py (train_xgb)
│   ├── evaluation.py (evaluate, threshold)
│   ├── daily_cap.py (_apply_daily_cap)
│   ├── feature_analysis.py (importance)
│   └── artifacts.py (save_artifacts)
├── utils/
├── config.py (PipelineConfig - centralna)
├── split.py (split_sequences)
└── sequence_training_pipeline.py (~150 linii, API)

ml/scripts/
├── train_sequence_model.py (CLI główny)
├── eval_model.py (ewaluacja)
└── analyze_features.py (analiza)

ml/outputs/
├── models/ (wytrenowane modele)
├── metrics/ (metryki ewaluacji)
├── analysis/ (analiza features)
└── logs/ (logi)
```

---

## 🗺️ 7 Etapów (Szybko)

| Etap | Co | Katalogi | Funkcje | Status |
|------|-------|----------|---------|--------|
| **1** | Katalogi & importy | 6 | 2 | ✅ Plan |
| **2** | Inżynieria cech | 1 | ~15 | ⏳ |
| **3** | Cel & sekwencje | 1 | 4 | ⏳ |
| **4** | Training & ewaluacja | 1 | 5 | ⏳ |
| **5** | Refaktor główny + CLI | 1 | 1 + 1 | ⏳ |
| **6** | Dodatkowe skrypty | - | 3 | ⏳ |
| **7** | Testy & docs | 1 | ~10 | ⏳ |

---

## ✅ Zasady (DO/NIE ROBIĆ)

### ✅ DO
- Separacja: `data/` → `src/` → `outputs/`
- Każdy moduł ma `__init__.py`
- Każda funkcja ma docstring
- Importy działają
- Testy dla każdego modułu

### ❌ NIE ROBIĆ
- Mieszać wyniki ze skryptami
- Hardkodować ścieżki
- Zmieniać logikę (do Etapu 7)
- Pracować bez planu
- Robić wiele etapów naraz

---

## 📋 Dokumenty (Gdzie Przeczytać)

| Dokument | Zawartość | Czas | Kiedy |
|----------|-----------|------|-------|
| **INDEX.md** | 📚 Mapa całości | 5 min | PIERWSZY |
| **ROADMAP.md** | 🗺️ Diagram | 5 min | DRUGI |
| **REFACTOR_PLAN.md** | 📋 Szczegóły | 15 min | TRZECI |
| **REFACTOR_ETAP_1.md** | 🏗️ Implementacja | 20 min | PRZED KODEM |

---

## 🚀 Zacznij (3 Kroki)

### Krok 1: Przeczytaj (15 min)
```
1. INDEX.md (5 min)
2. ROADMAP.md (5 min)
3. REFACTOR_PLAN.md (10 min)
```

### Krok 2: Przygotuj się (5 min)
```
Przeczytaj: REFACTOR_ETAP_1.md
```

### Krok 3: Implementuj Etap 1
```bash
# Katalogi
mkdir -p ml/src/pipelines/{data_loading,features,targets,sequences,training,utils}
mkdir -p ml/scripts
mkdir -p ml/outputs/{models,metrics,analysis,logs}

# __init__.py (instrukcje w REFACTOR_ETAP_1.md)
# Przenieś funkcje (instrukcje w REFACTOR_ETAP_1.md)
# Sprawdź importy (instrukcje w REFACTOR_ETAP_1.md)
```

---

## 📞 FAQ (Szybkie Odpowiedzi)

**P: Ile czasu?** O: ~1-2 tygodnie (7 etapów)
**P: Wiele etapów naraz?** O: NIE
**P: Zmieniać logikę?** O: NIE
**P: Gdzie wyniki?** O: Do `ml/outputs/`
**P: Od czego?** O: Od INDEX.md

---

## 🎬 Status

```
✅ PLAN (Etap 0)
- REFACTOR_PLAN.md
- REFACTOR_ETAP_1.md
- ROADMAP.md
- INDEX.md

⏳ ETAP 1 (Katalogi)
- Gotowy do implementacji

⏳ ETAPY 2-7 (Kod)
- Czekają na Etap 1
```

---

## 📍 Pliki Dokumentacji

```
ml/src/pipelines/
├── INDEX.md ..................... 📚 START TUTAJ
├── ROADMAP.md ................... 🗺️
├── REFACTOR_PLAN.md ............. 📋
└── REFACTOR_ETAP_1.md ........... 🏗️ (Etap 1)

ml/
└── REFACTORING_SUMMARY.md ....... 📝 (Podsumowanie)
```

---

## 🎯 Przy Każdym Etapie

1. **Przeczytaj** `REFACTOR_ETAP_N.md`
2. **Stwórz** katalogi i pliki
3. **Przenieś** kod
4. **Sprawdź** importy
5. **Przetestuj** (instrukcje w pliku)
6. **Commit** i przejdź do następnego

---

**Status**: 📚 Plan Complete, Ready to Implement
**Zacznij**: Przeczytaj `INDEX.md`
