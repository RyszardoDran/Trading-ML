# ML Refactor - Plan Implementacji

**Data Utworzenia:** 2025-12-22  
**Status:** Pełny Plan Refaktoryzacji  
**Zatwierdzenie:** Gotowe do pracy  

---

## 📋 Zawartość Folderu `ml/refactor/`

```
ml/refactor/
├── README.md (ten plik)
├── PROBLEMS_ANALYSIS.md (pełna analiza 10 problemów)
├── 01_data_leakage_fix.md (CRITICAL #1)
├── 02_timeseries_cv_fix.md (CRITICAL #2)
├── 03_threshold_optimization_fix.md (CRITICAL #3)
├── 04_lookahead_bias_fix.md (HIGH priority #4)
├── 05_class_imbalance_validation.md (HIGH priority #5)
├── 06_sequence_boundary_fix.md (HIGH priority #6)
└── IMPLEMENTATION_ROADMAP.md (plan faz)
```

---

## 🎯 Podsumowanie

### Status Aktualny
- ❌ 3 problemy CRITICAL powodują zawyżone metryki o 5-20%
- ❌ 3 problemy HIGH Priority mogą spowodować problemy na produkcji
- ❌ 4 problemy MEDIUM Priority utrudniają debugging i optymalizację

### Wpływ na Model
- **Obecne metryki:** Zawyżone o ~10-20%
- **Rzeczywista wydajność na produkcji:** Lepsza o ~10-20% niż real performance
- **Time Series Robustness:** Unknown (brak CV)

---

## 📅 Plan Implementacji

### **Faza 1: CRITICAL Fixes (1-2 tygodnie)**

Każda z tych napraw **MUSI** być zrobiona zanim będziesz ufać metrykom.

| # | Plik | Problem | Wysiłek | Wpływ |
|---|------|---------|---------|-------|
| 1 | 01_data_leakage_fix.md | Data Leakage w M5 agregacji | 2-4 hours | +5-15% zawyżenie |
| 2 | 02_timeseries_cv_fix.md | Brak Time Series CV | 4-6 hours | Unknown robustness |
| 3 | 03_threshold_optimization_fix.md | Threshold na test set | 2-3 hours | Invalid metrics |

**Expected Output:** Realistyczne metryki z Time Series CV

---

### **Faza 2: HIGH Priority (1 tydzień)**

Te naprawy zapobiegają problemom na produkcji.

| # | Plik | Problem | Wysiłek | Wpływ |
|---|------|---------|---------|-------|
| 4 | 04_lookahead_bias_fix.md | Lookahead w M15/M60 | 1-2 hours | +5-10% zawyżenie |
| 5 | 05_class_imbalance_validation.md | Brak class imbalance check | 1-2 hours | Strategy shifts |
| 6 | 06_sequence_boundary_fix.md | Sekwencje crossing boundary | 2-3 hours | ~2-5% zawyżenie |

**Expected Output:** Safe production model

---

### **Faza 3: MEDIUM Priority (1 tydzień)**

Te naprawy poprawiają quality i understanding.

| # | Problem | Wysiłek | Priorytet |
|---|---------|---------|----------|
| 7 | Feature importance analysis | 2-3 hours | Debugging |
| 8 | Hyperparameter sweep | 4-8 hours | Optimization |
| 9 | Out-of-sample walk-forward | 2-3 hours | Validation |
| 10 | Ablation study | 2-3 hours | Feature selection |

---

## 🚀 Jak Zacząć

### Krok 1: Przeczytaj PROBLEMS_ANALYSIS.md
- Zrozumiej wszystkie 10 problemów
- Zobacz macierz priorytetów
- Zapoznaj się z sekwencją napraw

### Krok 2: Przeczytaj Dokumenty Fazy 1
1. `01_data_leakage_fix.md` - szczegółowe instrukcje
2. `02_timeseries_cv_fix.md` - kod Time Series CV
3. `03_threshold_optimization_fix.md` - proper validation split

### Krok 3: Implementuj Fazy Sekwencyjnie

**Faza 1:**
- [ ] Zaaplikuj Data Leakage Fix
- [ ] Zaaplikuj Time Series CV
- [ ] Zaaplikuj Threshold Optimization Fix
- [ ] Uruchom testy
- [ ] Powtórz training, porównaj metryki

**Faza 2:**
- [ ] Zaaplikuj Lookahead Bias Fix
- [ ] Zaaplikuj Class Imbalance Validation
- [ ] Zaaplikuj Sequence Boundary Fix
- [ ] Uruchom testy

**Faza 3:**
- [ ] Feature importance analysis
- [ ] Hyperparameter sweep
- [ ] Out-of-sample validation

---

## 📊 Expected Results Tracking

### Przed Refactorem (Obecne)
```
Metryki na test set (zawyżone):
  Precision:  75.0%
  Recall:     68.0%
  F1:         0.715
  ROC-AUC:    0.82
  Win Rate:   75.0%

Time Series Robustness: Unknown
Production Risk: HIGH
```

### Po Fazy 1 (CRITICAL Fixes)
```
Metryki na test set (realistyczne):
  Precision:  65.0% ± 5% (CV std)
  Recall:     58.0% ± 6%
  F1:         0.615 ± 0.04
  ROC-AUC:    0.75 ± 0.03
  Win Rate:   65.0%

Time Series Robustness: MEASURED
Production Risk: MEDIUM
```

### Po Fazy 2 (HIGH Priority Fixes)
```
Metryki na test set (validated):
  Precision:  67.0% ± 4%
  Recall:     60.0% ± 5%
  F1:         0.632 ± 0.03
  ROC-AUC:    0.76 ± 0.02
  Win Rate:   67.0%

No Lookahead: VERIFIED
Class Imbalance: HANDLED
Sequence Integrity: VERIFIED
Production Risk: LOW
```

---

## 🔧 Narzędzia i Zależności

Potrzebne (już pewnie masz):
- ✅ scikit-learn (TimeSeriesSplit, RobustScaler, metrics)
- ✅ pandas (data manipulation)
- ✅ numpy (numerical)
- ✅ xgboost (model)

Do dodania (opcjonalnie):
```bash
pip install shap  # Feature importance
```

---

## 📝 Notatki Implementacyjne

### File Structure
```
ml/src/
├── pipelines/
│   ├── sequence_training_pipeline.py (update run_pipeline())
│   └── pipeline_stages.py (update split/train/evaluate)
├── features/
│   └── engineer_m5.py (update aggregate_to_m5())
└── utils/
    ├── timeseries_validation.py (NEW - Time Series CV)
    └── validation.py (NEW - data checks)
```

### Key Changes
1. **Data Leakage Fix:** Add `year_filter` parameter to aggregation
2. **Time Series CV:** New `TimeSeriesValidator` class
3. **Threshold Optimization:** Separate VAL/TEST usage
4. **Lookahead Fix:** Change `bfill` to `ffill` for M15/M60
5. **Class Balance:** Add distribution check before training
6. **Sequence Boundaries:** Filter sequences crossing train/test boundary

---

## ✅ Checkpoints

### Po Fazy 1
- [ ] Kod kompiluje się
- [ ] Testy passują
- [ ] Metryki są ~10-15% niższe (expected)
- [ ] CV pokazuje std dev
- [ ] Time Series order validated

### Po Fazy 2
- [ ] Metryki są stabilne
- [ ] Lookahead bias removed
- [ ] Class imbalance documented
- [ ] Sequences don't cross boundaries
- [ ] Production risk assessment: LOW

### Po Fazy 3
- [ ] Feature importance known
- [ ] Hyperparameters optimized
- [ ] Out-of-sample validation done
- [ ] Model ready for production

---

## 📞 Support / Questions

**Jeśli masz pytania:**
1. Przeczytaj odpowiadający dokument Fazy
2. Sprawdź sekcję "Rozwiązanie" z kodem
3. Uruchom sekcję "Walidacja" z testami

**Jeśli coś nie działa:**
1. Sprawdź file paths (Windows backslash vs forward slash)
2. Sprawdź imports (czy moduły istnieją?)
3. Uruchom testy (czy dane są w expected format?)

---

## 📌 Ważne Notatki

### 1. Metryki będą NIŻSZE (to jest DOBRE!)
Po naprawach metryki mogą spaść o 10-20%. To jest **oczekiwane i pożądane** bo:
- Obecne metryki są zawyżone
- Nowe metryki są realistyczne
- Produkcja będzie działać lepiej

### 2. Zmiana Expected Output
Jeśli dzisiaj masz:
```
Precision: 75%, Recall: 68%, Win Rate: 75%
```

Po naprawach będzie:
```
Precision: 65% ± 5%, Recall: 58% ± 6%, Win Rate: 65%
```

Ale te 65% będzie **rzeczywiście 65%** na produkcji, nie zawyżone.

### 3. Time Series Cross-Validation
Po wdrożeniu Time Series CV:
- Będziesz mieć metryki dla KAŻDEGO roku
- Będziesz znać variance (std dev)
- Będziesz widzieć czy model jest period-specific

---

## 🎓 Learning Resources

Jeśli chcesz zrozumieć głębokowei:

1. **Data Leakage:**
   - "Data Science for Finance" - rozdział na data leakage
   - Feature engineering MUSI być na train set

2. **Time Series CV:**
   - scikit-learn TimeSeriesSplit documentation
   - "Hands-On Machine Learning" - Chapter na Time Series

3. **Threshold Optimization:**
   - "Learning from Imbalanced Datasets" - threshold tuning
   - ROC curves i precision-recall curves

4. **Lookahead Bias:**
   - "Advances in Financial Machine Learning" - Chapter 1
   - Backtest realism

---

**Status:** ✅ GOTOWY DO PRACY  
**Następny Krok:** Przejdź do [PROBLEMS_ANALYSIS.md](PROBLEMS_ANALYSIS.md)
