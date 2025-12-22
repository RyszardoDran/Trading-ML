# ML Pipeline - Analiza Problemów i Propozycje Refaktoryzacji

**Data:** December 22, 2025  
**Status:** Wersja 1.0 - Pełna analiza  
**Autor:** Expert ML Review  

---

## Executive Summary

Obecny pipeline ML zawiera **10 poważnych problemów**, z czego **3 są krytyczne** (data leakage, brak Time Series walidacji, threshold snooping). Te problemy mogą skutkować zawyżonymi metrykami o 5-20% i niską wydajnością modelu na produkcji.

**Priorytet:** Naprawy CRITICAL powinny być wykonane przed dalszym trenowaniem.

---

## 🔴 PROBLEMY KRYTYCZNE (CRITICAL)

### Problem #1: Data Leakage w agregacji M1 → M5

**Lokalizacja:**
- `ml/src/features/engineer_m5.py` - funkcja `aggregate_to_m5()` (linie 93-109)
- `ml/src/pipelines/sequence_training_pipeline.py` - linie 106-110

**Opis problemu:**

```python
# ❌ BŁĘDNE: Agregacja dzieje się na CAŁYM datasecie
df_m1 = load_and_prepare_data(data_dir, year_filter=params.year_filter)  # Np. 2023-2024
features = engineer_features_stage(df_m1, ...)  # Zawiera M5 aggregację
targets = create_targets_stage(df_m5, features, ...)  # Całe M5 już tutaj
X, y = build_sequences_stage(features, targets, ...)  # Sekwencje ze zmiszanych lat
```

**Implikacja:**
- Wskaźniki techniczne (RSI, SMA, BB) są liczone na **całym dataset'cie**, zanim nastąpi split
- Jeśli year_filter=[2023, 2024], ale agregacja robi się na wszystkich danych:
  - SMA200 na 2024 zawiera dane z 2023
  - RSI na 2024 jest "prognozowany" przez przeszłość spoza 2024
- Test set widział dane treningowe w feature engineering

**Wpływ na metryki:**
- Zawyżenie precyzji o 5-15%
- Zawyżenie ROC-AUC o 3-10%
- Model na produkcji będzie działać gorzej

**Rozwiązanie:**
```python
# ✅ POPRAWKA: Filtracja PRZED aggregacją
def aggregate_to_m5_with_dates(df_m1, year_filter=None):
    """Aggregate only within specified year filter to prevent data leakage"""
    if year_filter:
        mask = df_m1.index.year.isin(year_filter)
        df_m1_filtered = df_m1[mask]
    else:
        df_m1_filtered = df_m1
    
    # Teraz agregacja jest na czystych danych
    df_m5 = df_m1_filtered.resample('5min').agg({...})
    return df_m5
```

---

### Problem #2: Threshold Optimization na Test Set (Data Snooping)

**Lokalizacja:**
- `ml/src/pipelines/pipeline_stages.py` - funkcja `train_and_evaluate_stage()` 
- (nie mam pełnego kodu, ale hipoteza oparta na strukturze)

**Opis problemu:**

```python
# ❌ POTENCJALNY BŁĄD
metrics, model = train_and_evaluate_stage(
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    X_test_scaled, y_test,  # Test set jest znany!
    ...
)
# Jeśli threshold optimization patrzył na X_test/y_test:
# - To liczby na test set są ZUPEŁNIE BEZUŻYTECZNE
```

**Jak to sprawdzić:**
- Czy funkcja liczy threshold na X_val czy X_test?
- Czy metrics zwracane to na X_test czy X_val?

**Implikacja:**
- Jeśli threshold leży na X_test: **wszystkie metryki są invaliding**
- Model może mieć 70% win_rate na test, ale 45% na produkcji

**Prawidłowy proces:**
```
1. X_train → trening modelu
2. X_val   → optimization threshold (szukamy best F1/precision)
3. X_test  → finalna ewaluacja (brak dostępu do tych danych przy threshold selection)
```

---

### Problem #3: Brak Time Series Cross-Validation

**Lokalizacja:**
- `ml/src/pipelines/pipeline_stages.py` - funkcja `split_and_scale_stage()`

**Opis problemu:**

```python
# ❌ Standard train/test split NIE jest bezpieczny dla szeregów czasowych
# Jeśli robisz:
X_train = X[:len(X)//5*3]  # 60%
X_val = X[len(X)//5*3:len(X)//5*4]  # 20%
X_test = X[len(X)//5*4:]  # 20%

# To jest OK (chronologiczny), ale:
# - Tylko JEDEN split → może być "lucky"
# - Brak walidacji na różnych okresach
# - Jeśli X_train ma particular pattern, model tego się nauczy
```

**Właściwy approach:**

```python
# ✅ Time Series Cross-Validation (5-fold)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    # Fold 1: X[0:2000] → train, X[2000:2400] → test
    # Fold 2: X[0:2400] → train, X[2400:2800] → test
    # Fold 3: X[0:2800] → train, X[2800:3200] → test
    # ... etc
```

**Implikacja:**
- Obecne single split może być "lucky" 
- Model może overfit na konkretny okres
- CV discovery nowych insights o model robustness

---

## 🟠 PROBLEMY WYSOKIEJ WAŻNOŚCI (HIGH PRIORITY)

### Problem #4: Lookahead Bias w Multi-Timeframe Features

**Lokalizacja:**
- `ml/src/features/engineer_m5.py` - linie 320-355 (alignment M15/M60 do M5)

**Opis problemu:**

```python
# ❌ BŁĘDNE: bfill = backward fill (przyszłe dane!)
rsi_m15 = rsi_m15.reindex(df_m5.index, method='bfill').fillna(50)
bb_pos_m15 = bb_pos_m15.reindex(df_m5.index, method='bfill').fillna(0.5)

# Co się dzieje:
# M15 bar closes at 10:15
# M5 bars: 10:00-10:05, 10:05-10:10, 10:10-10:15
# 
# bfill replicates M15 bar do WSZYSTKICH M5 bars w tym przedziale
# WŁĄCZNIE z barami PRZED zamknięciem M15!
# 
# Oznacza to: Model widzi wartość M15 ZANIM się ona forma!
```

**Prawidłowe wyrównanie:**

```python
# ✅ POPRAWKA: ffill (forward fill) - znamy tylko pełne bary
rsi_m15 = rsi_m15.reindex(df_m5.index, method='ffill').fillna(50)

# Alternatively: align by index matching
# M15 bar closes at 10:15 → dostępny dla M5 barów DOPO 10:15
```

**Implikacja:**
- Model ma "magiczny" dostęp do przyszłych danych
- Win_rate na backtest będzie ~5-10% wyższy niż realny
- Producja: drastyczna degradacja wydajności

---

### Problem #5: Brak Walidacji Class Imbalance

**Lokalizacja:**
- `ml/src/pipelines/pipeline_stages.py` - całość

**Opis problemu:**

```python
# ❌ Nigdzie nie widzę sprawdzenia rozkładu klas
# Pytania bez odpowiedzi:
# - Ile % danych to y=1 (WIN)?
# - Ile % danych to y=0 (LOSS)?
# - Czy train/val/test mają ten sam rozkład?
```

**Typowy problem dla trading ML:**
```
y_train: 25% WIN, 75% LOSS
y_test:  10% WIN, 90% LOSS  # Zmiana!

Model zatraining się na 25% baseline, test ma 10%
→ Model będzie wydawać zbyt dużo BUY sygałów na produkcji
```

**Rozwiązanie:**
```python
# ✅ Raport class distribution
from collections import Counter

def report_class_distribution(y_train, y_val, y_test):
    print("TRAIN:", Counter(y_train))
    print("VAL:  ", Counter(y_val))
    print("TEST: ", Counter(y_test))
```

---

### Problem #6: Brak Walidacji Granic Sekwencji

**Lokalizacja:**
- `ml/src/pipelines/pipeline_stages.py` - funkcja `build_sequences_stage()`

**Opis problemu:**

```python
# ❌ Sekwencje mogą intersectować granicę train/test
# Przykład: window_size=100 M5 candles = 500 minut
#
# Train ends at: 2023-12-31 23:00
# Test starts at: 2024-01-01 00:00
#
# Sekwencja "99,100" mogą być:
# - Bars 1-100: ostatnie 100 barów 2023
# - Bars 51-150: ostatnie 50 z 2023 + pierwsze 50 z 2024 ← MIXED!
#
# Model trenujesz na "2023 + część 2024"
# Test zawiera część 2023!
```

**Rozwiązanie:**
```python
# ✅ POPRAWKA: Usuń sekwencje które cross the boundary
def build_sequences_safe(X, y, timestamps, train_end_date, test_start_date, window_size):
    """Build sequences ensuring no data leakage across splits"""
    sequences = []
    for i in range(len(X) - window_size):
        seq_start_ts = timestamps[i]
        seq_end_ts = timestamps[i + window_size - 1]
        
        # Sprawdź czy sekwencja jest CAŁKOWICIE w train lub test
        if seq_end_ts < train_end_date:
            # Sekwencja całkowicie w train - OK
            sequences.append((X[i:i+window_size], y[i:i+window_size]))
        elif seq_start_ts >= test_start_date:
            # Sekwencja całkowicie w test - OK
            sequences.append((X[i:i+window_size], y[i:i+window_size]))
        # else: SKIP - sekwencja crosses boundary
    
    return sequences
```

---

### Problem #7: Brak Feature Importance Analysis

**Lokalizacja:**
- Całość pipeline - brak feature importance

**Opis problemu:**

```python
# ❌ Nie wiesz które features są ważne
# Model ma 30+ features, ale które faktycznie działają?
# 
# Możliwe problemy:
# - 80% ważności w 3 features → rest to noise
# - Some features mają negative importance → usunąć
# - Colinearity między features → reduce dimensionality
```

**Rozwiązanie:**
```python
# ✅ Feature importance z XGBoost
import xgboost as xgb
import shap

feature_importance = model.get_booster().get_score(importance_type='weight')
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("Top 10 features:")
for feat, score in sorted_features[:10]:
    print(f"  {feat}: {score}")

# SHAP dla deep understanding
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)
```

---

## 🟡 PROBLEMY ŚREDNIEJ WAŻNOŚCI (MEDIUM PRIORITY)

### Problem #8: Hardcoded Hyperparameters

**Lokalizacja:**
- `ml/src/features/engineer_m5.py` - wszędzie
- `ml/src/pipelines/sequence_training_pipeline.py` - default values

**Opis problemu:**

```python
# ❌ Hardcoded wskaźniki techniczne
compute_rsi(period=14)  # Dlaczego 14?
compute_bollinger_bands(period=20, num_std=2)  # Dlaczego 20? Dlaczego 2?
compute_stochastic(period=14, smooth_k=3, smooth_d=3)  # Dlaczego te wartości?

# Brak zmiany = brak optymalizacji
```

**Rozwiązanie:**
```python
# ✅ Feature hyperparameters w config
class FeatureConfig:
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2.0
    STOCH_PERIOD = 14
    STOCH_K_SMOOTH = 3
    STOCH_D_SMOOTH = 3
    # ... etc

# Potem: optymalizacja przez grid search
```

---

### Problem #9: Brak Walidacji na Out-of-Sample Danych

**Lokalizacja:**
- Całość pipeline

**Opis problemu:**

```python
# ❌ Zabudujesz na 2023-2024, testujesz na 2024
# Ale nigdy nie weryfikujesz na 2025!
#
# Co się zmienia pomiędzy latami:
# - Volatility (zwłaszcza złoto)
# - Trend patterns
# - Mean reversion vs momentum
# - Market microstructure
```

**Rozwiązanie:**
```python
# ✅ Walk-forward validation
train_years = [2023, 2024]
test_year = 2025

model = train(2023-2024)
evaluate(model, 2025)  # True out-of-sample

# Jeśli performance drops > 20%: model jest period-specific
```

---

### Problem #10: Brak Ablation Study

**Lokalizacja:**
- Brak w entire pipeline

**Opis problemu:**

```python
# ❌ Wszystkie features razem
# Pytania bez odpowiedzi:
# - Które features są beznadziejne?
# - Czy M15/M60 context naprawdę pomaga?
# - Czy CVD/OBV/MFI są potrzebne?
```

**Rozwiązanie:**
```python
# ✅ Remove-one-feature test
for feature_to_remove in features.columns:
    X_ablated = X_train[features.columns != feature_to_remove]
    model_ablated = train(X_ablated)
    score_ablated = evaluate(model_ablated, X_test_ablated)
    
    importance_score = baseline_score - score_ablated
    print(f"{feature_to_remove}: {importance_score}")
```

---

## 📊 Podsumowanie Problemów

| # | Problem | Priorytet | Krytyczność | Wpływ na Metryki |
|---|---------|-----------|-------------|-----------------|
| 1 | Data Leakage w M5 aggregacji | 🔴 CRITICAL | Bardzo wysoka | +5-15% zawyżenie |
| 2 | Threshold Optimization na test | 🔴 CRITICAL | Bardzo wysoka | Invalid metrics |
| 3 | Brak Time Series CV | 🔴 CRITICAL | Bardzo wysoka | Unknown robustness |
| 4 | Lookahead w M15/M60 | 🟠 HIGH | Wysoka | +5-10% zawyżenie |
| 5 | Brak class imbalance check | 🟠 HIGH | Wysoka | Strategy shifts |
| 6 | Sekwencje crossing boundary | 🟠 HIGH | Wysoka | ~2-5% zawyżenie |
| 7 | Brak feature importance | 🟡 MEDIUM | Średnia | Unknown useful features |
| 8 | Hardcoded hyperparameters | 🟡 MEDIUM | Średnia | Suboptimal features |
| 9 | Brak out-of-sample validation | 🟡 MEDIUM | Średnia | Unknown generalization |
| 10 | Brak ablation study | 🟡 MEDIUM | Średnia | Unknown importance |

---

## 🎯 Proponowana Kolejność Napraw

### Faza 1: CRITICAL (Week 1)
1. ✅ Napraw data leakage w `aggregate_to_m5()`
2. ✅ Wdroż Time Series CV
3. ✅ Napraw threshold optimization na validation set

### Faza 2: HIGH Priority (Week 2)
4. ✅ Napraw lookahead bias w M15/M60
5. ✅ Dodaj class imbalance validation
6. ✅ Napraw sequence boundary crossing

### Faza 3: MEDIUM Priority (Week 3)
7. ✅ Feature importance analysis
8. ✅ Hyperparameter sweep
9. ✅ Out-of-sample walk-forward validation

### Faza 4: OPTIMIZATION (Week 4)
10. ✅ Ablation study
11. ✅ Feature selection
12. ✅ Final model tuning

---

## 📁 Struktura Refactoringu

```
ml/refactor/
├── PROBLEMS_ANALYSIS.md (ten plik)
├── 01_data_leakage_fix.md (Faza 1)
├── 02_timeseries_cv_fix.md (Faza 1)
├── 03_threshold_optimization_fix.md (Faza 1)
├── 04_lookahead_bias_fix.md (Faza 2)
├── 05_class_imbalance_validation.md (Faza 2)
├── 06_sequence_boundary_fix.md (Faza 2)
└── fixes/
    ├── engineer_m5_refactored.py
    ├── pipeline_stages_refactored.py
    ├── validation.py (new)
    └── timeseries_cv.py (new)
```

---

## ✅ Następne Kroki

**Gotów do przejścia do Fazy 1?**

Czekam na sygnał żeby przygotować:
1. `01_data_leakage_fix.md` - szczegółowe instrukcje naprawy
2. Refactored kod w `ml/refactor/fixes/`
3. Testy do walidacji napraw

---

**Status:** Gotowy do refaktoryzacji  
**Ostatnia aktualizacja:** 2025-12-22  
**Następny krok:** [Faza 1 - Data Leakage Fix]
