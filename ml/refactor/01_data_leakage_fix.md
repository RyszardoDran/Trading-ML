# Faza 1: Data Leakage Fix

**Status:** Planowanie  
**Priorytet:** 🔴 CRITICAL  
**Wpływ:** +5-15% zawyżenie metryk  

---

## Problem Szczegółowo

### Obecna Sekwencja (BŁĘDNA)

```
load_and_prepare_data(year_filter=[2023, 2024])
    ↓ (data z całego dostępnego zakresu - mogą być inne lata!)
engineer_features_stage(df_m1)
    ↓ (agregacja M1→M5 na WSZYSTKICH danych)
aggregate_to_m5(df_m1)  ← BŁĄD TUTAJ!
    ↓
compute_rsi(period=14)  ← RSI dla 2024 używa 2023!
compute_sma_200()       ← SMA200 dla 2024 używa całej historii!
    ↓
create_targets_stage()
    ↓
build_sequences_stage()
    ↓
split_and_scale_stage(year_filter=[2023, 2024])  ← Split MOŻE nie pokrywać się z feature engineering!
```

### Konsekwencje

1. **RSI (14-period)** na 2024-01-01 jest obliczane na danych z 2023
2. **SMA200** na 2024 zawiera całą historię (mogą być dane z 2022, 2021!)
3. **Test set** otrzymuje sekwencje zawierające dane ze świata, które widział train

### Gdzie jest problem w kodzie

**Plik: `ml/src/features/engineer_m5.py`**

```python
def aggregate_to_m5(df_m1: pd.DataFrame) -> pd.DataFrame:
    """❌ BŁĄD: Brak filtracji daty"""
    
    # Linia 93-109: agregacja bez kontroli roku
    df_m5 = df_m1[required_cols].resample('5min').agg(agg_dict)
    df_m5 = df_m5.dropna()
    
    return df_m5  # Zawiera ALL data!
```

**Plik: `ml/src/pipelines/sequence_training_pipeline.py`**

```python
# Linia 106-110:
df_m1 = load_and_prepare_data(data_dir, year_filter=params.year_filter)  # Może to nie działać!
features = engineer_features_stage(df_m1, ...)  # Agregacja bez roku
# ...
(X_train_scaled, X_val_scaled, X_test_scaled, ...) = split_and_scale_stage(
    X=X,
    y=y,
    timestamps=timestamps,
    year_filter=params.year_filter,  # Split tutaj, ale features bez roku!
)
```

---

## Rozwiązanie

### Krok 1: Refactoruj `aggregate_to_m5()`

**Nowy kod:**

```python
def aggregate_to_m5(df_m1: pd.DataFrame, year_filter: Optional[list[int]] = None) -> pd.DataFrame:
    """Aggregate M1 to M5, optionally filtering by year to prevent data leakage.
    
    **CRITICAL FIX**: Filter BEFORE aggregation to ensure indicators like SMA200
    are calculated only on in-scope data.
    
    Args:
        df_m1: M1 DataFrame with DatetimeIndex
        year_filter: Optional list of years to include (e.g., [2023, 2024])
    
    Returns:
        M5 DataFrame with only data from specified years
    
    Notes:
        - If year_filter is None, uses all available data
        - Filtering happens BEFORE aggregation (prevents data leakage)
        - Aggregation rules: Open=first, High=max, Low=min, Close=last, Volume=sum
    """
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = set(required_cols) - set(df_m1.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # **FIX**: Filter by year FIRST
    if year_filter:
        logger.info(f"Filtering to years: {year_filter}")
        mask = df_m1.index.year.isin(year_filter)
        df_m1_filtered = df_m1[mask]
        
        if df_m1_filtered.empty:
            raise ValueError(f"No data found for years: {year_filter}")
        
        logger.info(f"Data range before filtering: {df_m1.index.min()} to {df_m1.index.max()}")
        logger.info(f"Data range after filtering: {df_m1_filtered.index.min()} to {df_m1_filtered.index.max()}")
    else:
        df_m1_filtered = df_m1
        logger.info(f"No year filter applied. Using all data: {df_m1.index.min()} to {df_m1.index.max()}")
    
    # NOW aggregate only on filtered data
    logger.info(f"Aggregating {len(df_m1_filtered)} M1 candles to M5...")
    
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    df_m5 = df_m1_filtered[required_cols].resample('5min').agg(agg_dict)
    df_m5 = df_m5.dropna()
    
    logger.info(f"Aggregated to {len(df_m5)} M5 candles")
    
    return df_m5
```

### Krok 2: Uaktualnij `engineer_m5_candle_features()`

Dodaj parametr `year_filter`:

```python
def engineer_m5_candle_features(
    df_m1: pd.DataFrame, 
    year_filter: Optional[list[int]] = None  # ← NEW PARAM
) -> pd.DataFrame:
    """Engineer features directly on M5 (5-minute) candles.
    
    **CRITICAL FIX**: year_filter prevents data leakage
    """
    logger.info("Engineering M5 features...")
    
    # **FIX**: Pass year_filter to aggregate function
    df_m5 = aggregate_to_m5(df_m1, year_filter=year_filter)
    
    # ... reszta kodu unchanged
```

### Krok 3: Uaktualnij `sequence_training_pipeline.py`

W funkcji `run_pipeline()`:

```python
def run_pipeline(params: PipelineParams) -> Dict[str, float]:
    """Execute end-to-end sequence XGBoost training pipeline."""
    
    # ... setup code ...
    
    # ===== STAGE 1: Load M1 data =====
    df_m1 = load_and_prepare_data(data_dir, year_filter=params.year_filter)
    
    # ===== STAGE 2: Engineer features (M1→M5 aggregation + feature engineering) =====
    # **FIX**: Pass year_filter to prevent leakage
    features = engineer_features_stage(
        df_m1, 
        window_size=params.window_size, 
        feature_version=params.feature_version,
        year_filter=params.year_filter  # ← ADD THIS
    )
    
    # Get M5 aggregated data for target creation (also with year filter)
    df_m5 = aggregate_to_m5(df_m1, year_filter=params.year_filter)  # ← ADD year_filter
    
    # ... reszta unchanged
```

### Krok 4: Uaktualnij `engineer_features_stage()` w `pipeline_stages.py`

Musisz znaleźć tę funkcję i dodać parametr:

```python
def engineer_features_stage(
    df_m1: pd.DataFrame, 
    window_size: int,
    feature_version: str = "v1",
    year_filter: Optional[list[int]] = None  # ← ADD THIS
) -> pd.DataFrame:
    """Engine features from M1 data."""
    
    features = engineer_m5_candle_features(
        df_m1,
        year_filter=year_filter  # ← PASS IT
    )
    
    return features
```

---

## Walidacja Naprawy

### Test 1: Sprawdzenie daty range

```python
# test_data_leakage.py
import pandas as pd
from ml.src.features.engineer_m5 import aggregate_to_m5

df_m1 = load_data('2023_2024')  # Contains 2023 AND 2024
df_m5 = aggregate_to_m5(df_m1, year_filter=[2024])

# Sprawdzenie
assert df_m5.index.min().year == 2024, "M5 zawiera dane z 2023!"
assert df_m5.index.max().year == 2024, "M5 zawiera dane poza 2024!"

print("✅ Data leakage fix verified")
```

### Test 2: Walidacja wskaźników

```python
# Sprawdzenie że RSI jest obliczane na limitowanej historii
df_m1_full = load_data()  # ALL data
df_m1_2024 = load_data('2024')  # ONLY 2024

df_m5_full = aggregate_to_m5(df_m1_full, year_filter=[2024])
df_m5_2024 = aggregate_to_m5(df_m1_2024, year_filter=[2024])

# Porównaj RSI
# Jeśli używamy tylko 2024 danych → RSI będzie inny niż gdy używamy full data
rsi_full = compute_rsi(df_m5_full['Close'], period=14)
rsi_2024 = compute_rsi(df_m5_2024['Close'], period=14)

# RSI będzie różny! To jest OK i oczekiwane.
# Ważne że obydwa są obliczane KONSYSTENTNIE
```

---

## Checkpoint

Po tej naprawie powinieneś:
- ✅ Mieć parametr `year_filter` w `aggregate_to_m5()`
- ✅ Mieć parametr `year_filter` w `engineer_m5_candle_features()`
- ✅ Mieć parametr `year_filter` w `engineer_features_stage()`
- ✅ Mieć parametr `year_filter` w `run_pipeline()` jako argument
- ✅ Testy walidujące że daty są w expected range

Po wdrożeniu: **Powtórz training** i porównaj metryki (powinny być ~5-10% niższe, ale bardziej realistyczne).

---

## Następny Problem

Gdy skończysz tę naprawę → przejdź do [02_timeseries_cv_fix.md](02_timeseries_cv_fix.md)
