# 🚀 ETAP 1 - ACTION PLAN (Gotowy do Wdrożenia)

## Status Quo
- ✅ `ml/src/features/` istnieje (PUSTY)
- ✅ `ml/src/targets/` istnieje (PUSTY)
- ✅ `ml/src/sequences/` istnieje (PUSTY)
- ✅ `ml/src/utils/` istnieje (PUSTY)
- ✅ `ml/src/pipelines/sequence_training_pipeline.py` (1740 linii - do refactoryzacji)

## Co Robimy w Etapie 1

### Faza 1: Stwórz Katalogi (Nowe) & __init__.py

```bash
# 1. Nowy katalog
mkdir -p ml/src/data_loading
mkdir -p ml/outputs/models ml/outputs/metrics ml/outputs/analysis ml/outputs/logs

# 2. __init__.py w istniejących katalogach (jeśli brak)
touch ml/src/features/__init__.py
touch ml/src/targets/__init__.py
touch ml/src/sequences/__init__.py
touch ml/src/utils/__init__.py
touch ml/src/data_loading/__init__.py
```

### Faza 2: Przenieś Funkcje z `sequence_training_pipeline.py`

#### Do `ml/src/data_loading/validators.py`:
```python
# Przenieść: _validate_schema()
```

#### Do `ml/src/data_loading/loaders.py`:
```python
# Przenieść: load_all_years()
```

#### Do `ml/src/sequences/config.py`:
```python
# Przenieść: SequenceFilterConfig
```

#### Do `ml/src/pipelines/config.py`:
```python
# Stwórz: PipelineConfig (centralna konfiguracja)
```

#### Do `ml/src/pipelines/split.py`:
```python
# Przenieść: split_sequences()
```

### Faza 3: Zaktualizuj __init__.py Pliki

**`ml/src/data_loading/__init__.py`**:
```python
"""Data loading and validation module."""
from .loaders import load_all_years
from .validators import validate_schema

__all__ = ["load_all_years", "validate_schema"]
```

**`ml/src/features/__init__.py`**:
```python
"""Feature engineering module."""
__all__ = []  # będzie w Etapie 2
```

**`ml/src/targets/__init__.py`**:
```python
"""Target creation module."""
__all__ = []  # będzie w Etapie 3
```

**`ml/src/sequences/__init__.py`**:
```python
"""Sequence creation module."""
from .config import SequenceFilterConfig

__all__ = ["SequenceFilterConfig"]  # reszta w Etapie 3
```

**`ml/src/utils/__init__.py`**:
```python
"""Utility functions module."""
__all__ = []  # będzie w Etapie 4
```

### Faza 4: Zaktualizuj `sequence_training_pipeline.py`

#### Usuń:
- `_validate_schema()`
- `load_all_years()`
- `SequenceFilterConfig`
- Inne funkcje które będą w kolejnych etapach

#### Dodaj Importy:
```python
from ml.src.data_loading import load_all_years, validate_schema
from ml.src.sequences.config import SequenceFilterConfig
from ml.src.pipelines.config import PipelineConfig
from ml.src.pipelines.split import split_sequences
```

---

## ✅ Checklist Etapu 1

### Katalogi & Pliki
- [ ] `mkdir -p ml/src/data_loading`
- [ ] `mkdir -p ml/outputs/{models,metrics,analysis,logs}`
- [ ] `touch ml/src/features/__init__.py`
- [ ] `touch ml/src/targets/__init__.py`
- [ ] `touch ml/src/sequences/__init__.py`
- [ ] `touch ml/src/utils/__init__.py`
- [ ] `touch ml/src/data_loading/__init__.py`

### Przeniesienie Kodu
- [ ] Przenieś `_validate_schema()` → `ml/src/data_loading/validators.py`
- [ ] Przenieś `load_all_years()` → `ml/src/data_loading/loaders.py`
- [ ] Przenieś `SequenceFilterConfig` → `ml/src/sequences/config.py`
- [ ] Stwórz `ml/src/pipelines/config.py` (PipelineConfig)
- [ ] Przenieś `split_sequences()` → `ml/src/pipelines/split.py`

### Aktualizacja __init__.py
- [ ] `ml/src/data_loading/__init__.py` - dodaj importy
- [ ] `ml/src/features/__init__.py` - stwórz pusty
- [ ] `ml/src/targets/__init__.py` - stwórz pusty
- [ ] `ml/src/sequences/__init__.py` - dodaj SequenceFilterConfig
- [ ] `ml/src/utils/__init__.py` - stwórz pusty

### Refaktor `sequence_training_pipeline.py`
- [ ] Usuń `_validate_schema()`
- [ ] Usuń `load_all_years()`
- [ ] Usuń `SequenceFilterConfig`
- [ ] Dodaj importy z nowych modułów
- [ ] Sprawdzić że plik się kompiluje

### Testy Importów
- [ ] `from ml.src.data_loading import load_all_years` - działa
- [ ] `from ml.src.sequences.config import SequenceFilterConfig` - działa
- [ ] `from ml.src.pipelines.config import PipelineConfig` - działa
- [ ] `from ml.src.pipelines.split import split_sequences` - działa
- [ ] `python ml/src/pipelines/sequence_training_pipeline.py` - bez błędów

---

## 🎯 Rezultat Etapu 1

```
Przed:
ml/src/pipelines/sequence_training_pipeline.py (1740 linii wszystko)

Po:
ml/src/
├── pipelines/sequence_training_pipeline.py (~900 linii - importuje moduły)
├── data_loading/
│   ├── __init__.py
│   ├── loaders.py (load_all_years)
│   └── validators.py (_validate_schema)
├── sequences/
│   ├── __init__.py
│   └── config.py (SequenceFilterConfig)
├── features/
│   └── __init__.py
├── targets/
│   └── __init__.py
└── utils/
    └── __init__.py

ml/pipelines/
├── config.py (PipelineConfig)
└── split.py (split_sequences)

ml/outputs/
├── models/
├── metrics/
├── analysis/
└── logs/
```

**Zysk**:
- ✅ Kod jest teraz modułowy
- ✅ Łatwo importować funkcje
- ✅ Przygotowanie do dalszych etapów
- ✅ Separacja wkładu/wyniku (`outputs/` dla wyników)

---

## 📋 Następny Krok

Po ukończeniu Etapu 1:
1. Sprawdzić czy wszystkie importy działają
2. Commitujesz: `feat: Etap 1 - przeniesienie podstawowych modułów`
3. Przechodzisz do Etapu 2 (Inżynieria Cech)

---

**Status**: ⏳ Gotowy do Wdrożenia
**Kolejność**: Faza 1 → Faza 2 → Faza 3 → Faza 4
**Czas Szacunkowy**: 1-2 godziny
