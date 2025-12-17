# ETAP 1: Przeniesienie Podstawowych Modułów

## Cel
Przenieść podstawowe funkcje z `sequence_training_pipeline.py` do **już istniejących katalogów** (`ml/src/features/`, `ml/src/targets/`, `ml/src/sequences/`, `ml/src/utils/`). Brak tworzenia katalogów - katalogi już istnieją!

---

## Lista Rzeczy Do Zrobienia (Etap 1)

### ✅ Katalogi - JUŻ ISTNIEJĄ!
```
ml/src/
├── features/        ✅ PUSTY - czeka na kod
├── targets/         ✅ PUSTY - czeka na kod
├── sequences/       ✅ PUSTY - czeka na kod
└── utils/           ✅ PUSTY - czeka na kod
```

**NIE TWORZYĆ KATALOGÓW** - są już gotowe!

### 1. Pliki `__init__.py` do Stworzenia ✅
- [ ] `ml/src/features/__init__.py` - Jeśli go nie ma
- [ ] `ml/src/targets/__init__.py` - Jeśli go nie ma
- [ ] `ml/src/sequences/__init__.py` - Jeśli go nie ma
- [ ] `ml/src/utils/__init__.py` - Jeśli go nie ma

### 2. Dodatkowe Katalogi & Pliki (Opcjonalnie)
- [ ] `ml/outputs/` - Wyniki (modele, metryki, logi)
- [ ] `ml/outputs/models/` - Wytrenowane modele
- [ ] `ml/outputs/metrics/` - Metryki ewaluacji
- [ ] `ml/outputs/analysis/` - Analiza features
- [ ] `ml/outputs/logs/` - Logi z uruchomień

### 3. Przeniesienie Funkcji - GŁÓWNE ZADANIE ETAPU 1

#### Grupa 1: Ładowanie Danych (→ `ml/src/data_loading/` - NOWY KATALOG)
- [ ] Stwórz `ml/src/data_loading/` (jeśli nie istnieje)
- [ ] Stwórz `ml/src/data_loading/__init__.py`
- [ ] Stwórz `ml/src/data_loading/validators.py` - przenieś `_validate_schema()`
- [ ] Stwórz `ml/src/data_loading/loaders.py` - przenieś `load_all_years()`

#### Grupa 2: Konfiguracja (→ `ml/src/pipelines/` - Już istnieje)
- [ ] Stwórz `ml/src/pipelines/config.py` - centralna konfiguracja
- [ ] Stwórz `ml/src/pipelines/sequences/config.py` - przenieś `SequenceFilterConfig`
- [ ] Stwórz `ml/src/pipelines/split.py` - przenieś `split_sequences()`

#### Grupa 3: Features (→ `ml/src/features/` - Już istnieje)
- [ ] Stwórz `ml/src/features/__init__.py`
- [ ] Stwórz `ml/src/features/engineer.py` - `engineer_candle_features()`
- [ ] (Detale w Etapie 2)

#### Grupa 4: Targets (→ `ml/src/targets/` - Już istnieje)
- [ ] Stwórz `ml/src/targets/__init__.py`
- [ ] Stwórz `ml/src/targets/target_maker.py` - `make_target()`

#### Grupa 5: Sequences (→ `ml/src/sequences/` - Już istnieje)
- [ ] Stwórz `ml/src/sequences/__init__.py`
- [ ] Stwórz `ml/src/sequences/sequencer.py` - `create_sequences()`
- [ ] Stwórz `ml/src/sequences/filters.py` - `filter_by_session()`

#### Grupa 6: Utils (→ `ml/src/utils/` - Już istnieje)
- [ ] Stwórz `ml/src/utils/__init__.py`
- [ ] Stwórz `ml/src/utils/helpers.py` - funkcje pomocnicze

### 4. Refaktor Głównego Pliku ✅
- [ ] `ml/src/pipelines/sequence_training_pipeline.py` - usunąć przeniesione funkcje
  - Usunąć `_validate_schema()` (przeniesione do `data_loading/validators.py`)
  - Usunąć `load_all_years()` (przeniesione do `data_loading/loaders.py`)
  - Usunąć `SequenceFilterConfig` (przeniesione do `sequences/config.py`)
  - Dodać importy z nowych modułów
  - `run_pipeline()` - zostaje i importuje wszystkie moduły

---

## Szczegółowa Implementacja

### 1️⃣ Katalogi - JUŻ ISTNIEJĄ!

```bash
# Te katalogi JUŻ istnieją - nie trzeba ich tworzyć!
ml/src/
├── features/        ✅
├── targets/         ✅
├── sequences/       ✅
└── utils/           ✅

# Nowy katalog do stworzenia:
mkdir -p ml/src/data_loading
mkdir -p ml/outputs/models
mkdir -p ml/outputs/metrics
mkdir -p ml/outputs/analysis
mkdir -p ml/outputs/logs
```

---

### 2️⃣ Pliki `__init__.py`

Stwórz lub zaktualizuj pliki `__init__.py` w istniejących katalogach:

**`ml/src/features/__init__.py`**
```python
"""Feature engineering module for sequence training pipeline."""

from .engineer import engineer_candle_features

__all__ = ["engineer_candle_features"]
```

**`ml/src/targets/__init__.py`**
```python
"""Target creation module for sequence training pipeline."""

from .target_maker import make_target

__all__ = ["make_target"]
```

**`ml/src/sequences/__init__.py`**
```python
"""Sequence creation and filtering module for sequence training pipeline."""

from .config import SequenceFilterConfig
from .sequencer import create_sequences
from .filters import filter_by_session

__all__ = ["SequenceFilterConfig", "create_sequences", "filter_by_session"]
```

**`ml/src/utils/__init__.py`**
```python
"""Utility functions for sequence training pipeline."""

__all__ = []
```

**`ml/src/data_loading/__init__.py`** (NOWY KATALOG)
```python
"""Data loading and validation module for sequence training pipeline."""

from .loaders import load_all_years
from .validators import validate_schema

__all__ = ["load_all_years", "validate_schema"]
```

**`ml/src/pipelines/features/__init__.py`**
```python
"""Feature engineering module for sequence training pipeline."""

from .engineer import engineer_candle_features

__all__ = ["engineer_candle_features"]
```

**`ml/src/pipelines/targets/__init__.py`**
```python
"""Target creation module for sequence training pipeline."""

from .target_maker import make_target

__all__ = ["make_target"]
```

**`ml/src/pipelines/sequences/__init__.py`**
```python
"""Sequence creation and filtering module for sequence training pipeline."""

from .config import SequenceFilterConfig
from .sequencer import create_sequences
from .filters import filter_by_session

__all__ = ["SequenceFilterConfig", "create_sequences", "filter_by_session"]
```

**`ml/src/pipelines/training/__init__.py`**
```python
"""Training and evaluation module for sequence training pipeline."""

from .xgb_trainer import train_xgb
from .evaluation import evaluate, pick_best_threshold
from .artifacts import save_artifacts
from .feature_analysis import analyze_feature_importance

__all__ = ["train_xgb", "evaluate", "pick_best_threshold", "save_artifacts", "analyze_feature_importance"]
```

**`ml/src/pipelines/utils/__init__.py`**
```python
"""Utility functions for sequence training pipeline."""

__all__ = []
```

---

### 3️⃣ Plik Konfiguracji `ml/src/pipelines/config.py`

```python
"""Central pipeline configuration."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for sequence training pipeline.
    
    Attributes:
        data_dir: Directory with input OHLCV CSV files
        models_dir: Directory to save trained models and artifacts
        outputs_dir: Root directory for all outputs (models, metrics, logs, analysis)
        random_state: Random seed for reproducibility
        verbose: Enable verbose logging
    """
    
    # Directories - relative to project root
    data_dir: Path = Path(__file__).parent.parent / "data"
    models_dir: Path = Path(__file__).parent.parent / "models"  # [DEPRECATED] Use outputs_dir/models
    outputs_dir: Path = Path(__file__).parent.parent.parent / "outputs"
    
    # Subdirectories in outputs
    @property
    def outputs_models_dir(self) -> Path:
        """Directory for trained models."""
        return self.outputs_dir / "models"
    
    @property
    def outputs_metrics_dir(self) -> Path:
        """Directory for metrics."""
        return self.outputs_dir / "metrics"
    
    @property
    def outputs_analysis_dir(self) -> Path:
        """Directory for analysis (feature importance, etc.)."""
        return self.outputs_dir / "analysis"
    
    @property
    def outputs_logs_dir(self) -> Path:
        """Directory for logs."""
        return self.outputs_dir / "logs"
    
    # Pipeline defaults
    random_state: int = 42
    verbose: bool = True
    
    def create_directories(self) -> None:
        """Create all output directories if they don't exist."""
        self.outputs_models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_logs_dir.mkdir(parents=True, exist_ok=True)


# Default instance
DEFAULT_CONFIG = PipelineConfig()
```

---

### 4️⃣ Plik `ml/src/pipelines/sequences/config.py`

Przenieść `SequenceFilterConfig` z głównego pliku:

```python
"""Configuration classes for sequence creation."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SequenceFilterConfig:
    """Configuration for session-level sequence filters.

    Attributes:
        enable_m5_alignment: If True, keep only windows that close one minute before
            a new M5 candle to synchronize decisions with bar openings.
        enable_trend_filter: If True, require price to be above a long-term trend
            proxy and volatility regime to exceed a minimum threshold.
        trend_min_dist_sma200: Minimum normalized distance above SMA200 accepted when
            trend filter is enabled.
        trend_min_adx: Minimum ADX value required when trend filter is enabled.
        enable_pullback_filter: If True, constrain RSI-based pullback conditions.
        pullback_max_rsi_m5: Maximum allowed RSI_M5 reading when pullback filter is
            enabled.
    """

    enable_m5_alignment: bool = True
    enable_trend_filter: bool = True
    trend_min_dist_sma200: Optional[float] = 0.0
    trend_min_adx: Optional[float] = 15.0
    enable_pullback_filter: bool = True
    pullback_max_rsi_m5: Optional[float] = 75.0
```

---

### 5️⃣ Plik `ml/src/pipelines/split.py`

Stworzyć nowy plik dla `split_sequences()`:

```python
"""Chronological train/val/test split utilities."""

from typing import Tuple
import numpy as np
import pandas as pd


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    """Split data chronologically into train/val/test sets.
    
    Uses 2023 for training, 2024 Q1-Q2 for validation, Q3-Q4 for testing.
    Maintains temporal order to prevent data leakage.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        timestamps: Datetime index for each sample
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test)
    """
    # Split by year/quarter - detailed implementation will be in Etap 3
    pass
```

---

### 6️⃣ Plik `ml/src/pipelines/data_loading/validators.py`

Przenieść `_validate_schema()` z głównego pliku:

```python
"""Data schema validation utilities."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate OHLCV schema and basic price constraints.

    Args:
        df: DataFrame with OHLCV columns

    Raises:
        ValueError: On missing columns, non-positive prices, or High<Low inconsistencies
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric dtypes (coerce and drop bad rows)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs after coercion
    before = len(df)
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with invalid numeric values")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC contains non-positive values")
    if (df["Volume"] < 0).any():
        raise ValueError("Volume contains negative values")
    if (df["High"] < df["Low"]).any():
        raise ValueError("Price inconsistency: High < Low detected")
    if df.index.has_duplicates:
        logger.warning("Duplicate timestamps detected; dropping duplicates")
        df.drop_duplicates(inplace=True)
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
```

---

### 7️⃣ Plik `ml/src/pipelines/data_loading/loaders.py`

Przenieść `load_all_years()` z głównego pliku:

```python
"""Data loading utilities."""

from pathlib import Path
from typing import List
import logging
import numpy as np
import pandas as pd
from .validators import validate_schema

logger = logging.getLogger(__name__)


def load_all_years(data_dir: Path, year_filter: List[int] = None) -> pd.DataFrame:
    """Load and validate all available yearly CSVs.

    Args:
        data_dir: Directory containing XAU_1m_data_*.csv files
        year_filter: Optional list of years to load (e.g., [2023, 2024])

    Returns:
        Concatenated DataFrame indexed by datetime, strictly increasing

    Raises:
        FileNotFoundError: If no data files found
        ValueError: On schema validation failures
    """
    # Implementation from original file - will be moved in this phase
    pass
```

---

## Kontrola Jakości (Etap 1)

### ✅ Testy do Wykonania
- [ ] Import `from ml.src.pipelines.data_loading import load_all_years`
- [ ] Import `from ml.src.pipelines.sequences.config import SequenceFilterConfig`
- [ ] Import `from ml.src.pipelines.config import PipelineConfig`
- [ ] Sprawdzić, że `PipelineConfig().create_directories()` tworzy katalogi
- [ ] Sprawdzić brak błędów w pliku głównym po przenosinach

### ✅ Dokumentacja
- [ ] Zaktualizować docstrings w przenoszonych funkcjach
- [ ] Dodać `__all__` do każdego `__init__.py`
- [ ] Dodać komentarze wyjaśniające strukturę katalogów

---

## Metryki Sukcesu (Etap 1)

✅ **Kompletacja**:
- Wszystkie katalogi stworzone
- Wszystkie pliki `__init__.py` obecne
- Wszystkie importy działają bez błędów

✅ **Kwalifikacja**:
- Kod jest identyczny do oryginalnego (nie ma zmian logiki)
- Żadne testy nie powinny się zepsuć

✅ **Struktura**:
- `ml/src/pipelines/` ma strukturę modularną
- `ml/outputs/` jest gotowy na artefakty
- Brak danych w katalogach `scripts/` czy `outputs/` (separacja wkładu/wyniku)

---

## Jak to Będzie Wyglądać Po Etapie 1

```
ml/src/
├── pipelines/
│   ├── __init__.py
│   ├── sequence_training_pipeline.py    # Główny plik (refaktor - mniej linii)
│   ├── training_pipeline.py
│   ├── config.py                        # ✨ NOWY
│   ├── split.py                         # ✨ NOWY
│   ├── sequences/                       # ✨ NOWY subdirs
│   │   ├── __init__.py
│   │   └── config.py
│   ├── INDEX.md                         (dokumentacja)
│   ├── REFACTOR_PLAN.md                 (dokumentacja)
│   └── ... (inne dokumenty)
│
├── data_loading/                        # ✨ NOWY KATALOG
│   ├── __init__.py
│   ├── loaders.py
│   └── validators.py
│
├── features/                            # ✅ PUSTY → będzie kod w Etapie 2
│   ├── __init__.py
│   └── engineer.py                      # (będzie w Etapie 2)
│
├── targets/                             # ✅ PUSTY → będzie kod w Etapie 3
│   ├── __init__.py
│   └── target_maker.py                  # (będzie w Etapie 3)
│
├── sequences/                           # ✅ PUSTY → będzie kod w Etapie 3
│   ├── __init__.py
│   ├── config.py                        # ✨ Przenieść z pipelines/sequences/
│   ├── sequencer.py                     # (będzie w Etapie 3)
│   └── filters.py                       # (będzie w Etapie 3)
│
├── utils/                               # ✅ PUSTY → będzie kod w Etapie 4
│   ├── __init__.py
│   └── helpers.py                       # (będzie w Etapie 4)
│
├── data/                                (OHLCV dane - bez zmian)
├── config/                              (konfiguracja - bez zmian)
└── ... (inne istniejące katalogi)

ml/
├── outputs/                             # ✨ NOWY KATALOG
│   ├── models/                          (wytrenowane modele)
│   ├── metrics/                         (metryki)
│   ├── analysis/                        (analiza)
│   └── logs/                            (logi)
└── ... (inne katalogi)
```

---

## Uwagi Ważne

⚠️ **Przygotowanie do Kolejnych Etapów**:
- Katalogi w Etapie 1 mogą być puste (będą wypełniane w Etapach 2-5)
- Struktura katalogów musi być gotowa PRZED migracją kodu
- Nie pracujemy na kodu logiki w Etapie 1 - tylko na pliku `sequence_training_pipeline.py` dodajemy importy

⚠️ **Brak Zmian Logiki**:
- Kod przenoszony w Etapie 1 będzie **identyczny** do oryginalnego
- Żadne `if`/`else`, żadne refaktoryzacje logiki
- Celowa jest transparentna migracja

---

**Status**: ⏳ Czeka na Implementation
**Poprzedni**: REFACTOR_PLAN.md (Przegląd)
**Następny**: REFACTOR_ETAP_2.md (Inżynieria Cech)
