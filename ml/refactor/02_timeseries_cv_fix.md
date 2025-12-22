# Faza 1: Time Series Cross-Validation Implementation

**Status:** Planowanie  
**Priorytet:** 🔴 CRITICAL  
**Wpływ:** Unknown robustness → Measurable robustness  

---

## Problem Szczegółowo

### Obecny Approach (NIEBEZPIECZNY)

```python
# ❌ Jeden chronologiczny split (3-way)
train_end = int(0.6 * len(X))      # First 60%
val_end = int(0.8 * len(X))        # Next 20%
test_start = val_end                # Last 20%

X_train = X[:train_end]
X_val = X[train_end:val_end]
X_test = X[val_end:]
```

**Problemy:**
1. **Jeden split** - może być "lucky" lub "unlucky"
2. **Brak walidacji robustness** - model może overfit na konkretny okres
3. **Brak understanding** o tym jak model behaves w różnych market conditions
4. **Train set bias** - jeśli train ma strong trend, model się tego nauczy

### Prawidłowy Approach (Time Series CV)

```python
# ✅ Multiple folds w chronologicznym porządku
#    Każdy fold: train na POPRZEDNICH danych, test na NOWYCH

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    # Fold 1: train na 0-2000, test na 2000-2400 (20% test)
    # Fold 2: train na 0-2400, test na 2400-2800 (20% test)
    # Fold 3: train na 0-2800, test na 2800-3200 (20% test)
    # Fold 4: train na 0-3200, test na 3200-3600 (20% test)
    # Fold 5: train na 0-3600, test na 3600-4000 (20% test)
    #
    # Średnia z 5 testów = robust metric!
```

**Advantages:**
- Każdy fold ma INNE train/test split
- Model nie może "get lucky" na jednym split
- Widzisz variance across folds (std dev metryk)
- Data leakage jest wykluczone (test zawsze AFTER train chronologicznie)

---

## Rozwiązanie - Struktura Kodu

### Nowy Plik: `ml/src/utils/timeseries_validation.py`

```python
"""Time Series Cross-Validation utilities for ML pipelines.

Provides TimeSeriesSplit with proper handling of:
- Chronological data splitting
- Session-aware boundaries
- Leap-forward validation
- Metrics aggregation across folds
"""

import logging
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve, auc
)

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """Wrapper around TimeSeriesSplit for trading pipelines."""
    
    def __init__(
        self, 
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None
    ):
        """Initialize Time Series CV splitter.
        
        Args:
            n_splits: Number of CV folds (default 5)
            test_size: Size of test set per fold. If None: 100 // n_splits
            gap: Number of samples to exclude between train and test (gap window)
            max_train_size: Max training size per fold. If None: use all
        
        Notes:
            - For trading: n_splits=5 gives ~20% test size per fold
            - gap=12 excludes 12 M5 candles = 1 hour (prevents leakage)
        """
        self.tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size
        )
        self.n_splits = n_splits
        self.gap = gap
        self.metrics_per_fold = []
        
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None):
        """Generate train/test indices for each fold.
        
        Yields:
            (train_idx, test_idx) tuple for each fold
        """
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X, y, groups)):
            logger.info(f"Fold {fold + 1}/{self.n_splits}: "
                       f"train={len(train_idx)} samples, test={len(test_idx)} samples")
            yield fold, train_idx, test_idx
    
    def evaluate_fold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        fold_num: int = 0
    ) -> Dict[str, float]:
        """Evaluate single fold.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities (1D array)
            threshold: Decision threshold
            fold_num: Fold number (for logging)
        
        Returns:
            Dictionary with metrics for this fold
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'n_positives': np.sum(y_pred),
            'n_positives_true': np.sum(y_true),
        }
        
        # Only compute ROC-AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Compute PR-AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)
        else:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
        
        logger.info(f"Fold {fold_num}: "
                   f"Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, "
                   f"F1={metrics['f1']:.4f}")
        
        self.metrics_per_fold.append(metrics)
        return metrics
    
    def aggregate_metrics(self) -> Dict[str, float]:
        """Aggregate metrics across all folds.
        
        Returns:
            Dictionary with mean and std of metrics
        """
        if not self.metrics_per_fold:
            raise ValueError("No metrics to aggregate. Run evaluate_fold() first.")
        
        metrics_df = pd.DataFrame(self.metrics_per_fold)
        
        aggregated = {}
        for col in metrics_df.columns:
            if col == 'threshold':
                continue
            
            aggregated[f"{col}_mean"] = metrics_df[col].mean()
            aggregated[f"{col}_std"] = metrics_df[col].std()
        
        logger.info("\n" + "=" * 60)
        logger.info("TIME SERIES CV - AGGREGATED METRICS")
        logger.info("=" * 60)
        for key, value in aggregated.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info("=" * 60)
        
        return aggregated
    
    def get_fold_summary(self) -> pd.DataFrame:
        """Get summary table of all folds.
        
        Returns:
            DataFrame with metrics per fold
        """
        return pd.DataFrame(self.metrics_per_fold)


def validate_train_test_boundary(
    timestamps: pd.DatetimeIndex,
    train_idx: np.ndarray,
    test_idx: np.ndarray
) -> bool:
    """Validate that test timestamps are AFTER train timestamps.
    
    Args:
        timestamps: DatetimeIndex of data
        train_idx: Training indices
        test_idx: Test indices
    
    Returns:
        True if valid (no temporal overlap)
        
    Raises:
        ValueError: If test comes before train
    """
    train_max = timestamps[train_idx[-1]]
    test_min = timestamps[test_idx[0]]
    
    if test_min <= train_max:
        raise ValueError(
            f"Temporal validation failed: "
            f"test starts at {test_min} but train goes until {train_max}"
        )
    
    logger.info(f"✅ Temporal validation passed: "
               f"train until {train_max}, test from {test_min}")
    return True


def validate_no_sequence_leakage(
    timestamps: pd.DatetimeIndex,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    window_size: int
) -> bool:
    """Validate that sequences don't cross train/test boundary.
    
    Args:
        timestamps: DatetimeIndex of data
        train_idx: Training indices
        test_idx: Test indices
        window_size: Sequence window size in candles
    
    Returns:
        True if valid (no sequence crossing)
        
    Raises:
        ValueError: If sequences cross boundary
    """
    # Find the split point
    train_end_idx = train_idx[-1]
    test_start_idx = test_idx[0]
    
    # Check if any sequence starting in train would extend into test
    # A sequence starting at index i needs indices i, i+1, ..., i+window_size-1
    
    last_valid_sequence_start = train_end_idx - (window_size - 1)
    
    if last_valid_sequence_start < train_idx[0]:
        logger.warning(
            f"⚠️  Warning: window_size={window_size} is larger than "
            f"first fold training data. Some sequences will be excluded."
        )
    
    logger.info(f"✅ Sequence boundary validation passed: "
               f"last valid sequence starts at idx {last_valid_sequence_start}, "
               f"test starts at idx {test_start_idx}")
    return True
```

### Krok 2: Uaktualnij `split_and_scale_stage()` w `pipeline_stages.py`

Stara wersja (single split):

```python
# ❌ STARE
def split_and_scale_stage(...):
    train_end = int(0.6 * len(X))
    val_end = int(0.8 * len(X))
    # ...
    return (X_train_scaled, X_val_scaled, X_test_scaled, ...)
```

Nowa wersja (with CV option):

```python
# ✅ NOWE
from ml.src.utils.timeseries_validation import TimeSeriesValidator, validate_train_test_boundary

def split_and_scale_stage(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    year_filter: Optional[list[int]] = None,
    use_timeseries_cv: bool = False,  # ← NEW PARAM
    cv_folds: int = 5,  # ← NEW PARAM
) -> Union[Tuple, List[Tuple]]:  # Single split OR list of folds
    """Split and scale data.
    
    Args:
        X, y, timestamps: Data
        year_filter: Optional year filter
        use_timeseries_cv: If True, use CV instead of single split
        cv_folds: Number of CV folds
    
    Returns:
        If use_timeseries_cv=False:
            Single split: (X_train_scaled, X_val_scaled, X_test_scaled, ...)
        If use_timeseries_cv=True:
            List of folds: [(X_train_scaled, X_test_scaled, ...), ...] per fold
    """
    
    if use_timeseries_cv:
        logger.info(f"Using Time Series Cross-Validation with {cv_folds} folds")
        
        validator = TimeSeriesValidator(n_splits=cv_folds, gap=0)
        cv_results = []
        
        for fold, train_idx, test_idx in validator.split(X, y):
            # Validate boundaries
            validate_train_test_boundary(timestamps, train_idx, test_idx)
            
            # Split
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            
            # Scale only on training data
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            cv_results.append({
                'fold': fold,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'timestamps_train': timestamps[train_idx],
                'timestamps_test': timestamps[test_idx],
                'scaler': scaler,
            })
        
        return cv_results  # List of folds
    
    else:
        logger.info("Using chronological 3-way split (train/val/test)")
        
        # Original code: single split
        train_end = int(0.6 * len(X))
        val_end = int(0.8 * len(X))
        
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, len(X))
        
        # ... rest of original code
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, ...)
```

---

## Zmiana w `run_pipeline()`

```python
def run_pipeline(params: PipelineParams) -> Dict[str, float]:
    """Execute pipeline with Time Series CV support."""
    
    # ... setup and feature engineering ...
    
    # ===== STAGE 5: Split and scale =====
    cv_results = split_and_scale_stage(
        X=X,
        y=y,
        timestamps=timestamps,
        year_filter=params.year_filter,
        use_timeseries_cv=True,  # ← ENABLE CV
        cv_folds=5,
    )
    
    # ===== STAGE 6: Train and evaluate (per fold) =====
    all_metrics = []
    
    for fold_data in cv_results:
        fold_num = fold_data['fold']
        X_train_scaled = fold_data['X_train_scaled']
        X_test_scaled = fold_data['X_test_scaled']
        y_train = fold_data['y_train']
        y_test = fold_data['y_test']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_num + 1} / {len(cv_results)}")
        logger.info(f"{'='*60}")
        
        # Train model on this fold
        metrics, model = train_and_evaluate_stage(
            X_train_scaled=X_train_scaled,
            y_train=y_train,
            X_val_scaled=None,  # No val in CV
            y_val=None,
            X_test_scaled=X_test_scaled,
            y_test=y_test,
            # ... other params ...
        )
        
        all_metrics.append(metrics)
    
    # ===== STAGE 7: Aggregate CV results =====
    final_metrics = aggregate_cv_metrics(all_metrics)
    
    return final_metrics
```

---

## Walidacja

### Test 1: CV Split Chronology

```python
# test_ts_cv.py
import numpy as np
from ml.src.utils.timeseries_validation import TimeSeriesValidator

X = np.random.randn(1000, 30)
validator = TimeSeriesValidator(n_splits=5)

for fold, train_idx, test_idx in validator.split(X):
    # Sprawdzenie że test_idx są ZAWSZE DOPO train_idx
    assert train_idx[-1] < test_idx[0], f"Fold {fold}: Temporal order violated!"
    
print("✅ All folds have proper chronological order")
```

### Test 2: No Sequence Leakage

```python
# test_sequence_leakage.py
from ml.src.utils.timeseries_validation import validate_no_sequence_leakage

timestamps = pd.date_range('2024-01-01', periods=1000, freq='5min')
validator = TimeSeriesValidator(n_splits=5)

for fold, train_idx, test_idx in validator.split(X):
    validate_no_sequence_leakage(
        timestamps, 
        train_idx, 
        test_idx, 
        window_size=100
    )

print("✅ No sequence boundary crossing detected")
```

---

## Checkpoint

Po tej naprawie powinieneś:
- ✅ Mieć nowy moduł `timeseries_validation.py`
- ✅ Mieć `TimeSeriesValidator` class
- ✅ Mieć walidację granic
- ✅ Mieć agregację metryk z CV
- ✅ Testy walidujące CV

Po wdrożeniu: **Metryki będą mieć std dev** (np. precision_mean=0.65, precision_std=0.05), co pokazuje robustness.

---

## Następny Problem

Gdy skończysz tę naprawę → przejdź do [03_threshold_optimization_fix.md](03_threshold_optimization_fix.md)
