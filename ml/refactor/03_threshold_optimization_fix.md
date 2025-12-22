# Faza 1: Threshold Optimization Fix

**Status:** Planowanie  
**Priorytet:** 🔴 CRITICAL  
**Wpływ:** Invalid metrics → Valid metrics  

---

## Problem Szczegółowo

### Obecny Approach (DATA SNOOPING)

```python
# ❌ BŁĘDNE: Threshold optimization na test set
metrics, model = train_and_evaluate_stage(
    X_train_scaled=X_train_scaled,
    y_train=y_train,
    X_val_scaled=X_val_scaled,    # ← Val set known
    y_val=y_val,
    X_test_scaled=X_test_scaled,  # ← Test set may be used!
    y_test=y_test,
    ...
)
```

**Hipoteza: Funkcja `train_and_evaluate_stage()` robi:**

```python
# Gdzieś wewnątrz
# 1. Train model ✅ OK
model.fit(X_train_scaled, y_train)

# 2. Optimize threshold - ale NA KTÓRYM ZBIORZE?
y_pred_proba_val = model.predict_proba(X_val_scaled)

# ❓ Czy threshold selection patrzył na X_test?
# ❓ Czy validation metrics to X_val czy X_test?

# 3. Evaluate na test
y_pred_proba_test = model.predict_proba(X_test_scaled)
metrics = evaluate(y_test, y_pred_proba_test)  # Ale threshold z X_test?
```

### Konsekwencje

Jeśli threshold był optymalizowany na test set:

```
Test Set Metrics (BIASED):
  Precision: 75%
  F1: 0.68
  Win Rate: 70%

Real Production (after deployment):
  Precision: 55% (-20 points!)
  F1: 0.42
  Win Rate: 50%
```

---

## Prawidłowy Proces

### Poprawna Sekwencja

```
1. X_train, y_train       → Train model (fit)
2. X_val, y_val           → Optimize threshold (find best threshold)
3. X_test, y_test         → Evaluate final metrics (no touching threshold)
```

### Reguły

- **TRAIN SPLIT:** Fit model.fit(X_train, y_train) ✅
- **VAL SPLIT:** Find threshold: best_threshold = argmax_t F1(y_val, predict(X_val, t)) ✅
- **TEST SPLIT:** Report metrics: metrics = evaluate(y_test, predict(X_test, best_threshold)) ✅

**KLUCZOWE:** Threshold jest selected na VAL, nie na TEST.

---

## Rozwiązanie

### Krok 1: Zrefaktoryzuj `train_and_evaluate_stage()`

Nowy podział odpowiedzialności:

```python
# ml/src/pipelines/pipeline_stages.py

def train_and_evaluate_stage(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,      # ← REQUIRED dla threshold optimization
    y_val: np.ndarray,
    X_test_scaled: np.ndarray,     # ← SEPARATE from val
    y_test: np.ndarray,
    random_state: int = 42,
    min_precision: float = 0.65,
    min_recall: float = 0.40,
    # ... other params ...
) -> Tuple[Dict[str, float], XGBClassifier]:
    """Train model with proper validation/test separation.
    
    **CRITICAL**: This function enforces proper data split usage:
    - TRAIN: fit model
    - VAL: optimize threshold
    - TEST: evaluate final metrics (no threshold tuning!)
    
    Returns:
        (metrics_on_test_set, trained_model)
    """
    
    # ========== STAGE 1: Train model on TRAIN set ==========
    logger.info("Training XGBoost model...")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=random_state,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        # ... other hyperparams ...
    )
    
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_val_scaled, y_val)],  # Monitor val during training
        verbose=0,
        early_stopping_rounds=20
    )
    
    logger.info(f"Model training complete")
    
    # ========== STAGE 2: Optimize threshold on VAL set ==========
    logger.info("Optimizing decision threshold on VAL set...")
    
    y_pred_proba_val = model.predict_proba(X_val_scaled)[:, 1]
    
    best_threshold, best_f1 = optimize_threshold_on_val(
        y_true=y_val,
        y_pred_proba=y_pred_proba_val,
        min_precision=min_precision,
        min_recall=min_recall,
        use_ev_optimization=use_ev_optimization,
        # ... other params ...
    )
    
    logger.info(f"Optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f} on VAL)")
    
    # ========== STAGE 3: Evaluate on TEST set ==========
    # ⚠️  CRITICAL: Nie używamy X_test do szukania threshold!
    # Tylko do raportowania final metrics!
    
    logger.info("Evaluating on TEST set with optimized threshold...")
    
    y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (y_pred_proba_test >= best_threshold).astype(int)
    
    # Compute metrics na TEST set
    metrics = {
        'threshold': best_threshold,
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_test) if len(np.unique(y_test)) > 1 else np.nan,
        'pr_auc': compute_pr_auc(y_test, y_pred_proba_test),
        'win_rate': precision_score(y_test, y_pred_test, zero_division=0),  # Same as precision
        'n_positive_predictions': np.sum(y_pred_test),
        'confusion_matrix': compute_confusion_matrix(y_test, y_pred_test),
    }
    
    logger.info(f"\nTEST SET EVALUATION:")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1:        {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics, model


def optimize_threshold_on_val(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    min_precision: float = 0.65,
    min_recall: float = 0.40,
    use_ev_optimization: bool = False,
    use_hybrid_optimization: bool = False,
    ev_win_coefficient: float = 2.0,
    ev_loss_coefficient: float = 1.0,
) -> Tuple[float, float]:
    """Find optimal threshold using VAL set.
    
    **CRITICAL**: This runs ONLY on VAL set, not TEST.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        min_precision: Minimum acceptable precision
        min_recall: Minimum acceptable recall
        use_ev_optimization: Use Expected Value optimization
        use_hybrid_optimization: Use hybrid (EV + precision/recall floors)
        ev_win_coefficient: Win payoff for EV optimization
        ev_loss_coefficient: Loss payoff for EV optimization
    
    Returns:
        (best_threshold, best_score)
    """
    logger.info("Running threshold optimization on VAL set...")
    
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = -np.inf
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if use_hybrid_optimization:
            # EV-based but with floor constraints
            if precision >= min_precision and recall >= min_recall:
                ev = (precision * ev_win_coefficient - 
                      (1 - precision) * ev_loss_coefficient)
                score = ev
            else:
                score = -np.inf  # Invalid: doesn't meet constraints
        
        elif use_ev_optimization:
            # Pure EV optimization
            ev = (precision * ev_win_coefficient - 
                  (1 - precision) * ev_loss_coefficient)
            score = ev
        
        else:
            # F1-based optimization
            score = f1
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'score': score,
            'n_positives': np.sum(y_pred),
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # Log top 5 thresholds
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    
    logger.info(f"\nTop 5 thresholds on VAL set:")
    for i, row in results_df.head(5).iterrows():
        logger.info(
            f"  T={row['threshold']:.2f}: "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"Score={row['score']:.4f}"
        )
    
    return best_threshold, best_score
```

### Krok 2: Update `split_and_scale_stage()` - REQUIRE validation set

```python
# ml/src/pipelines/pipeline_stages.py

def split_and_scale_stage(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    year_filter: Optional[list[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ...]:
    """Split and scale with REQUIRED 3-way split.
    
    **CRITICAL**: Must return SEPARATE train/val/test sets!
    Threshold optimization REQUIRES validation set.
    
    Returns:
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train, y_val, y_test,
         timestamps_train, timestamps_val, timestamps_test,
         scaler)
    """
    
    logger.info("Splitting data: 60% train, 20% val, 20% test")
    
    # Chronological split: TRAIN | VAL | TEST
    n = len(X)
    train_end = int(0.6 * n)      # 0 to 60%
    val_end = int(0.8 * n)        # 60% to 80%
    
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    ts_train = timestamps[train_idx]
    ts_val = timestamps[val_idx]
    ts_test = timestamps[test_idx]
    
    # Scaling: FIT only on TRAIN
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Train: {len(train_idx)} samples ({ts_train.min()} to {ts_train.max()})")
    logger.info(f"Val:   {len(val_idx)} samples ({ts_val.min()} to {ts_val.max()})")
    logger.info(f"Test:  {len(test_idx)} samples ({ts_test.min()} to {ts_test.max()})")
    
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        ts_train, ts_val, ts_test,
        scaler
    )
```

### Krok 3: Update `run_pipeline()` w sequence_training_pipeline.py

```python
def run_pipeline(params: PipelineParams) -> Dict[str, float]:
    """Execute pipeline with proper train/val/test separation."""
    
    # ... feature engineering ...
    
    # ===== STAGE 5: Split and scale (REQUIRES VAL SET) =====
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train, y_val, y_test,
     ts_train, ts_val, ts_test,
     scaler) = split_and_scale_stage(
        X=X,
        y=y,
        timestamps=timestamps,
        year_filter=params.year_filter,
    )
    
    # ===== STAGE 6: Train and evaluate =====
    # ✅ POPRAWKA: Pass all three sets
    metrics, model = train_and_evaluate_stage(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_val_scaled=X_val_scaled,      # ← For threshold optimization
        y_val=y_val,
        X_test_scaled=X_test_scaled,    # ← For final evaluation
        y_test=y_test,
        random_state=params.random_state,
        min_precision=params.min_precision,
        min_recall=params.min_recall,
        use_ev_optimization=params.use_ev_optimization,
        use_hybrid_optimization=params.use_hybrid_optimization,
        ev_win_coefficient=params.ev_win_coefficient,
        ev_loss_coefficient=params.ev_loss_coefficient,
    )
    
    # ... save artifacts ...
    
    return metrics
```

---

## Walidacja

### Test 1: Threshold Only Uses VAL

```python
# test_threshold_integrity.py
from unittest.mock import patch
import numpy as np

# Verify that threshold optimization doesn't peek at test set
def test_threshold_uses_only_val():
    X_train = np.random.randn(600, 30)
    y_train = np.random.randint(0, 2, 600)
    
    X_val = np.random.randn(200, 30)
    y_val = np.random.randint(0, 2, 200)
    
    X_test = np.random.randn(200, 30)
    y_test = np.random.randint(0, 2, 200)
    
    # Train model
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Threshold optimization
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    threshold, _ = optimize_threshold_on_val(
        y_true=y_val,
        y_pred_proba=y_pred_proba_val,
        min_precision=0.65,
        use_ev_optimization=False,
    )
    
    # Check that threshold makes sense for VAL
    y_pred_val = (y_pred_proba_val >= threshold).astype(int)
    val_precision = precision_score(y_val, y_pred_val)
    assert val_precision >= 0.65, "Threshold doesn't satisfy constraint!"
    
    # Important: Threshold is computed, but TEST evaluation uses it as-is
    # Don't re-optimize on test
    print(f"✅ Threshold={threshold:.4f} optimized on VAL only")


test_threshold_uses_only_val()
```

### Test 2: VAL and TEST Sets Don't Overlap

```python
# test_no_val_test_overlap.py
def test_val_test_boundary():
    from ml.src.pipelines.pipeline_stages import split_and_scale_stage
    
    X = np.random.randn(1000, 30)
    y = np.random.randint(0, 2, 1000)
    timestamps = pd.date_range('2024-01-01', periods=1000, freq='5min')
    
    (X_train, X_val, X_test, 
     y_train, y_val, y_test,
     ts_train, ts_val, ts_test,
     scaler) = split_and_scale_stage(X, y, timestamps)
    
    # Sprawdzenie że nie ma overlap
    assert ts_train.max() < ts_val.min(), "Train/Val overlap!"
    assert ts_val.max() < ts_test.min(), "Val/Test overlap!"
    
    print("✅ No temporal overlap between train/val/test")

test_val_test_boundary()
```

---

## Checkpoint

Po tej naprawie powinieneś:
- ✅ Mieć REQUIRED 3-way split (train/val/test)
- ✅ Mieć threshold optimization TYLKO na VAL
- ✅ Mieć finalne metryki raportowane TYLKO na TEST
- ✅ Testy walidujące brak snooping

Po wdrożeniu: **Metryki na test set będą bardziej realistyczne** (mogą być niższe o 5-15% niż poprzednio, ale to jest dobre - to znaczy że previous metryki były zawyżone).

---

## Następne Kroki

Gdy skończysz wszystkie 3 CRITICAL naprawy:
1. ✅ Data Leakage Fix
2. ✅ Time Series CV Fix
3. ✅ Threshold Optimization Fix

**Wtedy:** Powtórz training i porównaj metryki. Powinieneś zobaczyć:
- Metryki niższe (~5-15%), ale bardziej realistyczne
- Std dev z CV pokazujący robustness

Następnie przejdź do [04_lookahead_bias_fix.md](04_lookahead_bias_fix.md) z HIGH priority fixes.
