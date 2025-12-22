# Model Training Optimization TODO

**Status**: In Progress | **Last Updated**: 2025-12-21

---

## 📋 Optimization Points (Priority Order)

### ✅ **Punkt 5: Expected Value Optimization** 
- **Status**: ✅ COMPLETED
- **Description**: Maksymalizacja zysku (EV) zamiast samej accuracy
- **Implementation**: Hybrid threshold selection z floor'ami na precision i recall
- **Expected Impact**: +0-2% na recall przy utrzymaniu precision
- **Files Modified**: 
  - `ml/src/utils/risk_config.py` (parametry)
  - `ml/src/training/sequence_evaluation.py` (_pick_best_threshold_hybrid)
  - `ml/src/pipeline_stages.py` (train_and_evaluate_stage)

---

### ❌ **Punkt 4: Class Imbalance Handling (SMOTE) - REJECTED**
- **Status**: ❌ TESTED & REJECTED
- **Result**: Random oversampling made things WORSE (WIN RATE 85.71% → 58.82%)
- **Why It Failed**:
  - SMOTE/random oversampling creates DATA LEAKAGE in time-series
  - Duplicated samples have identical features → no diversity
  - Model overtrained on synthetic data and lower generalization
  - Financial time-series require different class imbalance handling
- **Alternative Approaches** (for future):
  - Class weights in XGBoost (already using `scale_pos_weight`)
  - Threshold optimization (already implemented - HYBRID method)
  - Ensemble methods with class-weighted voting (Point 7)
  - Asymmetric loss functions (cost-sensitive learning - Point 1)
- **Key Lesson**: For financial time-series, data augmentation via duplication is harmful; focus on threshold optimization instead

---

### 🎲 **Punkt 2: Bayesian Hyperparameter Optimization**
- **Status**: ⏳ PLANNED
- **Description**: Automatyczne znalezienie najlepszych hiperparametrów
- **Tools**: `skopt.gp_minimize` + `cross_val_score`
- **Expected Impact**: +2-5% na F1 score
- **Effort**: Średni (50-100 iteracji)

---

### 🔧 **Punkt 1: Cost-Sensitive Learning w XGBoost**
- **Status**: 🚀 IN PROGRESS
- **Description**: Penalizuj False Positives (przegrane transakcje) bardziej niż False Negatives
- **Implementation**: 
  - `sample_weight` array: TP (w=3.0), TN (w=1.0) - TP są 3x "ważniejsze"
  - Integracja w XGBoost.fit() via `sample_weight` parameter
- **Expected Impact**: +1-3% precision (mniej fałszywych alarmów)
- **How It Works**:
  1. Dla każdego sampla: jeśli y==1 → waga=3.0, inaczej waga=1.0
  2. Model penalizuje błędy na wagach: błędy TP (waga 3.0) są 3x bardziej kosztowne
  3. Model uczy się bardziej konserwatywnie (wyższa precision, możliwy spadek recall)
- **Files Modified**: 6 plików - risk_config, pipeline_cli, pipeline_config_extended, sequence_xgb_trainer, pipeline_stages, sequence_training_pipeline

---

### 📊 **Punkt 3: Feature Selection & Importance Analysis**
- **Status**: ⏳ PLANNED
- **Description**: Usunięcie szumnych features, redukcja overfittingu
- **Tools**: `shap.TreeExplainer` lub `permutation_importance`
- **Expected Impact**: Szybsze inference, -0.5% recall (lepsze precision)
- **Effort**: Średni

---

### 🔄 **Punkt 6: Walk-Forward Cross-Validation**
- **Status**: ⏳ PLANNED
- **Description**: Time-series aware testing (brak data leakage)
- **Implementation**: `TimeSeriesSplit` z realistyczną sekwencją czasową
- **Expected Impact**: Bardziej realistyczne wyniki (-3-5%)
- **Effort**: Średni
- **Critical**: Taki CV jest wymagany dla finansowych time-series

---

### 📈 **Punkt 7: Ensemble Methods (Stacking/Voting)**
- **Status**: ⏳ PLANNED
- **Description**: Połączenie predykcji wielu modeli (XGBoost + RF + SVM)
- **Implementation**: `VotingClassifier` lub `StackingClassifier`
- **Expected Impact**: +1-2% recall, bardziej stabilne
- **Effort**: Wysoki

---

### 🎲 **Punkt 8: Threshold Tuning per ATR-Zone**
- **Status**: ⏳ PLANNED
- **Description**: Różny threshold w zależności od volatility (ATR)
- **Implementation**: Quintile-based threshold per ATR zone
- **Expected Impact**: +5-10% recall w high ATR zones, +1-2% overall
- **Effort**: Najwyższy (Advanced)
- **Note**: Ostatni punkt do implementacji

---

## 📊 Expected Results After All Optimizations

| Optimization | WIN RATE | Precision | Recall | Notes |
|--------------|----------|-----------|--------|-------|
| **Current (Baseline)** | 85.71% | 85.71% | 18.75% | F1-optimized hybrid |
| **+Punkt 4 (SMOTE)** | ~84% | ~82% | ~25-26% | Więcej trades |
| **+Punkt 2 (BayesOpt)** | ~85% | ~84% | ~26-28% | Better hyperparameters |
| **+Punkt 1 (Cost-Sensitive)** | ~85% | ~86-87% | ~26-27% | Penalize FP more |
| **+Punkt 6 (Walk-Forward)** | ~82-83% | ~83% | ~25% | Realistic |
| **+Punkt 3 (Feature Select)** | ~85% | ~86% | ~24% | Less noise |
| **+Punkt 7 (Ensemble)** | ~85% | ~84% | ~27% | More stable |
| **+Punkt 8 (ATR-Zone)** | ~86% | ~85% | ~28-30% | **BEST CASE** |

---

## 🎯 Implementation Workflow

```
Point 1 (Cost-Sensitive Learning) ← START HERE
    ↓ 
Point 2 (BayesOpt) 
    ↓
Point 6 (Walk-Forward CV) 
    ↓ (Reality check for time-series)
Point 3 (Feature Selection) 
    ↓
Point 7 (Ensemble) 
    ↓
Point 8 (ATR-Zone Tuning) 
    ↓
✅ PRODUCTION READY

Note: Point 4 (SMOTE) REJECTED - caused data leakage in time-series
```

---

## 📝 Notes

- **Risk**: Punkty 4, 6, 7 mogą zmniejszyć WIN RATE (ale będzie bardziej realistyczny)
- **Benefit**: Punkty 1, 2, 3, 8 powinny poprawić recall bez spadku precision
- **Critical**: Punkt 6 (Walk-Forward) jest OBOWIĄZKOWY dla time-series (walidacja realistyczności)
- **Time Estimate**: ~4-6 godzin na wszystkie punkty (zależy od testowania)

---

## Git Commit Messages (Template)

```
feat(ml): Point 4 - Add SMOTE for class imbalance handling

- Implement SMOTE oversampling for minority class (Class 1)
- Integrate StratifiedKFold for balanced cross-validation
- Expected: +3-8% recall improvement
- Files: ml/src/pipeline_stages.py, ml/src/training/sequence_evaluation.py

Closes #ml-optimization-point-4
```
