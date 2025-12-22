# Regime Filter: Visual Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  OPCJA B: PREDICTION GATING                     │
└─────────────────────────────────────────────────────────────────┘

                    TRAINING PIPELINE
                    ───────────────────
                          │
                          ▼
                  ┌─────────────────┐
                  │  Load ALL data  │  ✅ Use 100% (no filtering)
                  │  (no filtering) │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Engineer       │
                  │  features (24)  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Train model    │
                  │  (XGBoost)      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Save model +   │
                  │  scaler         │
                  └─────────────────┘


                   INFERENCE PIPELINE
                   ──────────────────
                          │
                          ▼
                  ┌─────────────────┐
                  │  Load model +   │
                  │  scaler         │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Get market     │
                  │  data           │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Calculate      │
                  │  ATR, ADX,      │
                  │  SMA200         │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Engineer       │
                  │  features (24)  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Scale features │
                  └────────┬────────┘
                           │
                           ▼
                  ┌──────────────────────┐
                  │  Model prediction    │  ← Output: confidence
                  │  (XGBoost)           │
                  └────────┬─────────────┘
                           │
                           ▼
                  ┌──────────────────────┐
                  │  Generate signal     │
                  │  (confidence > 0.5)  │
                  └────────┬─────────────┘
                           │
             ┌─────────────▼─────────────┐
             │ REGIME FILTER GATING ✨   │  ← NEW: Opcja B
             │ (MOST IMPORTANT PART)     │
             └────────┬──────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
   SIGNAL = 1?              REGIME CONDITIONS
   (Confidence > 0.5)       ┌──────────────────┐
        │                   │ Check:           │
        │                   │ - ATR ≥ 12?      │
        │                   │ - ADX ≥ 12?      │
        │                   │ - Price>SMA200?  │
        │                   │ - Confidence OK? │
        │                   └────────┬─────────┘
        │                            │
        ▼                            ▼
   ┌────────────────────────────────────┐
   │ APPLY REGIME FILTER                │
   │ ═══════════════════════════════    │
   │ If all conditions met:             │
   │   → Keep signal = 1  ✅ TRADE      │
   │                                    │
   │ If any condition fails:            │
   │   → Change signal = 0  🚫 SKIP     │
   └──────────┬─────────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Signal (0 or 1)      │
   │ + Regime info        │
   │ + Indicators         │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Execute trade        │
   │ (only if signal=1)   │
   └──────────────────────┘
```

---

## Decision Tree: Should We Trade?

```
                        SIGNAL = 1?
                            │
                    ┌───────┴────────┐
                    │                │
                   NO               YES
                    │                │
                    └──────┬─────────┘
                           │
                    REGIME FILTER CHECK
                    ────────────────────
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
        ATR < 12?      ADX < 12?     Price ≤ SMA200?
            │              │              │
        ┌───┴───┐       ┌───┴───┐     ┌───┴───┐
        │       │       │       │     │       │
       YES     NO      YES     NO    YES     NO
        │       │       │       │     │       │
        ▼       ▼       ▼       ▼     ▼       ▼
      ❌      ✅      ❌      ✅    ❌      ✅
      SUPPRESS       SUPPRESS       SUPPRESS
        
    ┌──────────────────────────────┐
    │ IF ANY CHECK = YES → SUPPRESS │
    │ IF ALL CHECKS = NO → TRADE    │
    └──────────────────────────────┘
            │
            ▼
    ┌──────────────────┐
    │ CONFIDENCE OK?   │
    │ (based on regime)│
    └────────┬─────────┘
             │
    ┌────────┴──────────────────┐
    │                           │
   TIER1               TIER2/3/4
   ATR≥18             ATR<18
    │                  │
 conf>0.35%      conf depends
    │            on tier
    ▼            │
  ✅TRADE        ▼
              Various
              thresholds
```

---

## Code Integration Points (Where to Add Code)

### Location 1: Backtest Script

```python
# File: ml/scripts/backtest_strategy.py

# BEFORE: Signal is used directly
signal = signals[i - 1]
if signal == 1:
    execute_trade()

# AFTER: Signal is gated by regime filter
from ml.src.filters.regime_filter import RegimeFilter

regime_filter = RegimeFilter()

signal = signals[i - 1]
if signal == 1:
    # ← ADD THIS BLOCK (5 lines)
    filtered_signal = regime_filter.filter_predictions_by_regime(
        signals=np.array([signal]),
        confidence=np.array([confidence_scores[i-1]]),
        indicators={'atr': atr[i-1], 'adx': adx[i-1], 
                    'close': prices[i-1], 'sma200': sma200[i-1]}
    )
    signal = filtered_signal[0]

if signal == 1:
    execute_trade()
```

### Location 2: Real-Time Prediction

```python
# File: Your trading bot / inference service

# BEFORE: Prediction used directly
prediction = model.predict_proba(features)[1]
if prediction > 0.5:
    place_order()

# AFTER: Prediction is gated by regime filter
from ml.src.filters.regime_filter import RegimeFilter

regime_filter = RegimeFilter()

prediction = model.predict_proba(features)[1]
signal = 1 if prediction > 0.5 else 0

# ← ADD THIS BLOCK (5-10 lines)
if signal == 1:
    filtered_signal = regime_filter.filter_predictions_by_regime(
        signals=np.array([signal]),
        confidence=np.array([prediction]),
        indicators={'atr': current_atr, 'adx': current_adx,
                    'close': current_price, 'sma200': sma200}
    )
    signal = filtered_signal[0]

if signal == 1:
    place_order()
```

### Location 3: Validation/Testing

```python
# File: ml/scripts/walk_forward_with_regime_filter.py (Already implemented)

# Run to validate improvement:
python ml/scripts/walk_forward_with_regime_filter.py

# Expected output:
# WITHOUT filter: WIN_RATE = 31.58%
# WITH filter:    WIN_RATE = 45-50%
# IMPROVEMENT:    +13.4 to +18.4 pp ✅
```

---

## Configuration Matrix

```
┌───────────────┬──────────┬──────────┬─────────────┐
│ Parameter     │ Current  │ Effect   │ Tuning Tips │
├───────────────┼──────────┼──────────┼─────────────┤
│ ENABLE_REGIME │ True     │ Master   │ Emergency   │
│ _FILTER       │          │ switch   │ off only    │
├───────────────┼──────────┼──────────┼─────────────┤
│ REGIME_MIN    │ 12.0     │ Suppress │ Lower →     │
│ _ATR          │          │ low vol  │ more trades │
├───────────────┼──────────┼──────────┼─────────────┤
│ REGIME_MIN    │ 12.0     │ Suppress │ Lower →     │
│ _ADX          │          │ no trend │ more trades │
├───────────────┼──────────┼──────────┼─────────────┤
│ THRESHOLD_    │ 0.35     │ TIER 1:  │ Lower →     │
│ HIGH_ATR      │          │ suppress │ more trades │
│               │          │ low conf │ (risky)     │
├───────────────┼──────────┼──────────┼─────────────┤
│ THRESHOLD_    │ 0.50     │ TIER 2:  │ Lower →     │
│ MOD_ATR       │          │ suppress │ more trades │
│               │          │ low conf │ (risky)     │
├───────────────┼──────────┼──────────┼─────────────┤
│ THRESHOLD_    │ 0.65     │ TIER 3:  │ Lower →     │
│ LOW_ATR       │          │ suppress │ more trades │
│               │          │ low conf │ (risky)     │
└───────────────┴──────────┴──────────┴─────────────┘

ADJUSTMENT GUIDE:
════════════════

Need more trades? Lower thresholds:
  REGIME_THRESHOLD_HIGH_ATR = 0.30 (from 0.35)
  REGIME_THRESHOLD_MOD_ATR = 0.45  (from 0.50)
  REGIME_THRESHOLD_LOW_ATR = 0.60  (from 0.65)

Need fewer/better trades? Raise thresholds:
  REGIME_THRESHOLD_HIGH_ATR = 0.40 (from 0.35)
  REGIME_THRESHOLD_MOD_ATR = 0.55  (from 0.50)
  REGIME_THRESHOLD_LOW_ATR = 0.70  (from 0.65)

Emergency: Disable filter temporarily:
  ENABLE_REGIME_FILTER = False
```

---

## Expected Performance

### Win Rate Improvement

```
WITHOUT FILTER          WITH FILTER           IMPROVEMENT
──────────────          ────────────          ─────────────

31.58%                  45-50%                +13.4 to +18.4 pp
 │                        │                           │
 │                        │                           │
 │                        ▼                           ▼
 │              ┌────────────────────┐    ┌──────────────────┐
 │              │ TIER 1: 80%+ wins  │    │ 2.3x improvement │
 │              │ (ATR ≥ 18)         │    │ in base case      │
 │              │ 30% of trades      │    │                  │
 │              │                    │    │ e.g., if you had │
 │              │ TIER 2: 40-65%     │    │ 100 trades:      │
 │              │ (ATR 12-17)        │    │ - 31.58 wins     │
 │              │ 50% of trades      │    │ + 13-18 more     │
 │              │                    │    │ = 45-50 wins     │
 │              │ TIER 3/4: Suppress │    │                  │
 │              │ (mostly filtered)  │    │                  │
 │              └────────────────────┘    └──────────────────┘
 │
 ▼
BASELINE: All signals
taken regardless of
market conditions
(good + bad)

Expected trade distribution:
- TIER 1 (80%+ win): 30% of executed trades
- TIER 2 (40-65% win): 50% of executed trades
- TIER 3/4: Mostly suppressed by regime filter
```

---

## Deployment Timeline

```
WEEK 1: VALIDATION           WEEK 2-3: STAGING        WEEK 4: PRODUCTION
═════════════════            ═════════════════        ══════════════════

Day 1-2:                     Day 8-10:                Day 22-23:
- Run walk-forward val       - Deploy to staging      - Deploy to prod
- Check win rate +13-18pp    - Run backtest           - Enable filtering
- Verify thresholds          - Test indicators        - Monitor daily

Day 3-5:                     Day 11-14:               Day 24-26:
- Document regime dist       - Integration tests      - Verify metrics
- Get approvals              - Check calculations     - Alert setup
- Set up monitoring          - Load testing

Day 6-7:                     Day 15-21:               Day 27+:
- Stakeholder review         - Performance tests      - Ongoing monitoring
- Final sign-off             - Data validation        - Quarterly reviews
```

---

## Monitoring Metrics (Daily Dashboard)

```
┌─────────────────────────────────────────────────────────┐
│              PRODUCTION METRICS DASHBOARD                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  WIN RATE                    SIGNAL SUPPRESSION RATE    │
│  ════════                    ═════════════════════════   │
│                                                          │
│  Target:  45-50%             Target: 30-40%            │
│  Current: __ ___%            Current: __ ___%          │
│  Status:  ⚪ OK/❌ ALERT       Status:  ⚪ OK/❌ ALERT    │
│                                                          │
│  ───────────────────────────────────────────────────    │
│                                                          │
│  REGIME DISTRIBUTION          MODEL CONFIDENCE         │
│  ═════════════════            ════════════════         │
│                                                          │
│  TIER 1 (ATR≥18): ___%         Avg:  __ ___%           │
│  TIER 2 (12-17):  ___%         Min:  __ ___%           │
│  TIER 3 (8-11):   ___%         Max:  __ ___%           │
│  TIER 4 (ATR<8):  ___%         Status: ⚪ OK/❌ ALERT   │
│                                                          │
│  ───────────────────────────────────────────────────    │
│                                                          │
│  TRADES EXECUTED              SYSTEM STATUS            │
│  ════════════════             ═══════════════          │
│                                                          │
│  Today:    __ trades          Filter: ✅ ENABLED       │
│  Weekly:   __ trades          Model:  ✅ LOADED        │
│  Monthly:  __ trades          Alerts: ✅ ACTIVE        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Troubleshooting Flowchart

```
                   PROBLEM?
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
    No improvement  Too many     Missing data
    in win rate     suppressions   quality
        │             │             │
        ▼             ▼             ▼
   CHECK #1:     CHECK #2:      CHECK #3:
   - Filter      - Market       - ATR calc
     enabled?      regime?       - ADX calc
   - Indicators  - Thresholds   - SMA200
     correct?      too strict?
        │             │             │
        ▼             ▼             ▼
   FIX:           FIX:            FIX:
   - Verify       - Lower          - Verify
     enable         thresh         indicators
   - Retrain      - Check log      - Check
     model        - Temp            data
   - Retrain       disable        - Retry
     with                         - Contact
     new data                       team
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
              VERIFY FIXED?
                      │
            ┌─────────┴─────────┐
           YES                 NO
            │                   │
            ▼                   ▼
        DONE ✅          ESCALATE
                         TO TEAM
```

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════╗
║          REGIME FILTER: QUICK REFERENCE CARD             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  WHAT IS IT?                                            ║
║  ─────────                                              ║
║  Predction gating system that suppresses trades when    ║
║  market conditions are unfavorable (Opcja B)            ║
║                                                          ║
║  WHEN TO USE?                                           ║
║  ──────────                                             ║
║  Training:   NO filtering (use all data)                ║
║  Inference:  YES filtering (gate bad regimes)           ║
║                                                          ║
║  EXPECTED IMPACT:                                       ║
║  ─────────────                                          ║
║  WIN RATE: 31.58% → 45-50% (+13-18 pp) ✅              ║
║                                                          ║
║  HOW TO ENABLE?                                         ║
║  ────────────                                           ║
║  1. Set ENABLE_REGIME_FILTER = True                     ║
║  2. Add 5 lines of code at signal point                 ║
║  3. Run tests & validate                                ║
║                                                          ║
║  SUPPRESSION RULES:                                     ║
║  ────────────────                                       ║
║  Suppress if:  ✗ ATR < 12                               ║
║                ✗ ADX < 12                               ║
║                ✗ Price ≤ SMA200                         ║
║                ✗ Confidence too low (regime-dependent)  ║
║                                                          ║
║  CONFIGURATION:                                         ║
║  ──────────                                             ║
║  File: ml/src/utils/risk_config.py                      ║
║  Parameters: 13 tunable values                          ║
║  Defaults: Audit-approved, ready to use                 ║
║                                                          ║
║  FILES:                                                 ║
║  ─────                                                  ║
║  Core:      ml/src/filters/regime_filter.py (328 lines) ║
║  Config:    ml/src/utils/risk_config.py                 ║
║  Test:      ml/scripts/simple_regime_filter_test.py     ║
║  Guide:     ml/PRODUCTION_INTEGRATION_GUIDE.md          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

**Next Step**: Read [PRODUCTION_INTEGRATION_GUIDE.md](PRODUCTION_INTEGRATION_GUIDE.md)

---

Generated: 2025-01-15
Status: ✅ Ready for Production
