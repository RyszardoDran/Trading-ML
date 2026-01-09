"""Risk management configuration shared across ML Python code.

This module centralizes the canonical ATR-based SL/TP parameters
and trading filter settings. Keep these consistent with ml/SEQUENCE_PIPELINE_README.md.
"""

from __future__ import annotations

# ===== ATR-Based Risk Parameters =====
ATR_PERIOD_M5: int = 14
SL_ATR_MULTIPLIER: float = 1.0
TP_ATR_MULTIPLIER: float = 2.0

# ===== Model Threshold Parameters =====
MIN_PRECISION_THRESHOLD: float = 0.7  # Rozluźniony, by pozwolić na więcej sygnałów (więcej trade'ów)
MIN_TRADES_PER_TEST: int = 5         # Minimalna liczba transakcji w teście, aby uznać model za wiarygodny
MAX_TRADES_PER_DAY: int = 150          # Podwyższony limit by pozwolić modelowi wysyłać więcej sygnałów w real-live

# ===== Threshold Optimization Strategy =====
USE_EV_OPTIMIZATION: bool = True     # Domyślnie F1-optimized; EV enabled to explore profit-driven thresholds
USE_HYBRID_OPTIMIZATION: bool = True  # Hybryda: EV-optimized ale z precision AND recall floors (REKOMENDOWANE)
# CAP-FIRST DEFAULT:
# When MAX_TRADES_PER_DAY is active, a high recall floor can become infeasible and
# forces threshold selection into fallbacks. We default to no recall floor and
# let the daily cap control trade frequency.
MIN_RECALL_FLOOR: float = 0.30  # Zapewnij minimalny recall (więcej sygnałów), ale nie za wysoki aby uniknąć infeasibility
EV_WIN_COEFFICIENT: float = 1.2       # Mnożnik dla True Positives (wygrane transakcje)
EV_LOSS_COEFFICIENT: float = -1.0     # Mnożnik dla False Positives (przegrane transakcje)

# ===== Cost-Sensitive Learning (POINT 1) =====
USE_COST_SENSITIVE_LEARNING: bool = True  # Waży błędy: False Positives bardziej "kosztowne" niż False Negatives
SAMPLE_WEIGHT_POSITIVE: float = 2.0   # Zmniejszona waga pozytywów by zwiększyć recall (więcej trade'ów)
SAMPLE_WEIGHT_NEGATIVE: float = 1.0   # Waga dla True Negatives (baseline)

# ===== Target (SL/TP) Simulation Parameters =====
MIN_HOLD_M5_CANDLES: int = 2           # Minimalny czas: 2 świece M5 = 10 minut
MAX_HORIZON_M5_CANDLES: int = 60       # Maksymalny czas czekania: 60 świec M5 = 300 minut (5 godzin)

# ===== Sequence Parameters =====
WINDOW_SIZE: int = 80                # Skrócony window by szybciej wykrywać setups i wygenerować więcej próbek

# ===== Trading Filters =====
ENABLE_M5_ALIGNMENT: bool = False      # M5 candle close alignment
ENABLE_TREND_FILTER: bool = False      # Disabled: was choking live signals; rely on model + regime filter
ENABLE_PULLBACK_FILTER: bool = False   # RSI_M5 pullback guard

# ===== Trend Filter Parameters (when enabled) =====
# dist_sma_200 is normalized by ATR (see engineer_m5.py), so -0.5 means
# allow up to ~0.5 ATR below SMA200 to avoid choking signals in mild pullbacks.
TREND_MIN_DIST_SMA200: float = -0.5
# Relaxed to increase signal frequency (still filters very low-trend regimes)
TREND_MIN_ADX: float = 8.0

# ===== Pullback Filter Parameters (when enabled) =====
PULLBACK_MAX_RSI_M5: float = 75.0

# ===== MARKET REGIME FILTER (AUDIT 4) =====
# Gating: Only trade in favorable market conditions (high-ATR + trending markets)
# Implementation based on Audit 4 findings: ATR dominates performance (r=0.82)
ENABLE_REGIME_FILTER: bool = False     # Disabled: allow all regimes for backtesting/live (user request)

# Regime classification thresholds
REGIME_MIN_ATR_FOR_TRADING: float = 8.0       # Rozluźnione, by nie odrzucać umiarkowanie niskiej zmienności
                                               # Fold 9 (88%): ATR=20, Fold 11 (61.9%): ATR=16
                                               # Fold 2 (0%): ATR=8 → SKIP

REGIME_MIN_ADX_FOR_TRENDING: float = 8.0      # Lower ADX threshold to allow more trending detections
                                               # Fold 9: ADX=20+, Fold 11: ADX=16
                                               # Fold 2: ADX=8 → SKIP

REGIME_MIN_PRICE_DIST_SMA200: float = 0.0      # Allow setups closer to SMA200 (więcej sygnałów)
                                               # Fold 9: +30 pips above, Fold 11: +20 pips
                                               # Fold 2: ~0 pips (flat) → SKIP

# Adaptive threshold based on ATR (higher ATR = more aggressive)
REGIME_ADAPTIVE_THRESHOLD: bool = True         # Use adaptive threshold based on volatility
REGIME_THRESHOLD_HIGH_ATR: float = 0.35        # Threshold for ATR > 18 (Fold 9 regime)
REGIME_THRESHOLD_MOD_ATR: float = 0.50         # Threshold for ATR 12-17 (Fold 11 regime)
REGIME_THRESHOLD_LOW_ATR: float = 0.65         # Threshold for ATR < 12 (don't trade, but if forced)

# Regime thresholds for ATR tiers
REGIME_HIGH_ATR_THRESHOLD: float = 18.0        # TIER 1: High volatility (excellent: 80%+ wins)
REGIME_MOD_ATR_THRESHOLD: float = 12.0         # TIER 2: Moderate volatility (good: 40-65% wins)
# Below REGIME_MOD_ATR_THRESHOLD is TIER 3/4: Low volatility (avoid: 0-35% wins)

# ===== CVD (Cumulative Volume Delta) Parameters =====
ENABLE_CVD_INDICATOR: bool = True     # Włącza/wyłącza wskaźnik CVD w modelu
CVD_LOOKBACK_WINDOW: int = 50          # Liczba świec (M5) do obliczania normalizacji z-score
# ===== Feature Toggles (Włączanie/Wyłączanie cech modelu) =====
FEAT_ENABLE_RSI: bool = True           # Wskaźnik siły względnej (RSI) - wykrywa wykupienie/wyprzedanie
FEAT_ENABLE_BB_POS: bool = True        # Pozycja względem Wstęg Bollingera - mierzy zmienność i poziomy ekstremalne
FEAT_ENABLE_SMA_DIST: bool = True      # Odległość od średniej kroczącej (SMA20) - wykrywa powrót do średniej
FEAT_ENABLE_STOCH: bool = True         # Oscylator stochastyczny - mierzy pęd i punkty zwrotne
FEAT_ENABLE_MACD: bool = True          # MACD Histogram - wykrywa zmiany trendu i pędu
FEAT_ENABLE_ATR: bool = True           # Znormalizowany ATR - mierzy relatywną zmienność rynku
FEAT_ENABLE_ADX: bool = True           # Wskaźnik ADX - mierzy siłę aktualnego trendu
FEAT_ENABLE_SMA200: bool = True        # Odległość od SMA200 - określa główny trend długoterminowy
FEAT_ENABLE_RETURNS: bool = True       # Proste stopy zwrotu - informacja o kierunku ostatniej świecy
FEAT_ENABLE_VOLUME_M5: bool = True     # Analiza wolumenu M5 - znormalizowany wolumen względem średniej
FEAT_ENABLE_VOLUME_M15: bool = True    # Analiza wolumenu M15 - znormalizowany wolumen z wyższego interwału
FEAT_ENABLE_OBV: bool = True           # On-Balance Volume (M5, M15, M60) - presja kupno/sprzedaż
FEAT_ENABLE_MFI: bool = False           # Money Flow Index (M5, M15, M60) - RSI ważony ceną i volumem
FEAT_ENABLE_M15_CONTEXT: bool = True   # Kontekst z interwału 15-minutowego (RSI, BB, SMA)
FEAT_ENABLE_M60_CONTEXT: bool = False  # ❌ DISABLED: M60 dominates (75% importance) → overfitting on stale indicators

def risk_reward_ratio() -> float:
    return TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER
