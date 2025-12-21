"""Risk management configuration shared across ML Python code.

This module centralizes the canonical ATR-based SL/TP parameters
and trading filter settings. Keep these consistent with ml/SEQUENCE_PIPELINE_README.md.
"""

from __future__ import annotations

# ===== ATR-Based Risk Parameters =====
ATR_PERIOD_M5: int = 14
SL_ATR_MULTIPLIER: float = 1
TP_ATR_MULTIPLIER: float = 3.0

# ===== Model Threshold Parameters =====
MIN_PRECISION_THRESHOLD: float = 0.55
MIN_TRADES_PER_TEST: int = 5         # Minimalna liczba transakcji w teście, aby uznać model za wiarygodny
MAX_TRADES_PER_DAY: int = 10         # Limit transakcji na dzień (zapobiega overtradingowi)

# ===== Threshold Optimization Strategy =====
USE_EV_OPTIMIZATION: bool = False     # Domyślnie F1-optimized (lepsze wyniki), EV optional dla advanced users
USE_HYBRID_OPTIMIZATION: bool = True  # Hybryda: EV-optimized ale z precision AND recall floors (REKOMENDOWANE)
MIN_RECALL_FLOOR: float = 0.20        # Minimalny recall dla hybrydy (20% = więcej trades)
EV_WIN_COEFFICIENT: float = 1.0       # Mnożnik dla True Positives (wygrane transakcje)
EV_LOSS_COEFFICIENT: float = -1.0     # Mnożnik dla False Positives (przegrane transakcje)

# ===== Target (SL/TP) Simulation Parameters =====
MIN_HOLD_M5_CANDLES: int = 2           # Minimalny czas: 2 świece M5 = 10 minut
MAX_HORIZON_M5_CANDLES: int = 60       # Maksymalny czas czekania: 60 świec M5 = 300 minut (5 godzin)

# ===== Sequence Parameters =====
WINDOW_SIZE: int = 50                # Liczba poprzednich świec (M5) jako wejście dla modelu

# ===== Trading Filters =====
ENABLE_M5_ALIGNMENT: bool = False      # M5 candle close alignment
ENABLE_TREND_FILTER: bool = True       # Price above SMA200 and ADX threshold
ENABLE_PULLBACK_FILTER: bool = False   # RSI_M5 pullback guard

# ===== Trend Filter Parameters (when enabled) =====
TREND_MIN_DIST_SMA200: float = 0
TREND_MIN_ADX: float = 15.0

# ===== Pullback Filter Parameters (when enabled) =====
PULLBACK_MAX_RSI_M5: float = 75.0

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
FEAT_ENABLE_M60_CONTEXT: bool = True   # Kontekst z interwału 60-minutowego (RSI, BB)

def risk_reward_ratio() -> float:
    return TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER
