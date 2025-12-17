# 📚 INDEX DOKUMENTACJI REFACTORYZACJI

## Zawartość Całości

Refactoryzacja podzielona na **7 etapów** z dedykowaną dokumentacją dla każdego.

---

## 📋 Dokumenty (W Porządku Czytania)

### 1️⃣ **ROADMAP.md** (START TUTAJ) 🗺️
- **Cel**: Wizualny przegląd całej refactoryzacji
- **Zawartość**:
  - Diagram: Monolityczny kod → Architektura modułowa
  - Struktura katalogów (przed/po)
  - 7-etapowy plan w formie flowchart
  - Szacunkowa liczba linii kodu na etap
  - Zasady refactoryzacji (DO/NIE ROBIĆ)
  - Gdzie znaleźć informacje
- **Czas czytania**: ~5 min
- **Kiedy czytać**: NAJPIERW - zrozumienie ogólnego kierunku

---

### 2️⃣ **REFACTOR_PLAN.md** (PRZEGLĄD CAŁOŚCI)
- **Cel**: Szczegółowy plan refactoryzacji
- **Zawartość**:
  - Przegląd (1740 linii, struktura monolityczna)
  - Zasady refactoryzacji
  - Docelowa struktura katalogów (pełna)
  - Plan 7 etapów ze szczegółami każdego
  - Zalety refactoryzacji
  - Ostateczna struktura `src/`
- **Czas czytania**: ~15 min
- **Kiedy czytać**: Po ROADMAP.md, zanim zaczniesz implementację

---

### 3️⃣ **REFACTOR_ETAP_1.md** (STRUKTURA KATALOGÓW) 🏗️
- **Cel**: Implementacja Etapu 1
- **Zawartość**:
  - Lista rzeczy do zrobienia (katalogi, pliki __init__.py, moduły)
  - Szczegółowa implementacja każdego pliku
  - Pliki `__init__.py` (gotowe do kopiowania)
  - `config.py` z `PipelineConfig` dataclass
  - `sequences/config.py` z `SequenceFilterConfig`
  - `split.py` (szkielet)
  - `data_loading/validators.py` (przenieść `_validate_schema`)
  - `data_loading/loaders.py` (przenieść `load_all_years`)
  - Kontrola jakości (testy do wykonania)
  - Metryki sukcesu
  - Jak to będzie wyglądać po Etapie 1
- **Czas czytania**: ~20 min
- **Kiedy czytać**: Zanim zaczniesz Etap 1
- **Ćwiczenie**: Wykonaj Etap 1 według instrukcji

---

### 4️⃣ **REFACTOR_ETAP_2.md** (INŻYNIERIA CECH) ✨ `features/`
- **Cel**: Implementacja Etapu 2
- **Zawartość** (planowane):
  - Rozbicie `engineer_candle_features()` na moduły
  - `features/engineer.py` - główna funkcja
  - `features/indicators.py` - wszystkie indykatory techniczne
  - `features/m5_context.py` - kontekst M5 (resampling)
  - `features/time_features.py` - kodowanie godziny/minuty
  - Testy porównujące output z oryginalnym
- **Status**: ⏳ (Będzie po Etapie 1)

---

### 5️⃣ **REFACTOR_ETAP_3.md** (CEL & SEKWENCJE) 🎯
- **Cel**: Implementacja Etapu 3
- **Zawartość** (planowane):
  - `targets/target_maker.py` - przeniesienie `make_target()`
  - `sequences/sequencer.py` - przeniesienie `create_sequences()`
  - `sequences/filters.py` - filtry sesji/trendu/pullback
  - `split.py` - przeniesienie `split_sequences()`
  - Testy porównujące sekwencje z oryginalnym
- **Status**: ⏳ (Będzie po Etapie 2)

---

### 6️⃣ **REFACTOR_ETAP_4.md** (TRAINING & EWALUACJA) 🚀
- **Cel**: Implementacja Etapu 4
- **Zawartość** (planowane):
  - `training/xgb_trainer.py` - przeniesienie `train_xgb()`
  - `training/evaluation.py` - ewaluacja i threshold picking
  - `training/daily_cap.py` - limit na dzień
  - `training/feature_analysis.py` - analiza importance
  - `training/artifacts.py` - zapis artefaktów
  - Testy trenowania modelu
- **Status**: ⏳ (Będzie po Etapie 3)

---

### 7️⃣ **REFACTOR_ETAP_5.md** (REFAKTOR GŁÓWNY + SKRYPTY CLI) 🎬
- **Cel**: Implementacja Etapu 5
- **Zawartość** (planowane):
  - Refaktor `sequence_training_pipeline.py` - usunięcie przeniosonych funkcji
  - `run_pipeline()` - główna orchestracja (publiczne API)
  - `scripts/train_sequence_model.py` - CLI do trenowania
  - Argumenty CLI, logowanie, zapis do `outputs/`
  - Testy uruchomienia skryptu CLI
- **Status**: ⏳ (Będzie po Etapie 4)

---

### 8️⃣ **REFACTOR_ETAP_6.md** (DODATKOWE SKRYPTY) 📊
- **Cel**: Implementacja Etapu 6
- **Zawartość** (planowane):
  - `scripts/eval_model.py` - ewaluacja wytrenowanego modelu
  - `scripts/analyze_features.py` - analiza feature importance
  - `scripts/backtest_strategy.py` - backtest scenariuszy (opcjonalnie)
- **Status**: ⏳ (Będzie po Etapie 5)

---

### 9️⃣ **REFACTOR_ETAP_7.md** (TESTY & DOKUMENTACJA) ✅
- **Cel**: Implementacja Etapu 7
- **Zawartość** (planowane):
  - `tests/test_data_loading.py` - walidacja, obsługa błędów
  - `tests/test_feature_engineering.py` - porównanie output
  - `tests/test_sequences.py` - tworzenie sekwencji, filtry
  - `tests/test_training.py` - trening, ewaluacja
  - Pokrycie testami > 90%
  - Pełna dokumentacja modułów
- **Status**: ⏳ (Będzie po Etapie 6)

---

## 🗂️ Strukturа Katalogów (Gdzie Są Dokumenty)

```
ml/src/pipelines/
├── ROADMAP.md                     ← 🗺️ START TUTAJ
├── REFACTOR_PLAN.md               ← 📋 Przegląd całości
├── REFACTOR_ETAP_1.md             ← 🏗️ Struktura katalogów
├── REFACTOR_ETAP_2.md             ← ✨ Inżynieria cech (planowe)
├── REFACTOR_ETAP_3.md             ← 🎯 Cel & sekwencje (planowe)
├── REFACTOR_ETAP_4.md             ← 🚀 Training & ewaluacja (planowe)
├── REFACTOR_ETAP_5.md             ← 🎬 Refaktor główny (planowe)
├── REFACTOR_ETAP_6.md             ← 📊 Dodatkowe skrypty (planowe)
├── REFACTOR_ETAP_7.md             ← ✅ Testy & dokumentacja (planowe)
├── INDEX.md                       ← 📚 TUTAJ JESTEŚ
│
├── sequence_training_pipeline.py  ← Główny plik (do refactoryzacji)
├── config.py                      ← 🆕 (Etap 1)
├── split.py                       ← 🆕 (Etap 1)
│
├── data_loading/                  ← 🆕 (Etap 1)
│   ├── __init__.py
│   ├── loaders.py
│   └── validators.py
│
├── features/                      ← 🆕 (Etap 2)
│   ├── __init__.py
│   ├── engineer.py
│   ├── indicators.py
│   ├── m5_context.py
│   └── time_features.py
│
├── targets/                       ← 🆕 (Etap 3)
│   ├── __init__.py
│   └── target_maker.py
│
├── sequences/                     ← 🆕 (Etap 1+3)
│   ├── __init__.py
│   ├── config.py
│   ├── sequencer.py
│   └── filters.py
│
├── training/                      ← 🆕 (Etap 4)
│   ├── __init__.py
│   ├── xgb_trainer.py
│   ├── evaluation.py
│   ├── daily_cap.py
│   ├── feature_analysis.py
│   └── artifacts.py
│
└── utils/                         ← 🆕 (Etap 4)
    └── __init__.py
```

---

## 📖 Rekomendowany Plan Czytania

### Dla Nowych Osób
1. Przeczytaj **ROADMAP.md** (5 min) - zrozumienie kierunku
2. Przeczytaj **REFACTOR_PLAN.md** (15 min) - szczegóły całości
3. Przeczytaj **REFACTOR_ETAP_1.md** (20 min) - szczegóły implementacji
4. Zacznij implementować **Etap 1**

### Dla Osób Już Zaznajomionych
1. Przeskocz do odpowiedniego **REFACTOR_ETAP_N.md**
2. Implementuj etap
3. Wróć do **INDEX.md**, jeśli coś niejasne

### Dla Przeglądu (30 min)
1. **ROADMAP.md** - diagram i flow
2. **REFACTOR_PLAN.md** - tabela etapów i zalety

---

## 🎯 Cele Każdego Etapu

| Etap | Cel | Katalogi | Funkcje |
|------|-----|----------|---------|
| 1 | Struktura, importy | 6 nowych | 2 przeniesione |
| 2 | Inżynieria cech | 1 nowy | 1 duża rozbita |
| 3 | Cel & sekwencje | 1 nowy | 4 przeniesione |
| 4 | Training & ewaluacja | 1 nowy | 5 przeniesione |
| 5 | Refaktor główny + CLI | 1 nowy | 1 orchestracja |
| 6 | Dodatkowe skrypty | - | 3 nowe |
| 7 | Testy & dokumentacja | 1 nowy | ~10 testów |

---

## 🔍 Szybki Lookup

### Szukasz informacji o...

**Strukturze katalogów?**
- → Przeczytaj: ROADMAP.md (diagram) → REFACTOR_PLAN.md (pełna struktura)

**Jak zacząć Etap 1?**
- → Przeczytaj: REFACTOR_ETAP_1.md (lista, instrukcje)

**Inżynieria cech (Etap 2)?**
- → Czekaj: REFACTOR_ETAP_2.md (będzie po Etapie 1)

**CLI skrypty (Etap 5)?**
- → Czekaj: REFACTOR_ETAP_5.md (będzie po Etapach 1-4)

**Testy (Etap 7)?**
- → Czekaj: REFACTOR_ETAP_7.md (będzie na końcu)

**Wszystkie funkcje, które będą przeniesione?**
- → Przeczytaj: REFACTOR_PLAN.md (diagram na początku)

**Metryki sukcesu?**
- → Przeczytaj: Każdy REFACTOR_ETAP_N.md (sekcja "Kontrola Jakości")

---

## ✅ Kontrola Listy

Przed rozpoczęciem każdego etapu:

- [ ] Przeczytałeś **ROADMAP.md**
- [ ] Przeczytałeś **REFACTOR_PLAN.md**
- [ ] Przeczytałeś **REFACTOR_ETAP_1.md** (lub odpowiedni dla etapu)
- [ ] Zrozumiałeś zasady refactoryzacji (DO/NIE ROBIĆ)
- [ ] Wiesz, gdzie katalogi będą (struktura, separacja wkładu/wyniku)
- [ ] Jesteś gotów do implementacji

---

## 🚀 Uruchamianie Etapów

### Etap 1 (Teraz)
```bash
# Przeczytaj REFACTOR_ETAP_1.md
# Utwórz katalogi
# Utwórz pliki __init__.py
# Przenieś funkcje
# Sprawdź importy
```

### Etapy 2-7
```bash
# Każdy etap ma swój REFACTOR_ETAP_N.md
# Każdy etap można robić w osobnym PR
# Każdy etap ma listy kontrolne i metryki sukcesu
```

---

## 📞 Pytania Częste

**P: Ile czasu zajmie całość?**
O: ~1-2 tygodnie (7 etapów × 1-2 dni każdy), zależy od tempa

**P: Czy mogę pracować na kilku etapach jednocześnie?**
O: NIE - każdy etap zależy od poprzedniego (struktura katalogów musi być)

**P: Czy mogę zmienić logikę w trakcie refactoryzacji?**
O: NIE - refactoring jest TYLKO reorganizacją kodu, bez zmian logiki

**P: Gdzie trafiają wyniki trenowania?**
O: Do `ml/outputs/` (modele, metryki, logi) - nigdy do `scripts/`

**P: Czy mogę usunąć oryginalny plik `sequence_training_pipeline.py`?**
O: Dopiero po Etapie 5, gdy wszystkie funkcje będą w modułach

---

## 📊 Status Refactoryzacji

```
Etap 1: Struktura katalogów         [ ] ⏳ Gotowy do implementacji
Etap 2: Inżynieria cech            [ ] ⏳ Czeka na Etap 1
Etap 3: Cel & sekwencje            [ ] ⏳ Czeka na Etap 2
Etap 4: Training & ewaluacja       [ ] ⏳ Czeka na Etap 3
Etap 5: Refaktor główny + CLI      [ ] ⏳ Czeka na Etap 4
Etap 6: Dodatkowe skrypty          [ ] ⏳ Czeka na Etap 5
Etap 7: Testy & dokumentacja       [ ] ⏳ Czeka na Etap 6
```

---

**Last Updated**: 2025-12-16
**Status**: 📚 Dokumentacja Complete, Implementacja Pending
**Następny Krok**: Przeczytaj ROADMAP.md, potem REFACTOR_ETAP_1.md
