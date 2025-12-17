# ✅ CHECKLIST REFACTORYZACJI

Drukuj i zaznaczaj postęp! 📋

---

## 📖 FAZA 0: Przygotowanie

### Przeczytanie Dokumentacji
- [ ] Przeczytałem `QUICK_REFERENCE.md` (5 min)
- [ ] Przeczytałem `ROADMAP.md` (5 min)
- [ ] Przeczytałem `REFACTOR_PLAN.md` (15 min)
- [ ] Przeczytałem `INDEX.md` (5 min)
- [ ] Zrozumiałem 7-etapowy plan
- [ ] Zrozumiałem strukturę katalogów (przed/po)
- [ ] Zrozumiałem zasady DO/NIE ROBIĆ
- [ ] Zrozumiałem separację wkładu/wyniku

### Przygotowanie Środowiska
- [ ] Mam dostęp do `ml/src/pipelines/`
- [ ] Mam dostęp do `ml/data/` (dane OHLCV)
- [ ] Wiem, gdzie będą katalogi na wyniki (`ml/outputs/`)
- [ ] Mam dostęp do `ml/scripts/` (będzie tworzony)
- [ ] Mogę czytać i edytować pliki
- [ ] Mogę tworzyć katalogi i pliki

---

## 🏗️ ETAP 1: Struktura Katalogów

### Przeczytanie Instrukcji
- [ ] Przeczytałem `REFACTOR_ETAP_1.md`
- [ ] Rozumiem, co trzeba zrobić w Etapie 1
- [ ] Wiem, gdzie znajduje się gotowy kod do kopiowania

### Tworzenie Katalogów (ml/src/pipelines/)
- [ ] `data_loading/`
- [ ] `features/`
- [ ] `targets/`
- [ ] `sequences/`
- [ ] `training/`
- [ ] `utils/`

### Tworzenie Katalogów (ml/)
- [ ] `scripts/`
- [ ] `outputs/`
- [ ] `outputs/models/`
- [ ] `outputs/metrics/`
- [ ] `outputs/analysis/`
- [ ] `outputs/logs/`

### Tworzenie Plików __init__.py
- [ ] `ml/src/pipelines/data_loading/__init__.py`
- [ ] `ml/src/pipelines/features/__init__.py`
- [ ] `ml/src/pipelines/targets/__init__.py`
- [ ] `ml/src/pipelines/sequences/__init__.py`
- [ ] `ml/src/pipelines/training/__init__.py`
- [ ] `ml/src/pipelines/utils/__init__.py`

### Tworzenie Plików Konfiguracji
- [ ] `ml/src/pipelines/config.py` (z gotowego kodu z ETAP_1.md)
- [ ] `ml/src/pipelines/sequences/config.py` (SequenceFilterConfig z gotowego kodu)
- [ ] `ml/src/pipelines/split.py` (szkielet)

### Przenoszenie Funkcji (Etap 1)
- [ ] `_validate_schema()` → `ml/src/pipelines/data_loading/validators.py`
- [ ] `load_all_years()` → `ml/src/pipelines/data_loading/loaders.py`

### Testowanie Importów
- [ ] ✅ `from ml.src.pipelines.data_loading import load_all_years`
- [ ] ✅ `from ml.src.pipelines.sequences.config import SequenceFilterConfig`
- [ ] ✅ `from ml.src.pipelines.config import PipelineConfig`
- [ ] ✅ `from ml.src.pipelines.split import split_sequences` (dostawanie bez błędu)

### Sprawdzenie Pozostałego Kodu
- [ ] Usunąłem przeniesione funkcje z głównego pliku
- [ ] Dodałem importy do głównego pliku
- [ ] Główny plik `sequence_training_pipeline.py` kompiluje się bez błędów
- [ ] Żaden test się nie zepsuł (jeśli były)

### Dokumentacja Etapu 1
- [ ] Opisałem, co zostało zrobione w Etapie 1
- [ ] Dodałem notatki o jakichkolwiek problemach
- [ ] Zaznaczył czeklisty w tym dokumencie

### Commit Etapu 1
- [ ] Commitnąłem zmiany: `feat: Etap 1 - struktura katalogów`
- [ ] Wiadomość commitu zawiera co było robione

---

## ✨ ETAP 2: Inżynieria Cech (features/) - ✅ COMPLETE

### Przeczytanie Instrukcji
- [x] Przeczytałem `REFACTOR_ETAP_2.md` (będzie dostępny)

### Tworzenie Plików
- [x] `ml/src/pipelines/features/engineer.py`
- [x] `ml/src/pipelines/features/indicators.py`
- [x] `ml/src/pipelines/features/m5_context.py`
- [x] `ml/src/pipelines/features/time_features.py`

### Przenoszenie Kodu
- [x] `engineer_candle_features()` → `features/engineer.py`
- [x] Wszystkie indykatory → `features/indicators.py`
- [x] M5 context → `features/m5_context.py`
- [x] Time features → `features/time_features.py`

### Testowanie
- [x] ✅ `from ml.src.pipelines.features import engineer_candle_features`
- [x] ✅ Porównanie output z oryginalnym (instrukcje w ETAP_2.md)
- [x] ✅ Wszystkie testy zielone

### Dokumentacja
- [x] Opisałem zmiany
- [x] Zaznaczył czeklisty

### Commit
- [x] `feat: Etap 2 - inżynieria cech (features/)`

---

## 🎯 ETAP 3: Cel & Sekwencje

### Przeczytanie Instrukcji
- [ ] Przeczytałem `REFACTOR_ETAP_3.md` (będzie dostępny)

### Tworzenie Plików
- [ ] `ml/src/pipelines/targets/target_maker.py`
- [ ] `ml/src/pipelines/sequences/sequencer.py`
- [ ] `ml/src/pipelines/sequences/filters.py`

### Przenoszenie Kodu
- [ ] `make_target()` → `targets/target_maker.py`
- [ ] `create_sequences()` → `sequences/sequencer.py`
- [ ] `filter_by_session()` → `sequences/filters.py`
- [ ] `split_sequences()` → `split.py` (przeniesiony z Etapu 1 lub tutaj)

### Testowanie
- [ ] ✅ `from ml.src.pipelines.targets import make_target`
- [ ] ✅ `from ml.src.pipelines.sequences import create_sequences`
- [ ] ✅ Porównanie sekwencji z oryginalnym
- [ ] ✅ Wszystkie testy zielone

### Dokumentacja & Commit
- [ ] `feat: Etap 3 - cel & sekwencje`

---

## 🚀 ETAP 4: Training & Ewaluacja

### Przeczytanie Instrukcji
- [ ] Przeczytałem `REFACTOR_ETAP_4.md` (będzie dostępny)

### Tworzenie Plików
- [ ] `ml/src/pipelines/training/xgb_trainer.py`
- [ ] `ml/src/pipelines/training/evaluation.py`
- [ ] `ml/src/pipelines/training/daily_cap.py`
- [ ] `ml/src/pipelines/training/feature_analysis.py`
- [ ] `ml/src/pipelines/training/artifacts.py`

### Przenoszenie Kodu
- [ ] `train_xgb()` → `training/xgb_trainer.py`
- [ ] `evaluate()` + `_pick_best_threshold()` → `training/evaluation.py`
- [ ] `_apply_daily_cap()` → `training/daily_cap.py`
- [ ] `analyze_feature_importance()` → `training/feature_analysis.py`
- [ ] `save_artifacts()` → `training/artifacts.py`

### Testowanie
- [ ] ✅ Wszystkie importy działają
- [ ] ✅ Trening modelu działa
- [ ] ✅ Metryki zgadzają się z oryginalnym
- [ ] ✅ Artefakty są zapisywane do `ml/outputs/`

### Dokumentacja & Commit
- [ ] `feat: Etap 4 - training & ewaluacja`

---

## 🎬 ETAP 5: Refaktor Główny + CLI

### Przeczytanie Instrukcji
- [ ] Przeczytałem `REFACTOR_ETAP_5.md` (będzie dostępny)

### Refaktor Głównego Pliku
- [ ] Usunąłem wszystkie przeniesione funkcje z `sequence_training_pipeline.py`
- [ ] Dodałem importy z modułów
- [ ] `run_pipeline()` został zostawiony jako publiczne API
- [ ] Główny plik ma ~150 linii (zamiast 1740)

### Tworzenie Skrypty CLI
- [ ] `ml/scripts/train_sequence_model.py`
  - [ ] Sparsowanie argumentów (`--window-size`, `--year-filter`, etc.)
  - [ ] Wołanie `run_pipeline()`
  - [ ] Logowanie
  - [ ] Zapis wyników do `ml/outputs/`

### Testowanie Skryptu CLI
- [ ] ✅ `python ml/scripts/train_sequence_model.py --help`
- [ ] ✅ `python ml/scripts/train_sequence_model.py` (z domyślnymi parametrami)
- [ ] ✅ Wyniki trafiają do `ml/outputs/`

### Dokumentacja & Commit
- [ ] `feat: Etap 5 - refaktor główny + CLI`

---

## 📊 ETAP 6: Dodatkowe Skrypty

### Przeczytanie Instrukcji
- [ ] Przeczytałem `REFACTOR_ETAP_6.md` (będzie dostępny)

### Tworzenie Skryptów
- [ ] `ml/scripts/eval_model.py`
  - [ ] Ewaluacja wytrenowanego modelu
  - [ ] Argumenty: `--model-path`, `--data-path`
  - [ ] Zapis metryki do `ml/outputs/metrics/`

- [ ] `ml/scripts/analyze_features.py`
  - [ ] Analiza feature importance
  - [ ] Argumenty: `--model-path`
  - [ ] Zapis do `ml/outputs/analysis/`

- [ ] `ml/scripts/backtest_strategy.py` (opcjonalnie)
  - [ ] Backtest ze scenariuszami
  - [ ] Zapis wyników do `ml/outputs/`

### Testowanie
- [ ] ✅ Każdy skrypt ma `--help`
- [ ] ✅ Każdy skrypt się uruchamia bez błędów
- [ ] ✅ Wyniki są zapisywane do `ml/outputs/`

### Dokumentacja & Commit
- [ ] `feat: Etap 6 - dodatkowe skrypty`

---

## ✅ ETAP 7: Testy & Dokumentacja

### Przeczytanie Instrukcji
- [ ] Przeczytałem `REFACTOR_ETAP_7.md` (będzie dostępny)

### Tworzenie Testów
- [ ] `ml/tests/conftest.py` - pytest fixtures
- [ ] `ml/tests/test_data_loading.py`
  - [ ] Testy `load_all_years()`
  - [ ] Testy `validate_schema()`
  - [ ] Obsługa błędów
- [ ] `ml/tests/test_feature_engineering.py`
  - [ ] Porównanie output z oryginalnym
  - [ ] Testy indykatorów
- [ ] `ml/tests/test_sequences.py`
  - [ ] Tworzenie sekwencji
  - [ ] Filtry (sesja, trend, pullback)
- [ ] `ml/tests/test_training.py`
  - [ ] Trening modelu
  - [ ] Ewaluacja
  - [ ] Zapis artefaktów

### Pokrycie Testami
- [ ] ✅ Całkowite pokrycie > 90%
- [ ] ✅ Wszystkie testy zielone
- [ ] ✅ Brak ostrzeżeń linterów

### Dokumentacja Modułów
- [ ] ✅ Każdy moduł ma docstring (Purpose, How, Example)
- [ ] ✅ Każda funkcja ma pełny docstring (Args, Returns, Raises, Examples)
- [ ] ✅ Każdy moduł w `__init__.py` ma `__all__`

### Dokumentacja Projektu
- [ ] ✅ README.md w `ml/` z instrukcjami
- [ ] ✅ Instrukcje instalacji
- [ ] ✅ Instrukcje uruchomienia skryptów
- [ ] ✅ Przykład użycia API

### Finalna Walidacja
- [ ] ✅ Wszystkie importy działają
- [ ] ✅ Żaden plik nie ma warningów
- [ ] ✅ Kod jest sformatowany (Black, isort)
- [ ] ✅ Testy mają pokrycie > 90%

### Dokumentacja & Commit
- [ ] `feat: Etap 7 - testy & dokumentacja`
- [ ] `docs: Aktualizacja README i dokumentacji`

---

## 🎉 FINALIZACJA

### Ostateczne Sprawdzenia
- [ ] Wszystkie 7 etapów ukończone
- [ ] Wszystkie katalogi na miejscu
- [ ] Wszystkie pliki są w odpowiednich modulach
- [ ] Wszystkie importy działają
- [ ] Wszystkie testy są zielone
- [ ] Kod jest zdokumentowany
- [ ] Wyniki (modele, metryki) trafiają do `ml/outputs/`
- [ ] Skrypty trafiają do `ml/scripts/`
- [ ] Brak mieszania wkładu/wyniku

### Ostateczny Commit & PR
- [ ] Commitnąłem finalne zmiany
- [ ] Wiadomość commitu: `feat: Refactoryzacja complete - 7 etapów`
- [ ] Opisałem wszystkie zmiany w PR
- [ ] Zażądałem review
- [ ] PR został zaaprobowany i zmergowany

### Świętowanie 🎊
- [ ] Zaktualizowałem status w tablicy projektów
- [ ] Poinformowałem zespół o zakończeniu
- [ ] Zatwierdziłem, że projekt jest refaktoryzowany i gotowy na dalszy rozwój

---

## 📈 Metryki Sukcesu

### Przed
```
Linie kodu:                 1740 (w jednym pliku)
Modułów:                    0 (wszystko w jednym)
Testowalność:               Niska
Dokumentacja:               Brak
Pokrycie testami:           0%
```

### Po
```
Linie kodu:                 ~2400 (lepiej zorganizowane)
Modułów:                    20+
Testowalność:               Wysoka (każdy moduł testowany)
Dokumentacja:               KOMPLETNA
Pokrycie testami:           > 90%
```

---

## 📋 Referencje

| Dokument | Link |
|----------|------|
| QUICK_REFERENCE.md | Szybka karta |
| ROADMAP.md | Diagram i plan |
| REFACTOR_PLAN.md | Szczegóły |
| INDEX.md | Mapa dokumentacji |
| REFACTOR_ETAP_1.md | Etap 1 |
| REFACTOR_ETAP_2.md | Etap 2 (będzie) |
| ... | ... |
| PROJECT_MAP.md | Mapa projektu |

---

## 📞 Pytania?

**Coś nie działa?** → Przeczytaj dokumentację danego etapu
**Nie wiesz co robić?** → Przeczytaj INDEX.md
**Potrzebujesz szybkiej odpowiedzi?** → QUICK_REFERENCE.md

---

**Druk i Zaznaczaj!** 📋✅

*Powodzenia z refactoryzacją!* 🚀
