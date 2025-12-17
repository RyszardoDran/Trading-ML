# 🎯 UWAGA: Struktura Katalogów - Już Istnieje!

## Odkrycie

Sprawdziłem istniejącą strukturę projektu i okazało się, że **wiele katalogów już istnieje**!

### ✅ Co JUŻ Istnieje w `ml/src/`

```
ml/src/
├── analysis/            ✅ Istnieje
├── backtesting/         ✅ Istnieje
├── config/              ✅ Istnieje
├── data/                ✅ Istnieje
├── features/            ✅ Istnieje (PUSTY)
├── forecasting/         ✅ Istnieje
├── logs/                ✅ Istnieje
├── models/              ✅ Istnieje
├── notebooks/           ✅ Istnieje
├── pipelines/           ✅ Istnieje
├── scripts/             ✅ Istnieje
├── sequences/           ✅ Istnieje (PUSTY)
├── targets/             ✅ Istnieje (PUSTY)
└── utils/               ✅ Istnieje
```

### ✅ Co JUŻ Istnieje w `ml/src/pipelines/`

```
ml/src/pipelines/
├── sequence_training_pipeline.py       ✅ Główny plik (1740 linii)
├── training_pipeline.py                ✅ Inny pipeline
├── __init__.py                         ✅
├── INDEX.md                            ✅ (moje dokumenty)
├── REFACTOR_ETAP_1.md                  ✅ (moje dokumenty)
├── REFACTOR_PLAN.md                    ✅ (moje dokumenty)
└── ROADMAP.md                          ✅ (moje dokumenty)
```

---

## ⚠️ Co To Oznacza Dla Planu

### Mój Plan Zakładał
❌ Stwórzyć katalogi: `data_loading/`, `features/`, `targets/`, `sequences/`, `training/`, `utils/`

### Rzeczywistość
✅ Katalogi już istnieją (ale są **puste**)!

### Konsekwencja
🎯 **Plan Refactoryzacji Pozostaje Ważny**, ale muszę go zaadaptować:

1. **Etap 1 (ZMIENIĆ)**:
   - ❌ Nie tworzyć katalogów (już istnieją!)
   - ✅ Tworzyć pliki w istniejących katalogach
   - ✅ `ml/src/pipelines/data_loading/` → `ml/src/data_loading/`? (TRZEBA SPRAWDZIĆ)

2. **Etapy 2-7**: Plan pozostaje bez zmian

---

## 🤔 Pytania Do Potwierdzenia

**Gdzie powinny trafić moduły refactoryzacji?**

### Opcja A: Bezpośrednio w `ml/src/`
```
ml/src/
├── data_loading/
│   ├── __init__.py
│   ├── loaders.py
│   └── validators.py
├── features/
│   ├── __init__.py
│   ├── engineer.py
│   └── ...
└── ...
```

### Opcja B: W `ml/src/pipelines/`
```
ml/src/pipelines/
├── data_loading/
│   ├── __init__.py
│   ├── loaders.py
│   └── validators.py
├── features/
│   ├── __init__.py
│   ├── engineer.py
│   └── ...
└── ...
```

### Opcja C: Mieszane
```
ml/src/
├── features/           (istniejący, przenieść tutaj)
├── targets/            (istniejący, przenieść tutaj)
├── sequences/          (istniejący, przenieść tutaj)
└── pipelines/
    ├── data_loading/   (nowy moduł)
    ├── training/       (nowy moduł)
    └── sequence_training_pipeline.py
```

---

## 📋 Wymagane Działania

1. **Sprawdzić dokumentację projektu** - czy gdzieś napisane gdzie powinny trafić moduły?
2. **Pytanie do użytkownika** - gdzie refactoryzować?
3. **Zaktualizować plan** - aby był spójny z istniejącą strukturą

---

## 🚨 UWAGA DLA UŻYTKOWNIKA

Mój plan refactoryzacji założył stworzenie nowych katalogów, ale **projekt już ma większość z nich**!

**Pytania:**
1. Gdzie mają trafić moduły z refactoryzacji: `ml/src/` czy `ml/src/pipelines/`?
2. Czy istniejące katalogi `features/`, `targets/`, `sequences/` są zarezerwowane dla czegoś innego?
3. Czy `ml/src/pipelines/` to miejsce na logikę orchestracji (jak myślałem), czy na wszystko?

**Proszę potwierdzić strukturę docelową, aby zaktualizować plan!**

---

**Status**: ⏸️ Plan Oczekuje na Potwierdzenie Struktury
**Następny Krok**: Sprawdzenie gdzie refactoryzować
