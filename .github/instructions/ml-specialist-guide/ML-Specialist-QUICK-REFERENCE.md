# ML Specialist Agent - Quick Reference

**Data**: 22 grudnia 2025 | **Project**: ml/src (XAU/USD Sequence Model)

## Szybki Start

### 🎯 Jak używać ML Specialist Agent?

1. **W VS Code**: Otwórz Command Palette (`Ctrl+Shift+P`)
2. **Wyszukaj**: "Agents: Select Agent"
3. **Wybierz**: "ML Specialist Agent - XAU/USD Sequence Model Expert"
4. **Pytaj** o coś z ml/src projektu!

### 📍 Co potrafi ML Specialist?

```
✅ Wyjaśniać każdy fragment kodu z ml/src
✅ Debugować data leakage issues
✅ Pomagać z feature engineering
✅ Optymalizować threshold
✅ Pisać production-ready code
✅ Tworzyć testy dla krytycznych ścieżek
✅ Wyjaśniać architekturę M5 vs M1
✅ Monitorować production metrics
✅ Wdrażać best practices ML
```

### 🚀 Przykładowe Pytania

```
"Pokaż mi jak works target creation w sequence pipeline"
"Czemu win rate > 80% to podejrzane?"
"Jak debugować data leakage?"
"Dodaj nowy feature do engineer.py"
"Wyjaśnij walk-forward validation"
"Co to jest regime filter?"
"Pokaż mi test coverage dla targets module"
```

---

## 📂 Struktura Pliku Agenta

Plik agenta: `.github/agents/ML-Specialist.agent.md`

### Sekcje w Pliku

1. **Header** - Metadata (description, tools, permissions)
2. **Persona** - Kim jest agent (Senior ML Engineer z 20+ latami)
3. **Project Context** - Pełny opis XAU/USD sequence model
4. **Architecture** - M1→M5→Features→Sequences→XGBoost flow
5. **Critical Rules** - Never change ATR multipliers, etc.
6. **Data Leakage** - How to avoid common mistakes
7. **Testing Standards** - Life-critical code requirements
8. **Code Style** - Type hints, docstrings, logging
9. **ML Best Practices** - Feature engineering, validation, thresholds
10. **Workflows** - How-to scenarios (add feature, optimize, debug)
11. **Production** - Deployment, monitoring, drift detection
12. **Knowledge Base** - XAU/USD dynamics, common pitfalls

---

## 🔗 Powiązane Pliki

Agent wykorzystuje wiedze z:

| Plik | Rola | Czytaj gdy... |
|------|------|---------------|
| [Python.agent.md](./Python.agent.md) | Senior Python ML Engineer foundation | Potrzebujesz ogólnych best practices |
| [python-ml.instructions.md](../instructions/python-ml.instructions.md) | ML project guidelines | Budujesz ML pipeline |
| [copilot-instructions.md](../copilot-instructions.md) | Development workflow | Commitujesz kod lub tworzysz PR |
| [SEQUENCE_PIPELINE_README.md](../../ml/SEQUENCE_PIPELINE_README.md) | Pipeline documentation | Chcesz wiedzieć jak trenować model |
| [START_HERE_REGIME_FILTER.md](../../ml/START_HERE_REGIME_FILTER.md) | Regime filter guide | Implementujesz trend filtering |

---

## 🎓 Różnice: Python.agent vs ML-Specialist.agent

| Aspekt | Python Agent | ML Specialist |
|--------|--------------|---------------|
| **Fokus** | General Python + ML | XAU/USD sequence model |
| **Wiedza** | Best practices | Project architecture (deep) |
| **Scenariusze** | Add types, tests, docs | Add features, optimize threshold, debug leakage |
| **Code Context** | Generic examples | Real code z ml/src/ |
| **Constraints** | Standard coding standards | Fixed ATR, no data leakage, M5 timeframe |
| **Monitoring** | General principles | Win rate, drift detection, decay |

**Wybór**:
- 🐍 **Python Agent**: Gdy pracujesz poza `ml/src` lub chcesz generalnych rad
- 🤖 **ML Specialist Agent**: Gdy pracujesz nad XAU/USD modelą - znamy każdy szczegół!

---

## 📋 Checklist: Kiedy Wołać ML Specialist

- [ ] Dodajesz feature do `ml/src/features/`
- [ ] Zmieniam coś w pipeline orchestration
- [ ] Debuguję suspicyjnie wysoką win rate
- [ ] Piszę testy dla targets/features
- [ ] Optymalizuję threshold dla produkcji
- [ ] Wdrażam nowy validation approach
- [ ] Analizuję data drift w live system
- [ ] Chcę zrozumieć dlaczego coś tak działa

---

## 🛠️ Instalacja / Setup

**Nic do robienia!**

Agent jest już stworzony i gotowy do użytku. VS Code automatycznie:
1. ✅ Załaduje `.github/agents/ML-Specialist.agent.md`
2. ✅ Udostępni go w "Agents: Select Agent"
3. ✅ Zaindeksuje całą wiedzę z sekcji

**Jeśli nie widać agenta:**
1. Reload VS Code (`Ctrl+Shift+P` → "Developer: Reload Window")
2. Sprawdź czy `.github/agents/ML-Specialist.agent.md` istnieje
3. Spróbuj "Agents: Select Agent" znowu

---

## 📞 Jak Korzystać w Praktyce

### Scenariusz 1: Dodaj Feature
```
Ty:      "Chcę dodać nowy feature do engineer.py - volume_ratio"
Agent:   [Wyjaśnia where, explains logic, shows code]
Ty:      "Zrób to"
Agent:   [Pisze kod + testy, wyjaśnia gdzie dodać]
Ty:      "Run backtest?"
Agent:   [Pokazuje komendy, wyjaśnia metrics]
```

### Scenariusz 2: Debuguj Problem
```
Ty:      "Win rate 88% - podejrzanie wysoko"
Agent:   [Opisuje jak sprawdzić data leakage]
Agent:   [Pokazuje testy - czy forward-looking features?]
Agent:   [Pokazuje czy scaler fit był poprawny]
Agent:   [Sugeruje co sprawdzić]
```

### Scenariusz 3: Zrozum Kod
```
Ty:      "Wyjaśnij split_and_scale_stage"
Agent:   [Pokazuje kod real code z projektu]
Agent:   [Wyjaśnia każdy krok]
Agent:   [Pokazuje why chronological, why no leakage]
```

---

## 🎯 Key Mantras (Agent Powtarza)

Zapamiętaj te zasady (agent będzie je ci wciskać):

1. **"Data leakage is silent killer"**
   - Zawsze chronological split
   - Scaler fit TYLKO na training
   - Features używają TYLKO historii

2. **"Win rate > 80% = smells fishy"**
   - Investigate immediately
   - Backtest na out-of-sample
   - Check for leakage

3. **"Fixed ATR multipliers - don't touch!"**
   - 1.0 SL, 2.0 TP = ground truth
   - Changing = different strategy
   - Model musi nauczyć się z tymi parametrami

4. **"Chronological always"**
   - Train: older data
   - Val: middle data
   - Test: newest data
   - NEVER shuffle!

5. **"Test before change"**
   - Write failing tests FIRST
   - Implement code to pass
   - Verify on real data

---

## ❓ FAQ

**Q: Czy mogę pytać o backend/frontend?**
A: Nie - agent zna tylko `ml/src`. Na backend pytaj Python.agent, na frontend - frontend agent.

**Q: Co jeśli agent coś źle powie?**
A: Możesz go poprawić - zna kod i chętnie uczy się feedbacku. Albo pytaj "Pokaż mi source code" - agent pokażę real file.

**Q: Czy mogę zmienić ATR multipliers?**
A: Nie! Agent da warning. To "ground truth" strategii - zmiana = kompletnie inne targety.

**Q: Jak train model z custom parameterami?**
A: Pytaj agenta! Pokaże ci wszystkie opcje CLI (`--window-size`, `--use-hybrid-optimization`, etc)

**Q: Co robi regime filter?**
A: Filtruje trades - tylko Long'i na uptrend. Czytaj `START_HERE_REGIME_FILTER.md` lub pytaj agenta!

---

## 🚀 Next Steps

1. ✅ **Przeczytaj**: Agent ML-Specialist.agent.md sekcja "Project: XAU/USD..."
2. ✅ **Otwórz**: VS Code, Agents → ML Specialist
3. ✅ **Spróbuj**: Pytaj o strukturę ml/src
4. ✅ **Zrozum**: Architektura M1→M5→Features→Sequences
5. ✅ **Pracuj**: Dodaj feature, debupuj, deploy!

---

## 📞 Support

**Gdy agent się myli**:
1. Pokaż mu real code (`read_file` tool)
2. Wyjaśnij co jest nie tak
3. Agent się uczy i poprawia

**Gdy potrzebujesz ogólnych rad**:
- Backend/C# → Backend.agent.md
- Frontend/React → Frontend.agent.md
- General Python → Python.agent.md
- XAU/USD ML → ML-Specialist.agent.md (THIS!)

---

**Stworzono**: 22 grudnia 2025
**Agent**: ML Specialist - XAU/USD Sequence Model Expert
**Status**: ✅ Production Ready
**Timezone**: UTC+1 (Poland)

<!-- © Capgemini 2025 -->
