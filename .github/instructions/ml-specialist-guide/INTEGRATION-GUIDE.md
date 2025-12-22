# ML Specialist Agent - Integration Guide

**Status**: ✅ Ready to Deploy | **Date**: December 22, 2025

---

## 📋 Overview

Stworzyliśmy nowego specjalistycznego agenta dla projektu `ml/src`:

- **Plik agenta**: `.github/agents/ML-Specialist.agent.md`
- **Quick reference**: `.github/agents/ML-Specialist-QUICK-REFERENCE.md`
- **Typ**: Specialized domain expert (XAU/USD Trading ML)
- **Bazuje na**: Python.agent.md + python-ml.instructions.md + Project-specific knowledge

---

## 🎯 Czego uczy się agent

### 1. Architektura Projektu
- **M1→M5 Data Pipeline**: Jak agregujemy minutowe dane do 5-minutowych
- **57-Feature Engineering**: Każdy feature, jego rola, jak go obliczać
- **Sequence Building**: 100 consecutive M5 bars → 1 training sample
- **XGBoost Training**: Model architecture, probability calibration, threshold optimization
- **Complete Pipeline**: Data → Features → Targets → Sequences → Train → Deploy

### 2. Critical Rules (Nie Do Łamania)
- ✅ **Fixed ATR Multipliers**: 1.0 SL, 2.0 TP (ground truth!)
- ✅ **Chronological Split**: Zawsze train→val→test (czasowo!)
- ✅ **No Data Leakage**: Features używają tylko historii
- ✅ **Scaler on Training Only**: Fit na train, transform na test
- ✅ **M5 Timeframe**: Strategy operates on M5, granularity on M1
- ✅ **Production Monitoring**: Win rate, drift detection, decay alerts

### 3. Code Standards (Dla ml/src)
- Type hints na wszystkich public functions
- Comprehensive docstrings (praktycznie, nie over-engineered)
- Logging zamiast print()
- Tests dla każdej krytycznej ścieżki (opcjonalnie)
- No magic numbers - use constants/config

### 4. ML Best Practices
- Feature importance analysis
- Walk-forward validation (not random CV)
- Threshold optimization strategies (F1, EV, Hybrid)
- Reproducibility (fixed random seeds)
- Data quality checks (NaN, inf, outliers)
- Win rate > 80% = suspicious (investigate!)

### 5. Production Readiness
- Deployment checklist
- Live monitoring setup
- Drift detection
- Model decay detection
- Fallback strategies

---

## 📂 Files Touched by Agent

Agent ma dostęp i rozumie:

```
ml/
├── src/
│   ├── pipelines/
│   │   ├── sequence_training_pipeline.py ← Main orchestrator
│   │   ├── walk_forward_validation.py
│   │   └── sequence_split.py
│   ├── pipeline_stages.py ← 7 training stages
│   ├── pipeline_cli.py ← CLI arguments
│   ├── pipeline_config_extended.py ← Configuration
│   ├── features/
│   │   ├── engineer.py ← M5 feature engineering (57 features)
│   │   ├── engineer_m5.py ← M1→M5 aggregation
│   │   ├── indicators.py ← Technical indicators
│   │   ├── m5_context.py ← M5 context
│   │   └── time_features.py ← Time-based
│   ├── targets/ ← Target/label creation
│   ├── sequences/ ← Sequence config
│   ├── filters/
│   │   └── regime_filter.py ← Trend filtering
│   ├── training/ ← Model training + calibration
│   ├── backtesting/ ← Backtest framework
│   ├── data_loading/ ← CSV loading
│   └── utils/
│       ├── risk_config.py ← ATR multipliers (FIXED!)
│       └── sequence_training_config.py ← Defaults
└── scripts/
    ├── predict_sequence.py ← Inference
    └── train_sequence_model.py ← Training launcher

Also understands:
├── SEQUENCE_PIPELINE_README.md
├── START_HERE_REGIME_FILTER.md
└── .github/
    ├── instructions/python-ml.instructions.md
    ├── instructions/copilot-instructions.md
    └── agents/Python.agent.md (parent agent)
```

---

## 🚀 How to Use ML Specialist Agent

### Method 1: VS Code UI
```
1. Open Command Palette (Ctrl+Shift+P)
2. Type: "Agents: Select Agent"
3. Choose: "ML Specialist Agent - XAU/USD Sequence Model Expert"
4. Start chatting!
```

### Method 2: Chat Interface
```
Agent should appear in:
- VS Code Chat (bottom panel)
- Copilot Chat (@ML-Specialist)
- GitHub Copilot in IDE
```

### Example Conversations

**Scenario 1: Understanding Code**
```
User: "Explain how target creation works"
Agent: [Shows code from targets/, explains SL/TP simulation]
Agent: [Explains why fixed ATR multipliers are important]
Agent: [Shows how win rate is computed]
Agent: [Points to tests that validate logic]
```

**Scenario 2: Adding Feature**
```
User: "I want to add momentum_score to features"
Agent: [Asks clarifying questions about definition]
Agent: [Shows where to add in engineer.py]
Agent: [Implements feature + shows test approach]
Agent: [Runs pipeline, shows before/after metrics]
Agent: [Asks: "Want to add tests for this?"]
```

**Scenario 3: Debugging Issue**
```
User: "Win rate is 92% - seems too good"
Agent: [Suggests data leakage checklist]
Agent: [Shows how to verify features are historical only]
Agent: [Checks scaler fitting]
Agent: [Checks for forward-looking indicators]
Agent: [Runs walk-forward to verify on out-of-sample]
```

---

## 🔧 Agent Capabilities

### What It Can Do
✅ Read/understand any file in ml/src
✅ Explain architecture and design decisions
✅ Write production-ready code with tests
✅ Debug data leakage issues
✅ Help with feature engineering
✅ Optimize thresholds
✅ Design monitoring strategies
✅ Suggest improvements (collaborative, not enforcing)
✅ Generate test cases (if needed)
✅ Explain M1/M5 timeframe strategy

### What It Won't Do
❌ Suggest changing fixed ATR multipliers (will warn!)
❌ Allow random CV splits for time series (will object!)
❌ Ignore data leakage issues (will flag!)
❌ Use print() instead of logging (will refactor!)
❌ Enforce code review (collaborative approach only)

---

## 📊 Agent Knowledge Base

Agent was trained on:

### Project-Specific
- ✅ **Sequence Pipeline README** - How pipeline works
- ✅ **Regime Filter Guide** - Trend filtering implementation
- ✅ **Production Integration Guide** - Deployment procedures
- ✅ **Risk Config** - Fixed ATR multipliers, trade parameters
- ✅ **Architecture Docs** - M1→M5→Features→Model flow
- ✅ **Actual Source Code** - Real ml/src files (not pseudo-code!)

### ML/Python Standards
- ✅ **python-ml.instructions.md** - ML project guidelines
- ✅ **copilot-instructions.md** - Development workflow
- ✅ **Python.agent.md** - General Python best practices

### Critical Concepts
- ✅ Data leakage patterns (and how to prevent)
- ✅ Chronological time-series splitting
- ✅ Walk-forward validation
- ✅ Feature importance analysis
- ✅ Probability calibration for XGBoost
- ✅ Threshold optimization strategies
- ✅ Production monitoring patterns

---

## 🎓 Comparison Table

| Aspect | Python Agent | ML Specialist Agent |
|--------|--------------|-------------------|
| **Purpose** | General Python + ML | XAU/USD Sequence Model |
| **Project Knowledge** | Generic | Deep (ml/src specific) |
| **Code Examples** | Pseudo-code | Real code from repo |
| **Data Leakage** | General warning | Project-specific patterns |
| **Feature Engineering** | Generic indicators | 57 specific features |
| **Architecture** | ML pipelines | M1→M5→Features→XGBoost |
| **Testing** | Standard pytest | ml/src test patterns |
| **Production** | General principles | XAU/USD monitoring |
| **Who Uses?** | Anyone on team | ML engineers in ml/src |

---

## 📚 Documentation Structure

```
.github/
├── agents/
│   ├── ML-Specialist.agent.md ← MAIN AGENT FILE
│   │   ├── Persona (Senior ML Engineer)
│   │   ├── Project Context (XAU/USD model)
│   │   ├── Architecture (M1→M5→Features→XGBoost)
│   │   ├── Critical Rules (ATR, leakage, split)
│   │   ├── Code Standards (types, tests, docs)
│   │   ├── ML Best Practices (validation, monitoring)
│   │   ├── Workflows (how-tos)
│   │   └── Production Integration
│   │
│   ├── ML-Specialist-QUICK-REFERENCE.md ← QUICK START
│   │   ├── Quick start (how to use)
│   │   ├── Key mantras
│   │   ├── FAQ
│   │   └── Integration checklist
│   │
│   └── Python.agent.md (parent - general Python/ML)
│
├── instructions/
│   ├── ml-specialist-guide/ ← AGENT DOCUMENTATION
│   │   ├── INTEGRATION-GUIDE.md (this file)
│   │   ├── QUICK-START.md
│   │   └── ... (other guides)
│   │
│   ├── python-ml.instructions.md ← PROJECT ML STANDARDS
│   ├── copilot-instructions.md ← DEVELOPMENT WORKFLOW
│   └── ... (other tech stacks)
│
└── prompts/
    └── ... (prompt templates)

ml/
├── src/ (THE PROJECT)
│   ├── pipelines/
│   ├── features/
│   ├── targets/
│   ├── training/
│   └── ... (all modules)
│
└── *.md (Pipeline docs)
    ├── SEQUENCE_PIPELINE_README.md
    └── START_HERE_REGIME_FILTER.md
```

---

## ✅ Integration Checklist

- [x] **Agent file created**: `.github/agents/ML-Specialist.agent.md`
- [x] **Quick reference created**: `.github/agents/ML-Specialist-QUICK-REFERENCE.md`
- [x] **Integration guide created**: `.github/instructions/ml-specialist-guide/INTEGRATION-GUIDE.md`
- [x] **Agent has access to**:
  - [x] Python.agent.md (parent)
  - [x] python-ml.instructions.md
  - [x] copilot-instructions.md
  - [x] Project README files
  - [x] Source code patterns
  - [x] Architecture documentation

**To activate in VS Code**:
1. ✅ Files are in `.github/agents/` (automatically discovered)
2. ✅ Agent metadata is correct (description, tools)
3. ✅ Open "Agents: Select Agent" - should appear there
4. ✅ If not, reload VS Code (`Cmd+Shift+P` → "Reload Window")

---

## 🎯 Agent Interaction Patterns

### Pattern 1: Add New Feature
```python
User: "Add volatility_ratio feature"
Agent will:
1. Ask clarifying questions (how to compute?)
2. Show where to add in engineer.py
3. Implement feature (production-ready code)
4. Run tests if applicable
5. Run pipeline, show before/after metrics
6. Ask: "Want to add tests for this?"
```

### Pattern 2: Debug Data Leakage
```python
User: "Win rate suspiciously high (92%)"
Agent will:
1. Show leakage checklist
2. Verify features use only history
3. Check scaler fitting
4. Verify chronological split
5. Run walk-forward validation
```

### Pattern 3: Optimize Threshold
```python
User: "How to maximize EV?"
Agent will:
1. Explain 3 strategies (F1, EV, Hybrid)
2. Show CLI parameters
3. Run backtest with different thresholds
4. Compare win rate vs trades
5. Recommend best for risk profile
```

### Pattern 4: Deploy to Production
```python
User: "How to deploy model?"
Agent will:
1. Show training on full history
2. Backup model artifacts
3. Run walk-forward validation
4. Deploy with monitoring setup
5. Configure alerts
```

---

## 🔄 Agent Interaction Loop

```
User Question
    ↓
Agent receives context:
  - Project knowledge (ml/src architecture)
  - Code standards (types, tests, docs)
  - Critical rules (ATR, leakage, split)
  - Real code examples (actual source)
    ↓
Agent thinks like Senior ML Engineer:
  - "What are the risks here?"
  - "Did they consider data leakage?"
  - "Are there tests (optional)?"
  - "What would I do?"
    ↓
Agent responds:
  1. Clarify if needed
  2. Show real code examples
  3. Point to tests/docs
  4. Suggest best approach
  5. Offer to implement
  6. Ask about next steps
    ↓
User refines
    ↓
Agent implements/explains further
```

---

## 🚨 Red Flags Agent Will Catch

- ❌ **Data Leakage**: Using future data, scaler on test
- ❌ **ATR Changes**: Someone suggesting to change fixed multipliers
- ❌ **Random CV Split**: On time-series data
- ❌ **Type Hints**: Missing on public functions
- ❌ **No Docstrings**: Especially new functions
- ❌ **Magic Numbers**: Hardcoded thresholds
- ❌ **Print Statements**: Instead of logging
- ❌ **Win Rate > 80%**: Suspiciously high

---

## 📞 Support & Maintenance

**If agent gives wrong answer**:
1. Correct the agent ("Actually, it's...")
2. Agent learns from feedback
3. Or show real code: "Look at line X in Y.py"

**If agent isn't available**:
1. Check file exists: `.github/agents/ML-Specialist.agent.md` ✅
2. Reload VS Code
3. Check internet connection (VS Code checks server)
4. Fall back to Python.agent or search code manually

**To improve agent**:
1. Add more examples to `ML-Specialist.agent.md`
2. Link to project docs more
3. Add scenarios you frequently encounter
4. Document lessons learned

---

## 🎓 Key Mantras (Agent Will Repeat)

1. **"Data leakage is a silent killer"**
   - Always chronological
   - Scaler fit ONLY on training
   - Features historical only

2. **"Win rate > 80% = investigate"**
   - Check for leakage
   - Run walk-forward
   - Verify on out-of-sample

3. **"Fixed ATR multipliers - don't touch"**
   - 1.0 SL, 2.0 TP = ground truth
   - Changing = different strategy
   - Model learns to WIN with these

4. **"Practical, collaborative approach"**
   - Ask when in doubt
   - Tests optional (you decide)
   - Suggestions, not orders

5. **"Production-first mindset"**
   - Monitor from day 1
   - Drift detection ready
   - Decay alerts working

---

## 📋 Next Steps

1. ✅ **Read ML Specialist Agent**: `.github/agents/ML-Specialist.agent.md`
2. ✅ **Bookmark Quick Reference**: `.github/agents/ML-Specialist-QUICK-REFERENCE.md`
3. ✅ **Open in VS Code**: "Agents: Select Agent" → ML Specialist
4. ✅ **Ask first question**: "Explain sequence pipeline"
5. ✅ **Start working**: Collaborate with agent on features

---

## 📞 Questions?

If you have questions about:
- **XAU/USD sequence model**: Ask ML Specialist Agent
- **General Python/ML**: Ask Python Agent
- **Development workflow**: Check copilot-instructions.md
- **Project standards**: Check python-ml.instructions.md

---

**Created**: December 22, 2025
**Status**: ✅ Production Ready
**Version**: 2.0 (Updated - collaborative approach)
**Timezone**: UTC+1 (Europe/Warsaw)

<!-- © Capgemini 2025 - ML Specialist Integration -->
