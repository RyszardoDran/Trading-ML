# ML Specialist Agent - Documentation Manifest

**Purpose**: Complete inventory of documentation files | **Created**: December 22, 2025
**Location**: `.github/instructions/ml-specialist-guide/`

---

## 📦 Documentation Files in This Directory

### 1. 🎯 README.md (START HERE)
**Purpose**: Navigation guide for the documentation
**Size**: ~250 lines | **Read time**: 10 min

**Contains**:
- Overview of files in this directory
- Reading guide based on your role
- Key concepts summary
- Quick checklist
- Next steps

**Read when**: First time in this directory
**Key sections**: Files overview, reading guide, key concepts

---

### 2. ⚡ QUICK-REFERENCE.md (QUICK START)
**Purpose**: Fast reference card for agent usage
**Size**: ~230 lines | **Read time**: 5 min

**Contains**:
- Quick start in VS Code
- Agent capabilities checklist
- Example questions (copy-paste ready)
- Key mantras (5 core principles)
- FAQ with direct answers
- Setup instructions

**Read when**: You need quick answers or orientation
**Key sections**: Usage, capabilities, mantras, FAQ

---

### 3. 📖 INTEGRATION-GUIDE.md (COMPLETE REFERENCE)
**Purpose**: Comprehensive guide to agent capabilities and usage
**Size**: ~450 lines | **Read time**: 20 min

**Contains**:
- Overview of what agent knows
- Agent knowledge base summary
- Files touched by agent
- How to use agent (2 methods)
- Real-world scenarios (4 examples)
- Agent capabilities matrix
- Comparison with Python.agent
- Agent interaction loop
- Red flags agent will catch
- Troubleshooting guide

**Read when**: You're learning to use the agent effectively
**Key sections**: Overview, scenarios, red flags, support

---

## 🔗 Related Files (NOT in this directory)

### Agent File (In `.github/agents/`)
- **ML-Specialist.agent.md** - Main agent file with 4000+ lines of knowledge
  - Persona: Senior ML Engineer (20+ years)
  - Project overview, architecture, critical rules
  - Code standards, ML practices, workflows
  - Production integration, monitoring

### Instructions & Standards (In `.github/instructions/`)
- **python-ml.instructions.md** - ML project guidelines
- **copilot-instructions.md** - Development workflow
- **csharp.instructions.md** - C# backend standards
- **Other agent instructions** - Backend, Frontend, etc.

### Project Documentation (In `ml/`)
- **SEQUENCE_PIPELINE_README.md** - Pipeline orchestration
- **START_HERE_REGIME_FILTER.md** - Trend filter guide
- **Source code** - All modules in ml/src/

---

## 📊 File Reference Table

| File | Purpose | Size | Read Time | Audience |
|------|---------|------|-----------|----------|
| **README.md** | Navigation guide | 250 | 10 min | Everyone |
| **QUICK-REFERENCE.md** | Quick start | 230 | 5 min | Everyone |
| **INTEGRATION-GUIDE.md** | Complete reference | 450 | 20 min | New users |
| **ML-Specialist.agent.md** (in agents/) | Main knowledge | 4000+ | 2-3 hrs | Deep learners |

---

## 🎯 How to Use These Files

### First Time Using the Agent?
```
1. Read this README.md (10 min) ← You are here
2. Read QUICK-REFERENCE.md (5 min)
3. Open VS Code → Agents: Select Agent → ML Specialist
4. Ask your first question!
```

### Setting Up the Agent?
```
1. Verify `.github/agents/ML-Specialist.agent.md` exists
2. Read INTEGRATION-GUIDE.md → "Integration Checklist"
3. Reload VS Code if needed
4. Test: "Agents: Select Agent" should show ML Specialist
```

### Need Complete Understanding?
```
1. Read QUICK-REFERENCE.md (5 min)
2. Read this README.md (10 min)
3. Read INTEGRATION-GUIDE.md (20 min)
4. Then read ML-Specialist.agent.md sections as needed
5. Use QUICK-REFERENCE.md as ongoing reference
```

### Adding a Feature to ml/src/?
```
1. Read QUICK-REFERENCE.md → "Key Mantras"
2. Read INTEGRATION-GUIDE.md → "Scenarios" section
3. Ask agent: "I want to add [feature name]"
4. Follow agent's step-by-step guidance
5. Reference QUICK-REFERENCE.md for reminders
```

### Debugging Data Leakage?
```
1. Read INTEGRATION-GUIDE.md → "Red Flags"
2. Ask agent: "Check for data leakage in [location]"
3. Follow agent's verification steps
4. Reference QUICK-REFERENCE.md → "Key Mantras" #1
```

### Deploying to Production?
```
1. Read INTEGRATION-GUIDE.md → Look for "deploy" or "production"
2. Ask agent: "Show me production deployment steps"
3. Follow checklist provided by agent
4. Verify monitoring is set up
```

---

## 🚀 Quick Navigation Table

| Goal | Start with | Then ask agent |
|------|-----------|----------------|
| Learn to use agent | This README | "Explain sequence pipeline" |
| Prevent data leakage | QUICK-REFERENCE → Mantra #1 | "Check for leakage in..." |
| Add new feature | INTEGRATION-GUIDE → Scenarios | "I want to add..." |
| Understand architecture | QUICK-REFERENCE + Agent | "Explain M1→M5 flow" |
| Deploy model | INTEGRATION-GUIDE | "Show deployment steps" |
| Find quick answer | QUICK-REFERENCE → FAQ | (Already answered!) |
| Deep understanding | Read all files + Agent | (Time investment, worth it!) |

---

## 🎓 The Agent Approach

The ML Specialist Agent is **collaborative, practical, and non-enforcing**:

✅ **What it does**:
- Asks clarifying questions
- Shows real code examples
- Suggests best practices
- Offers to implement features
- Asks: "Want to add tests?" (at the end, not upfront)

❌ **What it doesn't do**:
- Enforce TDD or test-first
- Require code review approval
- Force specific patterns
- Mandate testing (suggestions only)

---

## 🛠️ Agent Capabilities at a Glance

**Can help with**:
- 🎯 Understanding any file in ml/src
- 🎯 Explaining architecture and design
- 🎯 Writing production-ready code
- 🎯 Debugging data leakage
- 🎯 Feature engineering
- 🎯 Threshold optimization
- 🎯 Monitoring strategies
- 🎯 Production deployment

**Will warn about**:
- ⚠️ Data leakage (using future data)
- ⚠️ Changing fixed ATR multipliers
- ⚠️ Random CV splits on time-series
- ⚠️ Missing type hints
- ⚠️ Suspiciously high win rates (>80%)

---

## ❓ Common Questions

**Q: Should I read all files?**
A: No! Start with QUICK-REFERENCE (5 min), then INTEGRATION-GUIDE (20 min). Read ML-Specialist.agent.md only if you need deep understanding.

**Q: Is the agent in `.github/agents/` or `.github/instructions/`?**
A: Agent file is in `.github/agents/ML-Specialist.agent.md`. These documentation files are in `.github/instructions/ml-specialist-guide/`.

**Q: Can I ask the agent anything?**
A: Yes! About `ml/src/` project specifically. For general Python/ML questions, use Python.agent instead.

**Q: What if agent gives wrong answer?**
A: Correct it! The agent learns from feedback. Or ask "Show me the source code" and point out the issue.

**Q: Do I have to write tests?**
A: No! The agent is collaborative. It suggests tests at the end - you decide if you want them.

---

## ✅ Verification Checklist

Before you start:

- [ ] Agent file exists: `.github/agents/ML-Specialist.agent.md`
- [ ] Documentation exists: This directory (`.github/instructions/ml-specialist-guide/`)
- [ ] VS Code is up to date
- [ ] Can reach "Agents: Select Agent" menu
- [ ] ML Specialist Agent appears in the list
- [ ] Understand: M1→M5 architecture
- [ ] Understand: Fixed ATR multipliers (don't change!)
- [ ] Understand: Chronological split (no random)

---

## 📞 Support & Resources

**When you need help**:
1. Check QUICK-REFERENCE.md FAQ section first
2. Read relevant INTEGRATION-GUIDE.md section
3. Ask agent directly in VS Code
4. Reference ML-Specialist.agent.md for deep knowledge

**Different questions?**:
- Backend (C#/.NET) → csharp.instructions.md
- General Python → python-ml.instructions.md
- General ML → Python.agent.md
- Development workflow → copilot-instructions.md

---

## 📋 Recommended Reading Path

### Quick Start (15 minutes)
1. ⏱️ QUICK-REFERENCE.md (5 min)
2. ⏱️ This README.md (10 min)
3. ✅ Ready to use the agent!

### Complete Understanding (45 minutes)
1. ⏱️ QUICK-REFERENCE.md (5 min)
2. ⏱️ This README.md (10 min)
3. ⏱️ INTEGRATION-GUIDE.md (20 min)
4. ⏱️ Open agent in VS Code (10 min to explore)

### Expert Level (2-3 hours)
1. All of above (45 min)
2. ML-Specialist.agent.md full read (2+ hours)
3. Collaborate with agent on real tasks
4. Reference files as needed

---

## 🎯 Key Takeaways

1. **Agent location**: `.github/agents/ML-Specialist.agent.md`
2. **Documentation location**: `.github/instructions/ml-specialist-guide/` (this folder)
3. **Quick start**: 15 minutes with QUICK-REFERENCE.md
4. **Deep learning**: Full agent file has all knowledge
5. **Approach**: Collaborative, practical, no enforcement
6. **Testing**: Optional - agent asks at end, doesn't mandate
7. **Critical rules**: Fixed ATR, chronological split, no leakage

---

## 🚀 Next Steps

1. ✅ You've read this file
2. 👉 **Next**: Read QUICK-REFERENCE.md (5 min)
3. 👉 **Then**: Open VS Code → Select ML Specialist Agent
4. 👉 **First question**: "Explain the sequence pipeline"
5. 👉 **Then**: Start collaborating on features!

---

**Status**: ✅ Complete
**Version**: 2.0 (Collaborative approach)
**Created**: December 22, 2025
**Timezone**: UTC+1 (Europe/Warsaw)

<!-- © Capgemini 2025 - ML Specialist Agent Documentation -->
