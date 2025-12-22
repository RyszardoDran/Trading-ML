# ML Specialist Agent - Documentation Guide

**Location**: `.github/instructions/ml-specialist-guide/`
**Purpose**: Documentation and guides for ML Specialist Agent (XAU/USD Sequence Model Expert)
**Status**: ✅ Production Ready

---

## 📚 Files in This Directory

### 1. **INTEGRATION-GUIDE.md** (Primary)
Comprehensive guide for understanding and using the ML Specialist Agent
- Overview of agent capabilities
- Integration checklist
- Agent knowledge base details
- Real-world usage scenarios
- Red flags agent will catch
- Maintenance and support

**Read when**: You're setting up the agent or need complete context
**Size**: ~460 lines
**Key sections**: Overview, capabilities, knowledge base, scenarios, red flags

### 2. **QUICK-REFERENCE.md** (Fast Start)
Quick reference card for rapid agent usage
- Quick start guide
- Agent capabilities overview
- Example questions
- Key mantras
- FAQ
- Setup instructions

**Read when**: You need quick answers or to remember key concepts
**Size**: ~236 lines
**Key sections**: Usage, capabilities, mantras, FAQ, support

---

## 🔗 Related Files

### Agent File (In `.github/agents/`)
- **ML-Specialist.agent.md** - Main agent file with complete knowledge base (~4000 lines)

### Instructions & Standards (In `.github/instructions/`)
- **python-ml.instructions.md** - ML project guidelines
- **copilot-instructions.md** - Development workflow
- **csharp.instructions.md** - C# backend standards

### Project Documentation (In `ml/`)
- **SEQUENCE_PIPELINE_README.md** - Pipeline orchestration details
- **START_HERE_REGIME_FILTER.md** - Trend filter implementation

---

## 📖 Reading Guide

### If you are:

**New to the project**:
1. Start with **QUICK-REFERENCE.md** (5 min read)
2. Then read **INTEGRATION-GUIDE.md** (10 min read)
3. Then open the agent in VS Code and ask: "Explain the sequence pipeline"

**Setting up the agent**:
1. Check **INTEGRATION-GUIDE.md** → "Integration Checklist" section
2. Verify files exist in `.github/agents/`
3. Reload VS Code and select ML Specialist Agent

**Already familiar with the project**:
1. Use **QUICK-REFERENCE.md** as a checklist
2. Refer to **INTEGRATION-GUIDE.md** when you need specific scenarios
3. Ask the agent directly for code help

**Debugging issues**:
1. Check **INTEGRATION-GUIDE.md** → "Red Flags" section
2. Cross-reference with **QUICK-REFERENCE.md** → "Key Mantras"
3. Ask agent: "Debuguj [your problem]"

---

## 🎯 Key Concepts

### Architecture
```
Agent File (.github/agents/)
    ↓ Contains
    - 4000+ lines of project knowledge
    - Persona: Senior ML Engineer (20+ years)
    - M1→M5→Features→Sequences→XGBoost flow
    - Critical rules (Fixed ATR, data leakage, split)
    - Code standards & best practices

Documentation Files (.github/instructions/ml-specialist-guide/)
    ↓ Provides
    - Integration guidance
    - Quick reference
    - Scenarios & workflows
    - Support & troubleshooting
    ↓ References
    - Project documentation (ml/)
    - Python/ML standards
    - Development workflow
```

### Red Flags (Agent Will Warn About)
- 🚨 Data leakage (using future data)
- 🚨 Changing fixed ATR multipliers
- 🚨 Random CV splits on time series
- 🚨 Missing type hints on public functions
- 🚨 Suspiciously high win rates (> 80%)

### Key Mantras
1. "Data leakage is a silent killer"
2. "Win rate > 80% = investigate"
3. "Fixed ATR multipliers - don't touch"
4. "Chronological always"
5. "Collaborative approach"

---

## ✅ Quick Checklist

- [ ] Agent file exists: `.github/agents/ML-Specialist.agent.md`
- [ ] Documentation is in `.github/instructions/ml-specialist-guide/`
- [ ] VS Code has latest version (reload if needed)
- [ ] Can find agent via "Agents: Select Agent"
- [ ] Understand M1→M5 architecture
- [ ] Know the 5 critical rules
- [ ] Read at least QUICK-REFERENCE.md

---

## 🚀 Next Steps

1. Read **QUICK-REFERENCE.md** (quick overview)
2. Read **INTEGRATION-GUIDE.md** (complete understanding)
3. Open VS Code → Agents: Select Agent → ML Specialist
4. Ask: "Explain how the sequence pipeline works"
5. Start contributing to ml/src/

---

## 🤝 Support

**Questions about:**
- XAU/USD sequence model → Ask ML Specialist Agent
- General Python/ML → Ask Python.agent
- Backend (C#/.NET) → Check csharp.instructions.md
- Development workflow → Check copilot-instructions.md

**Report issues**:
1. If agent gives wrong info → Correct it (agent learns)
2. If agent isn't available → Reload VS Code
3. If need clarification → Ask directly in agent chat

---

## 📝 Updates & Maintenance

**This documentation**:
- Last updated: December 22, 2025
- Version: 2.0 (Updated - collaborative approach)
- Status: ✅ Production Ready

**Agent file (ML-Specialist.agent.md)**:
- Personality: Senior ML Engineer with 20+ years experience
- Approach: Collaborative, practical, no enforcement
- Testing: Optional (agent asks at end, doesn't mandate)
- Code review: Suggestive, not enforcing

**Improvements welcome!**
- Add more scenarios if you discover new ones
- Link to additional documentation
- Document lessons learned
- Improve examples with real code

---

**Created**: December 22, 2025
**Maintained by**: ML Team
**Timezone**: UTC+1 (Europe/Warsaw)

<!-- © Capgemini 2025 - ML Specialist Agent Documentation -->
