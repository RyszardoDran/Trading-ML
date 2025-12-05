# EARS Specification: XAU/USD Day Trading System

**Document Version**: 1.0  
**Date Created**: December 2, 2025  
**Author**: Trading System Requirements Team  
**Asset**: XAU/USD (Gold)  
**Timeframe**: 5-minute candles  
**Trading Hours**: 07:00 - 23:00 Polish Time  

---

## 1. Executive Summary

This document specifies the functional and behavioral requirements for an **XAU/USD Day Trading Signal Generator** using EARS (Easy Approach to Requirements Syntax) notation. The system analyzes historical data using Machine Learning to generate probabilistic trading signals for intraday traders.

**Key Objectives:**
- Provide traders with high-probability entry signals (P(TP) ≥ 70%)
- Ensure all trades have minimum 1:2 Risk:Reward ratio with >10 minute hold time
- Display informational guidance only (no automatic execution in Phase 1)
- Monitor closed positions for ML model improvement

---

## 2. System Context & Constraints

| Aspect | Details |
|--------|---------|
| **Asset Class** | XAU/USD (Spot Gold) |
| **Trading Hours** | 07:00 - 23:00 Polish Time (UTC+1/+2) |
| **Candle Period** | 5 minutes |
| **Minimum Hold Time** | > 10 minutes to TP |
| **Minimum Success Rate** | P(TP) ≥ 70% |
| **Risk:Reward Ratio** | 1:2 (Fixed) |
| **Execution Model** | Informational only (Phase 1) |
| **ML Training Data** | 1 year of historical XAU/USD data |
| **Model Update Frequency** | Batch (scheduled daily/weekly) |
| **Notification Channel** | In-app prompt display |

---

## 3. Core Requirements (EARS Format)

### **Requirement #1: Signal Generation & Probability Analysis**

#### Wanted Behavior

```markdown
While trading hours are between 07:00 and 23:00 Polish time, 
when a new 5-minute candle closes, the **Trading Signal Engine** shall:

1. Analyze XAU/USD price data using ML model trained on 1-year historical dataset
2. Calculate the probability P(TP) of achieving 1:2 RR target AND 
   estimate complete transaction time from entry to TP achievement
3. Verify that transaction can be completed before market close (23:00 Polish time)
4. Generate and transmit entry signal only if ALL conditions are met:
   - P(TP) ≥ 70%
   - Estimated time to achieve TP > 10 minutes
   - Transaction completion time ≤ time remaining until 23:00
5. For qualifying signals, specify: 
   - Entry price level (forecasted on current trend, deliverable within ~2 minutes)
   - Direction (LONG or SHORT)
   - Estimated time until entry level is reached (concrete number, e.g., "1.5 minutes")
6. Discard all signals that do NOT meet above criteria 
   (no notification, no logging)
```

#### Unwanted Behaviors

**UB#1a: Signal Re-evaluation After P(TP) Drop**
```markdown
While a trading signal has been transmitted with P(TP) ≥ 70%, 
if the probability P(TP) drops below 70% before the trader enters the transaction, 
then the **Trading Signal Engine** shall **cancel the previously issued signal 
and notify the trader that the signal is no longer valid**
```

**UB#1b: Market Close Constraint Violation**
```markdown
While trading hours are approaching 23:00 Polish time, 
if the system calculates that a potential transaction cannot be completed before market close, 
then the **Trading Signal Engine** shall **suppress the signal and discard it without notification**
```

**UB#1c: Data Unavailability**
```markdown
While a 5-minute candle closes, 
if the system cannot retrieve complete historical or real-time price data, 
then the **Trading Signal Engine** shall **refrain from generating a signal and log the data failure event internally**
```

**UB#1d: Calculation Error**
```markdown
While calculating P(TP) and transaction duration, 
if the ML model fails or returns invalid results, 
then the **Trading Signal Engine** shall **discard the calculation and NOT transmit a signal, 
logging the error internally for debugging**
```

---

### **Requirement #2: Signal Notification to Trader**

#### Wanted Behavior

```markdown
When the **Trading Signal Engine** generates a valid entry signal 
that meets all criteria (P(TP) ≥ 70%, time to TP > 10 minutes, 
and completion before 23:00 Polish time), 
the **Notification System** shall **display an in-app prompt containing:**
  - Entry direction: LONG or SHORT
  - Entry price level (e.g., "2050.50 USD")
  - Estimated time until entry level is reached (e.g., "1.5 minutes")
  - Probability of achieving 1:2 RR target (e.g., "P(TP): 72%")

The signal shall **remain active for (estimated_entry_time + 5 minutes)**.
After this window expires, the signal shall be automatically dismissed 
and marked as expired.
```

#### Unwanted Behaviors

**UB#2a: Signal Expiration After Timeout**
```markdown
While a trading signal is displayed in the in-app prompt,
if the duration (estimated_entry_time + 5 minutes) has elapsed,
then the **Notification System** shall **automatically dismiss the prompt and 
prevent the trader from acting on the expired signal**
```

**UB#2b: P(TP) Drops During Signal Active Period**
```markdown
While a trading signal is active in the in-app prompt,
if the probability P(TP) drops below 70% before the signal expires,
then the **Notification System** shall **immediately remove the signal from display 
and show a cancellation notification to the trader**
```

---

### **Requirement #3: Entry Information & Risk Management Display**

#### Wanted Behavior

```markdown
When the trader views a valid trading signal in the in-app prompt,
the **Signal Display System** shall **calculate and display:**
  - Stop Loss (SL) price level based on entry price and 1:2 RR ratio
  - Take Profit (TP) price level based on entry price and 1:2 RR ratio
  - Risk-Reward summary (e.g., "SL: 2048.50 USD | TP: 2052.50 USD | RR: 1:2")

The signal shall remain **informational only** and **not execute any trades automatically** 
in this phase of development. The trader must manually execute the trade via their broker.
```

**SL/TP Calculation Example:**
- If entry price = 2050.50 USD and risk distance = 2.00 USD
- Then SL = 2048.50 USD (entry - risk)
- Then TP = 2054.50 USD (entry + 2×risk)

#### Unwanted Behaviors

**UB#3a: Incorrect SL/TP Calculation**
```markdown
While calculating Stop Loss and Take Profit levels based on entry price and 1:2 RR ratio,
if the calculation produces invalid or NaN values,
then the **Signal Display System** shall **refrain from displaying SL/TP values and 
show an error message to the trader instead**
```

**UB#3b: Signal Executed Without Confirmation**
```markdown
While a trading signal is displayed in the in-app prompt,
if the trader does NOT explicitly click an action button or acknowledge the signal,
then the **Signal Display System** shall **NOT attempt to execute any trade automatically**
```

---

### **Requirement #4: Position Monitoring for Analytics**

#### Wanted Behavior

```markdown
While a position has been opened by the trader following a system signal,
the **Analytics Engine** shall **continuously monitor the position's price movement 
and track whether it achieved SL, TP, or closed at breakeven**

For each closed position, the **Analytics Engine** shall **log:**
  - Entry price and time
  - Exit price and reason (SL hit, TP achieved, manual close)
  - Actual P&L vs. predicted RR ratio
  - Actual outcome vs. predicted P(TP) probability

This data shall be used for **ML model retraining and system performance analysis only**.
No real-time notifications or interventions shall occur during position monitoring.
```

#### Unwanted Behaviors

**UB#4a: Missing Position Data**
```markdown
While the Analytics Engine is monitoring a closed position,
if critical data points (entry time, exit time, price levels) are unavailable or corrupted,
then the **Analytics Engine** shall **discard the incomplete record and log the data loss 
event internally without impacting trader notifications**
```

**UB#4b: Attempted Real-Time Intervention**
```markdown
While monitoring an active position,
if a price movement suggests the position will miss the TP target,
then the **Analytics Engine** shall **refrain from sending alerts or modifying the position, 
and only log the outcome for post-trade analysis**
```

---

## 4. Future Considerations (Phase 2+)

The following features are identified for future development and are **NOT included in Phase 1**:

- ✅ **Automatic Trade Execution**: Direct API integration with broker for one-click execution
- ✅ **Multi-Asset Support**: Expand beyond XAU/USD to other FX pairs, crypto, indices
- ✅ **Advanced Risk Management**: Dynamic position sizing, portfolio correlation analysis
- ✅ **User Authentication & Account Management**: Multi-user support, API keys management
- ✅ **Real-time Dashboard**: Performance metrics, equity curve, drawdown tracking
- ✅ **Backtesting Engine**: Historical simulation and strategy optimization
- ✅ **Alert Escalation**: Email/SMS/Webhook notifications in addition to in-app
- ✅ **Model Explainability**: Feature importance, decision trees, model confidence scores

---

## 5. Acceptance Criteria

The system shall be considered ready for Phase 1 release when:

- [ ] Trading Signal Engine generates signals meeting all Requirement #1 specifications
- [ ] All signals meet P(TP) ≥ 70% and minimum 10-minute hold time criteria
- [ ] Notification System displays signals in correct format with 5-minute expiration
- [ ] SL/TP levels calculated correctly with 1:2 RR ratio and displayed accurately
- [ ] No automatic trades are executed (informational only)
- [ ] Analytics Engine successfully logs 95%+ of closed position data
- [ ] System operates correctly during all trading hours (07:00-23:00 Polish time)
- [ ] ML model trained on minimum 1 year of historical XAU/USD data
- [ ] Error handling prevents system crashes on data unavailability or calculation errors

---

## 6. Glossary

| Term | Definition |
|------|------------|
| **P(TP)** | Probability of achieving Take Profit target, expressed as percentage (0-100%) |
| **RR (Risk:Reward)** | Ratio of potential loss to potential gain (1:2 means 1 unit risk for 2 unit gain) |
| **SL (Stop Loss)** | Price level at which position closes automatically to limit losses |
| **TP (Take Profit)** | Price level at which position closes automatically to capture gains |
| **Entry Signal** | System recommendation to open a LONG or SHORT position at specific price |
| **Candle** | OHLC (Open, High, Low, Close) price data for specified timeframe (5 minutes) |
| **XAU/USD** | Gold futures contract quoted in US Dollars |
| **Batch Training** | ML model update process running on scheduled interval (not real-time) |

---

## 7. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-02 | Trading Requirements Team | Initial EARS specification |

---

© Capgemini 2025 | Trading System Project
