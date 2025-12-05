# Implementation Plan: XAU/USD Day Trading Signal System

**Document Version**: 1.0  
**Date Created**: December 2, 2025  
**Phase**: Phase 1 (MVP - Informational Signals Only)  
**Project Duration**: ~8-12 weeks (estimated)  

---

## 1. Project Overview

### 1.1 Objectives

- Develop an intelligent trading signal generator for XAU/USD (5-minute candles)
- Deploy ML model trained on 1-year historical data to predict high-probability entries
- Create user-friendly in-app interface for signal display and RR management
- Build analytics pipeline for model improvement and performance tracking
- Achieve Phase 1 MVP with informational signals (no auto-execution)

### 1.2 Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Signal Accuracy** | P(TP) ≥ 70% | Signals generated only when this threshold is met |
| **System Uptime** | 99.5% | During trading hours 07:00-23:00 Polish time |
| **Signal Latency** | < 500ms | From candle close to signal generation |
| **Data Accuracy** | 99.9% | Historical data and real-time feeds |
| **Analytics Coverage** | 95%+ | Position outcomes captured for retraining |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Web/Mobile)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Signal Display UI | RR/SL/TP Info | Alert Notifications │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API / WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│                    BACKEND SERVICES (C#/.NET)                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Trading Signal Engine          [Requirement #1]        │   │
│  │  - ML Model Inference           [Python/.NET interop]   │   │
│  │  - Signal Validation Logic                              │   │
│  │  - P(TP) Calculation & Forecasting                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Notification Service           [Requirement #2]        │   │
│  │  - In-App Prompt Management                             │   │
│  │  - Signal Expiration Logic (5-min window)               │   │
│  │  - Real-time Updates via WebSocket                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Display & Risk Management      [Requirement #3]        │   │
│  │  - SL/TP Level Calculation                              │   │
│  │  - RR Ratio Validation (1:2)                            │   │
│  │  - Entry Info Display                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Analytics & Monitoring Engine  [Requirement #4]        │   │
│  │  - Position Outcome Tracking                            │   │
│  │  - Data Collection for ML Retraining                    │   │
│  │  - Performance Metrics Aggregation                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────┬──────────────┬──────────────┬──────────────────────┘
             │              │              │
    ┌────────▼──────┐ ┌────▼──────────┐ ┌─▼────────────────┐
    │ Market Data   │ │ ML Model      │ │ Analytics &     │
    │ Service       │ │ Service       │ │ Logging Service │
    │               │ │               │ │                 │
    │ - Real-time   │ │ - Batch       │ │ - Position      │
    │   Feeds       │ │   Training    │ │   Outcomes      │
    │ - Historical  │ │ - Inference   │ │ - Performance   │
    │   Data        │ │ - Model       │ │   Metrics       │
    │               │ │   Versioning  │ │ - Audit Logs    │
    └────────┬──────┘ └────┬──────────┘ └─┬───────────────┘
             │             │              │
    ┌────────▼─────────────▼──────────────▼─────────────────┐
    │          DATABASE LAYER (SQL Server / PostgreSQL)     │
    │  ┌──────────────────────────────────────────────────┐ │
    │  │ Historical Prices | Signals | Position Outcomes │ │
    │  │ Model Metadata | Performance Metrics | Audit Log │ │
    │  └──────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Component | Responsibility | Technology | Owner |
|-----------|---|---|---|
| **Trading Signal Engine** | Generate signals based on ML model | C#/.NET Core + Python (ML) | AI/ML Team |
| **Market Data Ingestion** | Fetch XAU/USD real-time & historical data | .NET + REST APIs | Data Team |
| **ML Model Service** | Train, version, deploy ML models | Python (scikit-learn, TensorFlow/PyTorch) | ML Team |
| **Notification Service** | Manage in-app prompts & signal lifecycle | C#/.NET Core + WebSocket | Backend Team |
| **Analytics Engine** | Track position outcomes & collect metrics | C#/.NET Core | Backend Team |
| **Frontend UI** | Display signals, SL/TP, user interactions | React/Vue.js + TypeScript | Frontend Team |
| **Database** | Persist historical data, signals, outcomes | SQL Server or PostgreSQL | DevOps/Data Team |

---

## 3. Technology Stack

### 3.1 Backend

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Language** | C# 11+ (.NET 7/8) | Type-safe, async/await support, cloud-ready |
| **Framework** | ASP.NET Core 7/8 | High-performance, RESTful APIs, dependency injection |
| **ML Integration** | Python with IronPython/.NET interop | Leverage scikit-learn, extensive ML libraries |
| **Database** | SQL Server / PostgreSQL | ACID compliance, time-series support |
| **Caching** | Redis | Real-time signal state, price cache |
| **Message Queue** | RabbitMQ / Azure Service Bus | Async signal generation, event streaming |
| **Testing** | xUnit, Moq, NUnit | Industry standard for .NET |
| **CI/CD** | Azure DevOps / GitHub Actions | Pipeline automation |

### 3.2 Frontend

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Framework** | React 18+ / Vue.js 3+ | Component-based, reactive UI |
| **Language** | TypeScript | Type safety, IDE support |
| **UI Library** | Material-UI / Tailwind CSS | Professional design, responsive |
| **State Mgmt** | Redux / Pinia | Global state for signal notifications |
| **Real-time** | Socket.IO / WebSocket | Live signal updates |
| **Charting** | TradingView Lightweight Charts | Professional trading interface |

### 3.3 Data & ML

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Source** | IB (Interactive Brokers) / OANDA / Finnhub | XAU/USD price feeds |
| **Data Storage** | SQL Server / InfluxDB | Time-series optimization |
| **ML Framework** | scikit-learn / XGBoost | Classification for high-prob signals |
| **Feature Engineering** | Pandas / NumPy | Technical indicators, momentum |
| **Model Training** | Jupyter Notebooks / Azure ML | Development & experimentation |
| **Model Deployment** | ONNX / ML.NET Model | Cross-platform inference |

---

## 4. Development Roadmap

### Phase 1: MVP (Weeks 1-12)

#### Week 1-2: Foundation & Setup
- [ ] Project structure setup (C#/.NET Core solution)
- [ ] Database schema design (price history, signals, outcomes)
- [ ] API contract definition (OpenAPI/Swagger)
- [ ] Development environment setup (Docker, VS Code, Git)
- [ ] Git branching strategy (main, develop, feature branches)

#### Week 3-4: Market Data Pipeline
- [ ] Real-time XAU/USD data ingestion from broker API
- [ ] Historical data download & preprocessing (1-year dataset)
- [ ] Data validation & quality checks
- [ ] Price cache layer (Redis)
- [ ] Unit tests for data pipeline (95%+ coverage)

#### Week 5-6: ML Model Development
- [ ] Feature engineering (technical indicators, volatility, trend)
- [ ] Dataset preparation & train/test split
- [ ] Model selection (Logistic Regression, Random Forest, XGBoost)
- [ ] Hyperparameter tuning & cross-validation
- [ ] Model performance evaluation (precision, recall, P(TP) calibration)
- [ ] Model serialization (ONNX)

#### Week 7-8: Trading Signal Engine (Backend)
- [ ] Signal generation algorithm implementation
- [ ] P(TP) probability calculation
- [ ] Entry level forecasting (trend-based)
- [ ] SL/TP level calculation (1:2 RR)
- [ ] Market hours validation (07:00-23:00 Polish time)
- [ ] Signal expiration logic (5-minute window)
- [ ] Error handling & logging

#### Week 9: Notification & Display Service
- [ ] In-app notification UI component
- [ ] WebSocket real-time updates
- [ ] Signal display formatting
- [ ] Expiration countdown
- [ ] Backend notification orchestration

#### Week 10: Analytics Engine
- [ ] Position outcome tracking database
- [ ] Data collection for closed positions
- [ ] Performance metrics aggregation
- [ ] Analytics dashboard (basic)

#### Week 11: Integration & QA
- [ ] End-to-end testing (signal generation → display)
- [ ] Performance testing (latency < 500ms)
- [ ] Load testing (100 concurrent users)
- [ ] Security review (data protection, API auth)
- [ ] UAT with internal trader

#### Week 12: Deployment & Documentation
- [ ] Production deployment (Azure / AWS / On-prem)
- [ ] Monitoring setup (Application Insights, ELK)
- [ ] User documentation
- [ ] Operations runbook

---

## 5. Detailed Component Specifications

### 5.1 Trading Signal Engine

**Purpose**: Generate high-probability trading signals

**Inputs**:
- Real-time 5-minute OHLC data
- Trained ML model
- Current time (market hours validation)

**Processing**:
1. Load latest 5-min candle
2. Extract features (technical indicators, volatility, trend)
3. Invoke ML model for P(TP) prediction
4. Forecast entry price (trend extrapolation)
5. Calculate transaction completion time
6. Validate all constraints (P≥70%, time>10min, before 23:00)
7. If valid → Generate signal; else → Discard

**Outputs**:
- Signal object: {direction, entry_price, entry_time_est, TP, SL, P(TP), expires_at}

**Technology**: C# with async/await, Python ML interop via IronPython or REST

---

### 5.2 ML Model Service

**Purpose**: Train and deploy ML models for signal probability

**Model Type**: Binary Classification (Will TP be achieved? Yes/No)

**Features** (Technical Indicators):
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands (upper/middle/lower, %B)
- ATR (Average True Range) - volatility
- Trend (SMA 20/50/200 positioning)
- Candle patterns (engulfing, doji, hammer)
- Volume analysis
- Time-of-day features

**Training Data**:
- 1 year of XAU/USD 5-minute candles
- Historical outcomes (TP/SL hit, outcome time)
- Split: 70% train, 15% validation, 15% test

**Model Selection**:
- Candidate 1: Logistic Regression (baseline, interpretable)
- Candidate 2: Random Forest (ensemble, feature importance)
- Candidate 3: XGBoost (state-of-art, gradient boosting)

**Evaluation Metrics**:
- Precision (minimize false positives)
- Recall (catch true positives)
- ROC-AUC (probability calibration)
- **Target**: Precision ≥ 70% for P(TP) ≥ 70% threshold

**Retraining Schedule**: Weekly batch job on Sunday evenings

---

### 5.3 Analytics Engine

**Purpose**: Track position outcomes & collect improvement data

**Data Collection**:
- Entry signal metadata (price, time, direction, P(TP))
- Position open/close events (time, price)
- Exit reason (TP/SL/manual)
- Actual outcome (win/loss, RR achieved)

**Outputs**:
- Performance dashboards
- Model accuracy metrics
- Signals performance vs. predictions
- Feature importance updates

**Technology**: C#/.NET with batch processing, scheduled jobs

---

## 6. Database Schema (High-Level)

```sql
-- Price History Table
CREATE TABLE PriceHistory (
    Id BIGINT PRIMARY KEY,
    Symbol NVARCHAR(10) NOT NULL,      -- XAU/USD
    Timestamp DATETIME2 NOT NULL,
    Open DECIMAL(10,5) NOT NULL,
    High DECIMAL(10,5) NOT NULL,
    Low DECIMAL(10,5) NOT NULL,
    Close DECIMAL(10,5) NOT NULL,
    Volume BIGINT,
    Timeframe NVARCHAR(10),            -- 5m
    INDEX IDX_Symbol_Timestamp (Symbol, Timestamp DESC)
);

-- Trading Signals Table
CREATE TABLE TradingSignals (
    Id UNIQUEIDENTIFIER PRIMARY KEY,
    GeneratedAt DATETIME2 NOT NULL,
    Direction NVARCHAR(10) NOT NULL,   -- LONG/SHORT
    EntryPrice DECIMAL(10,5) NOT NULL,
    EntryTimeEst NVARCHAR(50),         -- "1.5 minutes"
    StopLoss DECIMAL(10,5) NOT NULL,
    TakeProfit DECIMAL(10,5) NOT NULL,
    ProbabilityTP DECIMAL(5,2),        -- 0-100
    ExpiresAt DATETIME2 NOT NULL,
    Status NVARCHAR(20),               -- Active/Expired/Cancelled
    INDEX IDX_GeneratedAt_Status (GeneratedAt DESC, Status)
);

-- Position Outcomes Table
CREATE TABLE PositionOutcomes (
    Id UNIQUEIDENTIFIER PRIMARY KEY,
    SignalId UNIQUEIDENTIFIER,
    EntryPrice DECIMAL(10,5),
    EntryTime DATETIME2,
    ExitPrice DECIMAL(10,5),
    ExitTime DATETIME2,
    ExitReason NVARCHAR(50),           -- TP/SL/Manual
    PnL DECIMAL(10,5),
    RRAchieved DECIMAL(5,2),
    TraderValidation NVARCHAR(10),     -- Yes/No (manual feedback)
    FOREIGN KEY (SignalId) REFERENCES TradingSignals(Id)
);

-- Model Metadata Table
CREATE TABLE ModelMetadata (
    Id INT PRIMARY KEY,
    Version NVARCHAR(20),
    TrainedAt DATETIME2,
    Accuracy DECIMAL(5,4),
    Precision DECIMAL(5,4),
    Recall DECIMAL(5,4),
    RocAuc DECIMAL(5,4),
    Features INT,
    IsActive BIT
);
```

---

## 7. API Endpoints (REST)

### Signal API
```
POST   /api/v1/signals/generate              - Manually trigger signal generation
GET    /api/v1/signals/active                - Get active signals
GET    /api/v1/signals/{id}                  - Get signal details
DELETE /api/v1/signals/{id}                  - Cancel signal
```

### Analytics API
```
GET    /api/v1/analytics/performance         - Performance metrics
GET    /api/v1/analytics/outcomes            - Position outcomes
GET    /api/v1/analytics/model-accuracy      - Model accuracy over time
```

### Market Data API
```
GET    /api/v1/market/price/current          - Current XAU/USD price
GET    /api/v1/market/price/history          - Historical prices (OHLC)
GET    /api/v1/market/status                 - Market open/closed status
```

### WebSocket Events (Real-time)
```
signal.generated    - New signal available
signal.expired      - Signal expired
signal.cancelled    - Signal cancelled (P(TP) dropped)
market.tick         - New price tick
```

---

## 8. Testing Strategy

### Unit Testing
- Signal generation logic (all constraints)
- SL/TP calculations
- P(TP) probability validation
- Market hours validation
- **Target**: 95%+ code coverage

### Integration Testing
- Data pipeline → Signal generation → Display
- Market data ingestion
- Database persistence
- ML model inference

### Performance Testing
- Signal generation latency < 500ms
- API response time < 200ms
- WebSocket message latency < 100ms
- Load: 100 concurrent users

### UAT Testing
- Real trader feedback on signal quality
- Entry timing accuracy
- Risk management effectiveness

---

## 9. Deployment Strategy

### Infrastructure
- **Compute**: Azure App Service / AWS ECS (containerized)
- **Database**: Azure SQL / AWS RDS (PostgreSQL)
- **Cache**: Azure Cache for Redis / AWS ElastiCache
- **Messaging**: Azure Service Bus / RabbitMQ
- **Monitoring**: Application Insights / CloudWatch / ELK Stack

### Containerization
```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY . .
EXPOSE 80
CMD ["dotnet", "TradingSignalApi.dll"]
```

### CI/CD Pipeline
1. **Commit** → Trigger build
2. **Build** → Compile, unit tests, code quality
3. **Test** → Integration, performance tests
4. **Package** → Docker image, artifacts
5. **Deploy** → Dev → Staging → Production
6. **Monitor** → Logs, metrics, alerts

---

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **ML Model Accuracy < 70%** | Core functionality fails | Early model validation, baseline comparison |
| **Data Feed Downtime** | No signals generated | Fallback data source, alert on unavailability |
| **Real-time Latency** | Missed entry opportunities | Optimize inference, use caching, CDN |
| **Trader Misinterprets Signal** | Poor trading decisions | Clear UI, documentation, onboarding |
| **Database Failure** | Data loss | Replication, backups, disaster recovery plan |
| **Security Breach** | Unauthorized access | Authentication, encryption, audit logs |

---

## 11. Success Checklist (Phase 1 Complete)

- [ ] Signals generated with P(TP) ≥ 70% accuracy achieved
- [ ] System uptime 99.5% during trading hours
- [ ] Signal latency < 500ms consistently
- [ ] In-app UI displays signals with correct formatting
- [ ] SL/TP calculated correctly (1:2 RR)
- [ ] Signals expire after 5 minutes (estimated_entry_time + 5min)
- [ ] Analytics captures 95%+ of position outcomes
- [ ] Documentation complete & user onboarding ready
- [ ] Internal testing by trader(s) passed
- [ ] Security review approved
- [ ] Production deployment stable for 1+ week

---

## 12. Future Phases (Phase 2+)

### Phase 2: Enhancement & Automation
- Automatic trade execution via broker API
- Multi-asset support (EUR/USD, BTC/USD, SPX, etc.)
- Advanced risk management (dynamic position sizing)
- User authentication & account management

### Phase 3: Advanced Analytics
- Real-time dashboard & equity curves
- Backtesting engine for strategy optimization
- Feature importance visualization
- Model confidence intervals

### Phase 4: Scaling & Distribution
- Multi-currency strategies
- Portfolio optimization across assets
- Deployment to cloud with auto-scaling
- Mobile app (iOS/Android)

---

© Capgemini 2025 | XAU/USD Trading System Project
