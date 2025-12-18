# TradingML Backend - Model Prediction Console

## 📋 Opis

Backend część systemu handlowania XAU/USD zawierająca konsolową aplikację do predykcji modelu ML.

## 🎯 Cel

Prosty program konsolowy w C#, który:
- ✅ Wczytuje wytrenowany model XGBoost 
- ✅ Akceptuje 1000+ świeczek (OHLCV)
- ✅ Zwraca predykcję sygnału handlowego (BUY/SELL)
- ✅ Prosta i intuicyjna CLI
- ✅ JSON output z wynikami

## 📁 Struktura

```
backend/
├── TradingML.ModelPrediction/              # Main Console App
│   ├── Models/
│   │   ├── Candle.cs                      # OHLCV structure
│   │   ├── PredictionResult.cs            # Prediction output
│   │   └── ModelMetadata.cs               # Model configuration
│   │
│   ├── Services/
│   │   ├── ModelLoader.cs                 # Load ML artifacts
│   │   ├── CandleParser.cs                # Parse market data
│   │   ├── PredictionService.cs           # Run predictions
│   │   └── ILogger.cs                     # Logging
│   │
│   ├── Program.cs                         # Entry point
│   └── TradingML.ModelPrediction.csproj
│
├── TradingML.ModelPrediction.Tests/        # Unit tests
│   ├── CandleParserTests.cs
│   ├── ModelLoaderTests.cs
│   └── TradingML.ModelPrediction.Tests.csproj
│
├── TradingML.sln                          # Visual Studio Solution
├── README.md                              # Usage documentation
├── ARCHITECTURE.md                        # Technical design
└── .gitignore

```

## 🚀 Quick Start

### 1. Zbuduj projekt
```bash
cd backend
dotnet build
```

### 2. Uruchom z przykładowymi danymi
```bash
dotnet run --project TradingML.ModelPrediction -- --sample 1000
```

### 3. Z rzeczywistymi danymi
```bash
dotnet run --project TradingML.ModelPrediction -- \
  --candles-file data.csv \
  --output result.json
```

## 📊 Wyjście Programu

```
╔════════════════════════════════════════════════════════════════╗
║                    PREDICTION RESULTS                          ║
╚════════════════════════════════════════════════════════════════╝

  Signal Type:          BUY                               
  Probability:          75.30%                            
  Decision Threshold:   63.00%                            
  Prediction Class:     BUY                               
  Candles Used:         1000                              
  Prediction Time:      2025-01-18 15:32:45 UTC          
  Model Win Rate:       85.00%                            

╔════════════════════════════════════════════════════════════════╗
✓ SIGNAL CONFIRMED - Probability exceeds threshold
╚════════════════════════════════════════════════════════════════╝
```

## 📋 Argumenty CLI

| Argument | Opis |
|----------|------|
| `--help` | Pomoc |
| `--sample N` | Generuj N świeczek |
| `--candles-file PATH` | Wczytaj CSV |
| `--models-dir PATH` | Ścieżka do modelu |
| `--output PATH` | Zapisz JSON |
| `--python PATH` | Python executable |

## 🧪 Testy

```bash
dotnet test TradingML.ModelPrediction.Tests/
```

Pokrycie:
- ✅ CandleParser (parsing, validacja)
- ✅ ModelLoader (wczytywanie artefaktów)
- ✅ Argument parsing
- ✅ Edge cases

## 🔧 Wymagania

- **.NET 8.0+**
- **Python 3.9+** (dla ML inference)
- Model w `../ml/outputs/models/`

## 📚 Dokumentacja

- **[README.md](README.md)** - Szczegółowe instrukcje użycia
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architektура i design
- **[../ml/](../ml/)** - ML model i skrypty

## 🎓 Technologie

- **C# 11+** - Language
- **.NET 8.0** - Framework
- **xUnit** - Testing
- **JSON** - Data format
- **Python Subprocess** - ML inference

## 🔐 Bezpieczeństwo

✅ Input validation
✅ File checks
✅ Exception handling
✅ Temp file cleanup
✅ Process timeout (30s)

## 📝 Model Info

- **Type:** XGBoost Classifier
- **Input:** 900 features (15 × 60 candles)
- **Output:** BUY probability (0-1)
- **Threshold:** 0.63 (63%)
- **Win Rate:** 85%
- **Min Candles:** 260

## 🚧 Przyszłe Plany

- [ ] ONNX support (fast inference)
- [ ] WebAPI endpoint
- [ ] Database logging
- [ ] Real-time streaming
- [ ] Backtesting integration
- [ ] Dashboard

## 👥 Autor

Capgemini 2025

---

**Status:** ✅ Production Ready

Dla pytań: sprawdź dokumentację lub README.md
