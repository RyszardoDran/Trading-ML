# TradingML.ModelPrediction

Prosty program konsolowy w C# do predykcji modelu XGBoost na danych świeczek XAU/USD.

## Czym jest?

Projekt zawiera konsolową aplikację .NET, która:
- **Wczytuje model ML** (XGBoost) z pliku `sequence_xgb_model.pkl`
- **Akceptuje 1000+ świeczek** w formacie OHLCV (Open, High, Low, Close, Volume)
- **Oblicza featuresy** zgodnie z konfiguracją modelu
- **Generuje predykcję** - zwraca szansę na sygnał BUY oraz klasyfikację

## Struktura Projektu

```
TradingML.ModelPrediction/
├── Models/
│   ├── Candle.cs                 # Struktura świecy (OHLCV)
│   ├── PredictionResult.cs        # Wynik predykcji
│   └── ModelMetadata.cs           # Metadane modelu
│
├── Services/
│   ├── ModelLoader.cs             # Ładowanie modelu z plików
│   ├── CandleParser.cs            # Parser danych świeczek
│   ├── PredictionService.cs       # Serwis predykcji
│   └── ILogger.cs                 # Logger interfejs
│
├── Program.cs                     # Główny punkt wejścia
├── TradingML.ModelPrediction.csproj
└── README.md
```

## Wymagania

- **.NET 8.0** lub nowszy
- **Python 3.9+** (dla serwisu predykcji ML)
- Model XGBoost w `ml/outputs/models/`

## Instalacja

### 1. Klonuj lub wejdź do repozytorium
```bash
cd c:\Users\Arek\Documents\Repos\Traiding\Trading-ML\backend
```

### 2. Zbuduj projekt
```bash
dotnet build TradingML.ModelPrediction/TradingML.ModelPrediction.csproj
```

## Użycie

### Opcja 1: Używając wygenerowanych przykładowych świeczek

```bash
dotnet run --project TradingML.ModelPrediction -- --sample 1000
```

**Dane wyjściowe:**
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
```

### Opcja 2: Używając pliku CSV z rzeczywistymi danymi

Najpierw przygotuj plik CSV w formacie:
```csv
Timestamp,Open,High,Low,Close,Volume
2025-01-18 15:00:00,2000.50,2010.75,1995.25,2008.30,150000
2025-01-18 16:00:00,2008.30,2015.80,2003.10,2012.50,160000
...
```

Następnie uruchom:
```bash
dotnet run --project TradingML.ModelPrediction -- --candles-file data.csv --output result.json
```

## Dostępne argumenty

| Argument | Opis | Przykład |
|----------|------|---------|
| `--help`, `-h` | Wyświetla pomoc | `--help` |
| `--sample <count>` | Generuje N przykładowych świeczek | `--sample 1000` |
| `--candles-file <path>` | Wczytuje świece z pliku CSV | `--candles-file data.csv` |
| `--models-dir <path>` | Ścieżka do artefaktów modelu | `--models-dir ./models` |
| `--output <path>` | Zapisuje wynik do JSON | `--output result.json` |
| `--python <path>` | Ścieżka do Pythona | `--python C:\Python39\python.exe` |

## Dostępne Opcje

### Generowanie Próbek
```bash
# 1000 losowych świeczek
dotnet run --project TradingML.ModelPrediction -- --sample 1000

# Z zapisem wyniku
dotnet run --project TradingML.ModelPrediction -- --sample 1000 --output result.json
```

### Wczytywanie z Pliku
```bash
# Najprostszy przypadek
dotnet run --project TradingML.ModelPrediction -- --candles-file input.csv

# Ze custom ścieżką do modelu
dotnet run --project TradingML.ModelPrediction -- \
  --candles-file input.csv \
  --models-dir C:\custom\models \
  --output result.json
```

### Custom Konfiguracja
```bash
# Custom Python interpreter
dotnet run --project TradingML.ModelPrediction -- \
  --sample 1000 \
  --python "C:\Program Files\Python311\python.exe" \
  --output result.json
```

## Testy

Uruchom testy jednostkowe:

```bash
dotnet test TradingML.ModelPrediction.Tests/
```

Testowane komponenty:
- ✅ CandleParser - parsowanie i walidacja świeczek
- ✅ ModelLoader - ładowanie artefaktów modelu
- ✅ Argument parsing
- ✅ Validacja danych wejściowych

## Plik Wyjściowy JSON

Gdy używasz `--output`, otrzymasz plik JSON ze strukturą:

```json
{
  "signalType": "BUY",
  "probability": 0.753,
  "threshold": 0.63,
  "isSignal": true,
  "prediction": 1,
  "candlesUsed": 1000,
  "predictionTime": "2025-01-18T15:32:45.1234567Z",
  "modelWinRate": 0.85,
  "modelThreshold": 0.63,
  "firstCandleTime": "2025-01-17T15:32:45Z",
  "lastCandleTime": "2025-01-18T15:32:45Z"
}
```

## Architektura Modelu

Model XGBoost:
- **Okno czasowe:** 60 świeczek (sekwencja historyczna)
- **Liczba featurów:** ~900 (15 featurów × 60 świeczek)
- **Próg decyzji:** 0.63 (63% pewności = BUY signal)
- **Win Rate:** 85%
- **Minimalna liczba świeczek:** 260

## Cechy Świeczek (Features)

Model używa następujących zaawansowanych featurów technicznych:

### Timeframy
- **M5** - 5 minutowy interwał
- **M15** - 15 minutowy interwał  
- **M60** - 60 minutowy interwał

### Wskaźniki Techniczne
- **RSI** - Relative Strength Index
- **Bollinger Bands** - BB Position, BB Width
- **SMA** - Simple Moving Average (20, 200 dni)
- **MACD** - Moving Average Convergence Divergence
- **Stochastic** - Stoch K, Stoch D
- **ADX** - Average Directional Index
- **ATR** - Average True Range
- **Dystrybucja cen** - Dystans od SMA

## Integracja z ML

Program komunikuje się z Python'em poprzez:
1. **Zapis świeczek** → JSON
2. **Wywołanie Python'a** → `predict_single.py` w ml/scripts
3. **Odczyt wyników** → JSON output

## Rozszerzenia

### Przyszłe Ulepszenia
- [ ] Integracja z ONNX Runtime (szybsza predykcja bez Python'a)
- [ ] WebAPI do predykcji
- [ ] Streaming danych ze źródeł OANDA/Interactive Brokers
- [ ] Backtesting z różnymi progami
- [ ] Dashboard z wynikami predykcji
- [ ] Cache wyników dla powtarzających się danych

## Błędy i Rozwiązywanie

### Błąd: "Models directory not found"
```
Rozwiązanie: Sprawdź ścieżkę --models-dir. Domyślnie szuka:
c:\Users\Arek\Documents\Repos\Traiding\Trading-ML\ml\outputs\models
```

### Błąd: "Insufficient candles"
```
Rozwiązanie: Model wymaga minimum 260 świeczek. Podaj --sample 1000 lub więcej danych.
```

### Błąd: "Python prediction failed"
```
Rozwiązanie: Sprawdź:
1. Czy Python jest zainstalowany
2. Czy sciezka do python.exe jest poprawna (--python)
3. Czy istnieje plik ml/scripts/predict_single.py
```

## Kontakt i Wspieranie

Dla pytań lub błędów, sprawdź dokumentację w `docs/` lub `plans/`.

## Licencja

Capgemini 2025
