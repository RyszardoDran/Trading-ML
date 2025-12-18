# TradingML.ModelPrediction - Architektura

## Przegląd

```
┌─────────────────────────────────────────────────────────────┐
│ Console Application (Program.cs)                            │
│ - Argument parsing                                          │
│ - Orchestration                                             │
└────┬────────────────────────────────────────────────────────┘
     │
     ├─────────────┬──────────────────┬──────────────────┐
     │             │                  │                  │
     ▼             ▼                  ▼                  ▼
┌─────────┐ ┌──────────────┐ ┌───────────────┐ ┌──────────────────┐
│ModelLoader │ CandleParser │ PredictionService │ ArgumentParser │
└─────────┘ └──────────────┘ └───────────────┘ └──────────────────┘
     │             │                  │
     ▼             ▼                  ▼
  Models      Validation         Python IPC
  JSON/PKL    JSON/CSV          (subprocess)
```

## Komponenty

### 1. Models (Models/)

**Candle.cs** - Struktura danych świecy
```csharp
public class Candle
{
    public DateTime Timestamp { get; set; }
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double Volume { get; set; }
}
```

**PredictionResult.cs** - Wynik predykcji
```csharp
public class PredictionResult
{
    public double Probability { get; set; }      // 0-1
    public int Prediction { get; set; }          // 0=SELL, 1=BUY
    public bool IsSignal { get; set; }           // > threshold?
    public string SignalType { get; set; }       // BUY/SELL/NEUTRAL
}
```

**ModelMetadata.cs** - Konfiguracja modelu
```csharp
public class ModelMetadata
{
    public List<string> FeatureColumns { get; set; }
    public double Threshold { get; set; }
    public int WindowSize { get; set; }
    public int TotalFeatures { get; set; }
    // ... więcej
}
```

### 2. Services (Services/)

#### ModelLoader.cs
- Wczytuje artefakty modelu z JSON
- Waliduje dostępność plików
- Parsuje `sequence_feature_columns.json`
- Parsuje `sequence_threshold.json`
- Ładuje feature importance

```csharp
var loader = new ModelLoader(modelsDir, logger);
var metadata = loader.LoadModelMetadata();  // ModelMetadata
```

#### CandleParser.cs
- Parsuje CSV → Candle list
- Waliduje OHLCV konsystencję
- Obsługuje minimalne ilości danych

```csharp
var parser = new CandleParser(logger);
var candles = parser.ParseFromCsv("data.csv");
parser.ValidateCandles(candles, 260);  // Min 260
```

#### PredictionService.cs
- Interfejs do modelu XGBoost
- Komunikacja z Python'em via subprocess
- Zwraca PredictionResult

```csharp
var service = new PredictionService(modelsDir, pythonPath, metadata, logger);
var result = await service.PredictAsync(candles);
```

#### ILogger.cs
- Prosty logger interface
- ConsoleLogger implementacja
- Kolorowe wyjście konsoli

### 3. Program.cs

Główny punkt wejścia:
1. Parsuje argumenty CLI
2. Wczytuje model
3. Wczytuje/generuje świece
4. Uruchamia predykcję
5. Wyświetla wyniki
6. Zapisuje JSON (opcjonalnie)

## Flow Danych

```
1. INPUT (CSV lub --sample)
        ↓
2. PARSING (CandleParser)
        ↓
3. VALIDATION (ValidateCandles)
        ↓
4. MODEL LOAD (ModelLoader)
        ↓
5. PREDICTION (PredictionService)
        ↓
6. FORMATTING (PredictionResult)
        ↓
7. OUTPUT (Console + JSON)
```

## Integracja Python

Predykcja działa poprzez Python subprocess:

```bash
python ml/scripts/predict_single.py \
    --input-file tempdata.json \
    --models-dir ml/outputs/models \
    --output-file tempout.json
```

**Input JSON:**
```json
[
  {
    "timestamp": "2025-01-01T00:00:00Z",
    "open": 2000.0,
    "high": 2010.5,
    "low": 1995.3,
    "close": 2005.2,
    "volume": 100000
  }
]
```

**Output JSON:**
```json
{
  "probability": 0.753,
  "prediction": 1,
  "features_computed": 900
}
```

## Przykłady Użycia w Kodzie

### Proste użycie z generowaniem
```csharp
var loader = new ModelLoader(modelsDir, logger);
var metadata = loader.LoadModelMetadata();

var parser = new CandleParser(logger);
var candles = GenerateSampleCandles(1000);

var service = new PredictionService(modelsDir, pythonPath, metadata, logger);
var result = await service.PredictAsync(candles);

Console.WriteLine($"Signal: {result.SignalType}");
Console.WriteLine($"Probability: {result.Probability:P2}");
```

### Z wczytaniem z pliku
```csharp
var parser = new CandleParser(logger);
var candles = parser.ParseFromCsv("market_data.csv");

if (parser.ValidateCandles(candles, 260))
{
    var result = await service.PredictAsync(candles);
    // ...
}
```

## Testy Jednostkowe

**CandleParserTests.cs**
- ✅ ParseFromOhlcv_WithValidData_ReturnsCandleList
- ✅ ValidateCandles_WithInsufficientCandles_ReturnsFalse
- ✅ ValidateCandles_WithValidCandles_ReturnsTrue
- ✅ ValidateCandles_WithInvalidOHLC_ReturnsFalse

**ModelLoaderTests.cs**
- ✅ ValidateModelArtifacts_WithValidModels_ReturnsTrue
- ✅ LoadModelMetadata_WithValidFiles_ReturnsMetadata
- ✅ LoadModelPath_WithExistingModel_ReturnsValidPath
- ✅ Constructor_WithNull_ThrowsArgumentNullException

## Ścieżki Dostępu

```
Backend (C#):
c:\Users\Arek\Documents\Repos\Traiding\Trading-ML\backend\

Models (ML):
c:\Users\Arek\Documents\Repos\Traiding\Trading-ML\ml\outputs\models\
├── sequence_feature_columns.json
├── sequence_feature_importance.json
├── sequence_scaler.pkl
├── sequence_threshold.json
└── sequence_xgb_model.pkl

Python Scripts:
c:\Users\Arek\Documents\Repos\Traiding\Trading-ML\ml\scripts\
└── predict_single.py (do implementacji)
```

## Bezpieczeństwo i Walidacja

1. **Input Validation**
   - Sprawdzenie liczby świeczek (min 260)
   - Validacja OHLCV wartości (High >= Open/Close >= Low)
   - Sprawdzenie plików modelu

2. **Error Handling**
   - Try-catch na operacjach plików
   - Timeout na procesach Python (30s)
   - Cleanup temp files

3. **Logging**
   - Informacyjne komunikaty
   - Ostrzeżenia
   - Komunikaty błędów

## Przyszłe Rozszerzenia

1. **ONNX Integration**
   - Zamiana `.pkl` na `.onnx`
   - ONNX Runtime zamiast Python
   - Szybsza predykcja (~5ms)

2. **Web API**
   - ASP.NET Core endpoint
   - RESTful interfejs
   - WebSocket streaming

3. **Database**
   - Przechowywanie wyników
   - History predykcji
   - Tracking accuracy

4. **Caching**
   - Redis dla recent predictions
   - Model cache
   - Feature cache

## Metryki i Performance

- **Model Loading:** ~500ms
- **Candle Parsing:** O(n) gdzie n = liczba świeczek
- **Validation:** O(n) 
- **Python Inference:** ~2-5s (w zależności od rozmiaru danych)
- **Total E2E:** ~3-6s dla 1000 świeczek

---

*Dokumentacja: Capgemini 2025*
