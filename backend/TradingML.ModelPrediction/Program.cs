using TradingML.ModelPrediction.Models;
using TradingML.ModelPrediction.Services;
using System.Globalization;

namespace TradingML.ModelPrediction;

/// <summary>
/// Console application for model prediction on XAU/USD candles.
/// 
/// Usage:
///   TradingML.ModelPrediction.exe --candles-file input.csv --models-dir path/to/models --output output.json
///   TradingML.ModelPrediction.exe --sample 1000
/// </summary>
class Program
{
    private const int LondonNyStartHour = 8;  // UTC approx, must match ML session filter
    private const int LondonNyEndHour = 22;   // UTC approx, must match ML session filter

    static async Task<int> Main(string[] args)
    {
        try
        {
            var parser = new ArgumentParser();
            var options = parser.Parse(args);

            if (options.ShowHelp)
            {
                parser.PrintHelp();
                return 0;
            }

            // Setup
            var logger = new ConsoleLogger<Program>();
            var modelsDir = options.ModelsDirectory ?? GetDefaultModelsDirectory();
            var pythonPath = options.PythonPath ?? "python";

            logger.LogInformation("Starting TradingML Model Prediction Console");
            logger.LogInformation($"Models Directory: {modelsDir}");

            // Validate models directory
            if (!Directory.Exists(modelsDir))
            {
                logger.LogError($"Models directory not found: {modelsDir}");
                return 1;
            }

            // Load model metadata
            var modelLoader = new ModelLoader(modelsDir, new ConsoleLogger<ModelLoader>());
            if (!modelLoader.ValidateModelArtifacts())
            {
                logger.LogError("Model artifacts validation failed");
                return 1;
            }

            var metadata = modelLoader.LoadModelMetadata();
            logger.LogInformation($"Model loaded: {metadata.TotalFeatures} features, threshold={metadata.Threshold:P2}");

            // Load or generate candles
            List<Candle> candles;
            if (!string.IsNullOrEmpty(options.CandlesFile))
            {
                var candleParser = new CandleParser(new ConsoleLogger<CandleParser>());
                logger.LogInformation($"Loading candles from {options.CandlesFile}");
                candles = candleParser.ParseFromCsv(options.CandlesFile);

                if (!candleParser.ValidateCandles(candles, metadata.RecommendedMinCandles))
                {
                    logger.LogError("Candle validation failed");
                    return 1;
                }
            }
            else if (options.SampleCandleCount > 0)
            {
                logger.LogInformation($"Generating {options.SampleCandleCount} sample candles");
                candles = GenerateSampleCandles(options.SampleCandleCount);

                // Validate sample candles
                var candleParser = new CandleParser(new ConsoleLogger<CandleParser>());
                if (!candleParser.ValidateCandles(candles, metadata.RecommendedMinCandles))
                {
                    logger.LogError("Sample candle validation failed");
                    return 1;
                }
            }
            else
            {
                logger.LogError("Either --candles-file or --sample must be specified");
                parser.PrintHelp();
                return 1;
            }

            logger.LogInformation($"Loaded {candles.Count} candles");

            // Run prediction
            var predictionService = new PredictionService(
                modelsDir,
                pythonPath,
                metadata,
                new ConsoleLogger<PredictionService>(),
                options.SkipRegime // pass through CLI flag
            );

            logger.LogInformation("\n" + new string('=', 80));
            logger.LogInformation("RUNNING SLIDING WINDOW PREDICTION");
            logger.LogInformation(new string('=', 80));

            // Run sliding window predictions
            var windowResults = await RunSlidingWindowPrediction(
                candles,
                predictionService,
                metadata,
                logger
            );

            // Save output if requested
            if (!string.IsNullOrEmpty(options.OutputFile))
            {
                SaveSlidingWindowResults(windowResults, metadata, options.OutputFile, logger);
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"FATAL ERROR: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            Console.ResetColor();
            return 1;
        }
    }

    static async Task<List<SlidingWindowResult>> RunSlidingWindowPrediction(
        List<Candle> candles,
        PredictionService predictionService,
        ModelMetadata metadata,
        ConsoleLogger<Program> logger)
    {
        // Model uses M5 candles, but backend receives M1 candles
        // Convert window size: 80 M5 candles = 400 M1 candles = 400 minutes
        int windowSizeMinutes = metadata.WindowSize * 5; // M5 to M1 conversion
        int stepSize = 60; // Start with 60-minute steps (hourly scanning)
        const int STEP_SIZE_HOURLY = 60; // Move 60 candles at a time (hourly steps)
        const int STEP_SIZE_MINUTE = 1;  // Move 1 candle at a time (minute steps when trade active)

        var results = new List<SlidingWindowResult>();

        if (candles.Count < windowSizeMinutes)
        {
            logger.LogError($"Insufficient candles. Need {windowSizeMinutes}, have {candles.Count}");
            return results;
        }

        var totalWindows = Math.Max(0, ((candles.Count - windowSizeMinutes) / STEP_SIZE_HOURLY) + 1);
        logger.LogInformation($"Starting sliding window analysis: {candles.Count} candles, window size: {windowSizeMinutes} minutes ({metadata.WindowSize} M5 candles), step: {STEP_SIZE_HOURLY} (hourly)");
        logger.LogInformation($"Total windows (hourly estimate): {totalWindows}");
        Console.WriteLine();

        int windowCount = 0;
        int buySignalCount = 0;

        // Diagnostics
        int windowsWithPrediction1 = 0;
        int windowsWithProbGEThreshold = 0;
        int prediction1ButNotSignal = 0;
        var probabilities = new List<double>();

        // Track active trade to prevent opening multiple trades simultaneously
        SlidingWindowResult? activeTrade = null;

        // Start sliding window from where we have enough context for indicator calculation
        int minContextCandlesM5 = Math.Max(metadata.RecommendedMinCandles, metadata.WindowSize + 200); // M5 candles
        int minContextCandlesM1 = minContextCandlesM5 * 5; // Convert to M1 candles
        int startIndex = Math.Max(0, minContextCandlesM1 - windowSizeMinutes);

        for (int i = startIndex; i <= candles.Count - windowSizeMinutes; i += stepSize)
        {
            // Get current window - send ALL candles up to window end for proper indicator calculation
            var windowEndIndex = i + windowSizeMinutes - 1;
            var contextCandles = candles.GetRange(0, windowEndIndex + 1);  // All candles from start to window end
            var windowEndTime = candles[windowEndIndex].Timestamp;

            // Always monitor an active trade (even outside session hours)
            if (activeTrade != null && activeTrade.Outcome == "Pending")
            {
                EvaluateActiveTrade(activeTrade, candles, windowEndIndex);

                if (activeTrade.Outcome != "Pending")
                {
                    logger.LogInformation($"  Trade #{activeTrade.WindowIndex + 1} resolved: {activeTrade.Outcome} | P&L: {activeTrade.ProfitLoss?.ToString("0.#####", CultureInfo.InvariantCulture) ?? "N/A"}");
                    activeTrade = null;
                    stepSize = STEP_SIZE_HOURLY;
                    logger.LogInformation($"  Switched back to HOURLY scanning (step size: {stepSize} minutes)");
                }
            }

            // When a trade is active, do not open new trades; also avoid running Python every minute.
            if (activeTrade != null)
                continue;

            // Session filter (London + NY): only evaluate/open signals during 08:00-22:00
            if (!IsInLondonNySession(windowEndTime))
                continue;

            // Ensure we have minimum required context
            if (contextCandles.Count < minContextCandlesM1)
            {
                logger.LogDebug($"Skipping window at index {i}: insufficient context ({contextCandles.Count} < {minContextCandlesM1})");
                continue;
            }

            // Run prediction on this window
            var result = await predictionService.PredictAsync(contextCandles);

            windowCount++;

            // Collect diagnostics
            probabilities.Add(result.Probability);
            if (result.Prediction == 1) windowsWithPrediction1++;
            if (result.Probability >= result.Threshold) windowsWithProbGEThreshold++;
            if (result.Prediction == 1 && !result.IsSignal) prediction1ButNotSignal++;

            // Log progress
            if (windowCount % 100 == 0)
            {
                logger.LogInformation($"  Processed {windowCount} windows, found {buySignalCount} BUY signals");
            }

            // Filter: only BUY signals with IsSignal = true AND no active trade
            if (result.IsSignal && result.Prediction == 1 && activeTrade == null)
            {
                buySignalCount++;

                var windowResult = new SlidingWindowResult
                {
                    WindowIndex = windowCount - 1,
                    EntryCandelIndex = i + windowSizeMinutes,
                    EntryTime = windowEndTime,
                    EntryPrice = result.EntryPrice,
                    StopLoss = result.StopLoss,
                    TakeProfit = result.TakeProfit,
                    Probability = result.Probability,
                    Threshold = result.Threshold,
                    Confidence = result.Confidence,
                    AtrM5 = result.AtrM5,
                    SlAtrMultiplier = result.SlAtrMultiplier,
                    TpAtrMultiplier = result.TpAtrMultiplier,
                    RiskRewardRatio = result.RiskRewardRatio,
                    ExpectedWinRate = result.ExpectedWinRate
                };

                // Evaluate trade outcome based on next candles
                EvaluateTradeOutcome(windowResult, candles, windowEndIndex);

                results.Add(windowResult);

                // Set as active trade if it's still pending
                if (windowResult.Outcome == "Pending")
                {
                    activeTrade = windowResult;
                    stepSize = STEP_SIZE_MINUTE; // Switch to minute-by-minute scanning
                    logger.LogInformation($"  Trade #{buySignalCount} opened - Switched to MINUTE scanning (step size: {stepSize} minute)");
                }

                // Print found signal
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"✓ BUY Signal #{buySignalCount} at {windowEndTime:yyyy-MM-dd HH:mm:ss} (Window {windowCount})");
                Console.ResetColor();
                Console.WriteLine($"  Entry Price: {result.EntryPrice?.ToString("0.#####", CultureInfo.InvariantCulture) ?? "N/A"}");
                Console.WriteLine($"  Stop Loss:   {result.StopLoss?.ToString("0.#####", CultureInfo.InvariantCulture) ?? "N/A"}");
                Console.WriteLine($"  Take Profit: {result.TakeProfit?.ToString("0.#####", CultureInfo.InvariantCulture) ?? "N/A"}");
                Console.WriteLine($"  Probability: {result.Probability:P2}");

                // Print outcome
                if (windowResult.Outcome != "Pending")
                {
                    var outcomeColor = windowResult.Outcome == "Win" ? ConsoleColor.Green : ConsoleColor.Red;
                    Console.ForegroundColor = outcomeColor;
                    Console.WriteLine($"  Outcome: {windowResult.Outcome} | P&L: {windowResult.ProfitLoss?.ToString("0.#####", CultureInfo.InvariantCulture) ?? "N/A"}");
                    Console.ResetColor();
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"  Status: ACTIVE TRADE - awaiting TP/SL");
                    Console.ResetColor();
                }
                Console.WriteLine();
            }
            else if (result.IsSignal && result.Prediction == 1 && activeTrade != null)
            {
                // Signal received but active trade is in progress
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"⊗ BUY Signal at {candles[i + windowSizeMinutes - 1].Timestamp:yyyy-MM-dd HH:mm:ss} SKIPPED - active trade in progress");
                Console.ResetColor();
                Console.WriteLine();
            }
        }

        // Write diagnostics summary
        var diag = new
        {
            TotalWindows = windowCount,
            WindowsWithPrediction1 = windowsWithPrediction1,
            WindowsWithProbabilityGEThreshold = windowsWithProbGEThreshold,
            Prediction1ButNotSignal = prediction1ButNotSignal,
            MeanProbability = probabilities.Any() ? probabilities.Average() : 0.0,
            MaxProbability = probabilities.Any() ? probabilities.Max() : 0.0,
            MinProbability = probabilities.Any() ? probabilities.Min() : 0.0,
            TopProbabilities = probabilities.OrderByDescending(p => p).Take(20).ToList()
        };

        try
        {
            var diagDir = Path.Combine(AppContext.BaseDirectory, "..", "outputs");
            Directory.CreateDirectory(diagDir);
            var diagFile = Path.Combine(diagDir, $"prediction_diagnostics_{DateTime.UtcNow:yyyyMMddHHmmss}.json");
            File.WriteAllText(diagFile, System.Text.Json.JsonSerializer.Serialize(diag, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            logger.LogInformation($"Diagnostics saved to {diagFile}");
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Failed to save diagnostics");
        }

        logger.LogInformation($"\nAnalysis complete. Total windows: {windowCount}, BUY signals found: {buySignalCount}");
        return results;
    }

    private static bool IsInLondonNySession(DateTime timestamp)
    {
        // Mirrors ML session filter used during training: london_ny = 08:00..22:00 (UTC approx).
        // Candle timestamps in this project are typically timezone-naive; we treat their hour as UTC.
        var hour = timestamp.Hour;
        return hour >= LondonNyStartHour && hour < LondonNyEndHour;
    }

    static void EvaluateActiveTrade(SlidingWindowResult activeTrade, List<Candle> allCandles, int currentWindowEndIndex)
    {
        // Evaluate trade from its actual entry point forward
        // This checks if TP/SL is hit anytime after entry
        if (!activeTrade.EntryPrice.HasValue || !activeTrade.StopLoss.HasValue || !activeTrade.TakeProfit.HasValue)
            return;

        const int M5_TO_M1 = 5;
        const int MIN_HOLD_M5_CANDLES = 2;     // must match ML target creation
        const int MAX_HORIZON_M5_CANDLES = 60; // must match ML target creation

        var entryIdx = activeTrade.EntryCandelIndex;
        if (entryIdx >= allCandles.Count)
            return;

        // Only evaluate using candles that are available up to the current window end,
        // and never beyond the configured max horizon.
        var horizonEndIdx = Math.Min(entryIdx + (MAX_HORIZON_M5_CANDLES * M5_TO_M1), allCandles.Count - 1);
        var evalEndIdx = Math.Min(Math.Max(currentWindowEndIndex, entryIdx), horizonEndIdx);
        var minHoldEndIdx = Math.Min(entryIdx + (MIN_HOLD_M5_CANDLES * M5_TO_M1), evalEndIdx);

        // If min-hold window not yet elapsed, keep trade pending.
        if (evalEndIdx < minHoldEndIdx)
            return;

        var sl = activeTrade.StopLoss.Value;
        var tp = activeTrade.TakeProfit.Value;
        var entryPrice = activeTrade.EntryPrice.Value;

        // Check candles from min-hold end up to evaluation end.
        for (int i = minHoldEndIdx; i <= evalEndIdx; i++)
        {
            var candle = allCandles[i];

            // Check if take profit was hit
            if (candle.High >= tp)
            {
                activeTrade.Outcome = "Win";
                activeTrade.ExitPrice = tp;
                activeTrade.ExitTime = candle.Timestamp;
                activeTrade.ProfitLoss = tp - entryPrice;
                return;
            }

            // Check if stop loss was hit
            if (candle.Low <= sl)
            {
                activeTrade.Outcome = "Loss";
                activeTrade.ExitPrice = sl;
                activeTrade.ExitTime = candle.Timestamp;
                activeTrade.ProfitLoss = sl - entryPrice;
                return;
            }
        }

        // If neither TP nor SL hit yet:
        // - If max horizon reached, close the trade at horizon end.
        // - Otherwise keep it pending and mark-to-market at current close.
        if (currentWindowEndIndex >= horizonEndIdx)
        {
            var horizonCandle = allCandles[horizonEndIdx];
            activeTrade.Outcome = "Timeout";
            activeTrade.ExitTime = horizonCandle.Timestamp;
            activeTrade.ExitPrice = horizonCandle.Close;
            activeTrade.ProfitLoss = horizonCandle.Close - entryPrice;
            return;
        }

        var markCandle = allCandles[evalEndIdx];
        activeTrade.ExitTime = markCandle.Timestamp;
        activeTrade.ExitPrice = markCandle.Close;
        activeTrade.ProfitLoss = markCandle.Close - entryPrice;
    }

    static void EvaluateTradeOutcome(SlidingWindowResult signal, List<Candle> allCandles, int windowEndIndex)
    {
        // Validate inputs
        if (!signal.EntryPrice.HasValue || !signal.StopLoss.HasValue || !signal.TakeProfit.HasValue)
            return;

        // Trade is opened at Open of the next candle after signal (realistic entry)
        var entryIdx = windowEndIndex + 1;
        if (entryIdx >= allCandles.Count)
            return; // No future data available

        var entryCandle = allCandles[entryIdx];
        signal.EntryTime = entryCandle.Timestamp;
        signal.EntryPrice = entryCandle.Open;

        // If ATR is missing we cannot grade the trade deterministically.
        var atr = signal.AtrM5;
        if (!atr.HasValue || atr.Value <= 0)
        {
            signal.Outcome = "InvalidATR";
            signal.ProfitLoss = null;
            return;
        }

        // SL/TP wyliczane względem nowego EntryPrice, ale ATR i multiplikatory z sygnału
        var slMult = signal.SlAtrMultiplier ?? 1.0;
        var tpMult = signal.TpAtrMultiplier ?? 2.0;
        signal.StopLoss = signal.EntryPrice - atr.Value * slMult;
        signal.TakeProfit = signal.EntryPrice + atr.Value * tpMult;

        const int M5_TO_M1 = 5;
        const int MIN_HOLD_M5_CANDLES = 2;     // must match ML target creation
        const int MAX_HORIZON_M5_CANDLES = 60; // must match ML target creation

        // Look at future candles after the entry (starting from entryIdx) using the same horizon
        // as the training target creation: 60 M5 candles = 300 minutes of M1 candles.
        var maxHorizonMinutes = MAX_HORIZON_M5_CANDLES * M5_TO_M1;
        var minHoldMinutes = MIN_HOLD_M5_CANDLES * M5_TO_M1;
        var lookaheadLimit = Math.Min(entryIdx + maxHorizonMinutes, allCandles.Count);

        // Enforce min-hold: ignore TP/SL hits before min-hold window elapses (matches ML labels).
        var startEvalIdx = Math.Min(entryIdx + minHoldMinutes, lookaheadLimit);

        for (int i = startEvalIdx; i < lookaheadLimit; i++)
        {
            var candle = allCandles[i];
            var sl = signal.StopLoss.Value;
            var tp = signal.TakeProfit.Value;
            var entryPrice = signal.EntryPrice.Value;

            // Check if take profit was hit
            if (candle.High >= tp)
            {
                signal.Outcome = "Win";
                signal.ExitPrice = tp;
                signal.ExitTime = candle.Timestamp;
                signal.ProfitLoss = tp - entryPrice;
                return;
            }

            // Check if stop loss was hit
            if (candle.Low <= sl)
            {
                signal.Outcome = "Loss";
                signal.ExitPrice = sl;
                signal.ExitTime = candle.Timestamp;
                signal.ProfitLoss = sl - entryPrice; // Will be negative
                return;
            }
        }

        // If neither TP nor SL hit within horizon, close at horizon end.
        if (entryIdx < allCandles.Count)
        {
            var lastIdx = Math.Min(Math.Max(entryIdx, lookaheadLimit - 1), allCandles.Count - 1);
            var lastCandle = allCandles[lastIdx];
            signal.ExitTime = lastCandle.Timestamp;
            signal.ExitPrice = lastCandle.Close;
            signal.ProfitLoss = lastCandle.Close - signal.EntryPrice.Value;
            signal.Outcome = "Timeout";
        }
    }

    static void PrintResults(PredictionResult result, ModelMetadata metadata, ConsoleLogger<Program> logger)
    {
        Console.WriteLine();
        Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                    PREDICTION RESULTS                          ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        Console.ForegroundColor = result.IsSignal ? ConsoleColor.Green : ConsoleColor.Yellow;
        Console.WriteLine($"  Signal Type:          {result.SignalType,-30}");
        Console.ResetColor();

        Console.WriteLine($"  Probability:          {result.Probability:P2}");
        Console.WriteLine($"  Decision Threshold:   {result.Threshold:P2}");
        Console.WriteLine($"  Prediction Class:     {(result.Prediction == 1 ? "BUY" : "SELL"),-30}");
        Console.WriteLine($"  Candles Provided:     {result.CandlesProvided,-30}");
        Console.WriteLine($"  Candles Used:         {result.CandlesUsed,-30}");
        if (result.EffectiveInputCandlesCount.HasValue && result.EffectiveInputFromUtc.HasValue && result.EffectiveInputToUtc.HasValue)
        {
            var fp = string.IsNullOrWhiteSpace(result.EffectiveInputFingerprint) ? "(n/a)" : result.EffectiveInputFingerprint;
            Console.WriteLine(
                $"  Python Effective:      last 7d => {result.EffectiveInputCandlesCount.Value} candles, " +
                $"{result.EffectiveInputFromUtc.Value:yyyy-MM-dd HH:mm}..{result.EffectiveInputToUtc.Value:yyyy-MM-dd HH:mm} UTC, fp={fp}");
        }
        Console.WriteLine($"  Prediction Time:      {result.PredictionTime:yyyy-MM-dd HH:mm:ss UTC}");
        Console.WriteLine($"  Model Win Rate:       {metadata.WinRate:P2}");

        if (!string.IsNullOrWhiteSpace(result.Confidence))
            Console.WriteLine($"  Python Confidence:    {result.Confidence,-30}");
        if (result.ExpectedWinRate.HasValue)
            Console.WriteLine($"  Expected Win Rate:    {result.ExpectedWinRate.Value:P2}");
        if (result.EntryPrice.HasValue)
            Console.WriteLine($"  Entry Price:          {result.EntryPrice.Value.ToString("0.#####", CultureInfo.InvariantCulture),-30}");
        if (result.AtrM5.HasValue)
            Console.WriteLine($"  ATR (M5):             {result.AtrM5.Value.ToString("0.#####", CultureInfo.InvariantCulture),-30}");
        if (result.StopLoss.HasValue)
            Console.WriteLine($"  Stop Loss:            {result.StopLoss.Value.ToString("0.#####", CultureInfo.InvariantCulture),-30}");
        if (result.TakeProfit.HasValue)
            Console.WriteLine($"  Take Profit:          {result.TakeProfit.Value.ToString("0.#####", CultureInfo.InvariantCulture),-30}");
        if (result.SlAtrMultiplier.HasValue || result.TpAtrMultiplier.HasValue || result.RiskRewardRatio.HasValue)
        {
            var slm = result.SlAtrMultiplier?.ToString("0.###", CultureInfo.InvariantCulture) ?? "(n/a)";
            var tpm = result.TpAtrMultiplier?.ToString("0.###", CultureInfo.InvariantCulture) ?? "(n/a)";
            var rr = result.RiskRewardRatio?.ToString("0.###", CultureInfo.InvariantCulture) ?? "(n/a)";
            Console.WriteLine($"  Risk Params:          SLx={slm}, TPx={tpm}, RR={rr}");
        }

        if (!string.IsNullOrWhiteSpace(result.PythonOutputJson))
            Console.WriteLine($"  Python JSON:          {result.PythonOutputJson}");

        Console.WriteLine();
        Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");

        if (result.IsSignal)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ SIGNAL CONFIRMED - Probability exceeds threshold");
            Console.ResetColor();
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("⚠ NO SIGNAL - Probability below threshold");
            Console.ResetColor();
        }

        Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();
    }

    static void SaveSlidingWindowResults(
        List<SlidingWindowResult> results,
        ModelMetadata metadata,
        string outputFile,
        ConsoleLogger<Program> logger)
    {
        try
        {
            // Create output directory if it doesn't exist
            var outputDir = Path.GetDirectoryName(outputFile);
            if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
                logger.LogInformation($"Created output directory: {outputDir}");
            }

            // Calculate summary statistics
            var totalSignals = results.Count;
            var winCount = results.Count(r => r.Outcome == "Win");
            var lossCount = results.Count(r => r.Outcome == "Loss");
            var pendingCount = results.Count(r => r.Outcome == "Pending");
            var totalProfitLoss = results.Where(r => r.ProfitLoss.HasValue).Sum(r => r.ProfitLoss.Value);
            var avgWin = results.Where(r => r.Outcome == "Win" && r.ProfitLoss.HasValue).Average(r => (double?)r.ProfitLoss) ?? 0;
            var avgLoss = results.Where(r => r.Outcome == "Loss" && r.ProfitLoss.HasValue).Average(r => (double?)r.ProfitLoss) ?? 0;
            var winRate = totalSignals > 0 ? (double)winCount / totalSignals : 0;
            var profitableSignals = results.Where(r => r.ProfitLoss.HasValue && r.ProfitLoss.Value > 0).Count();
            var profitabilityRate = totalSignals > 0 ? (double)profitableSignals / totalSignals : 0;

            var output = new
            {
                Summary = new
                {
                    TotalSignals = totalSignals,
                    WonTrades = winCount,
                    LostTrades = lossCount,
                    PendingTrades = pendingCount,
                    WinRate = $"{winRate:P2}",
                    ProfitabilityRate = $"{profitabilityRate:P2}",
                    TotalProfitLoss = $"{totalProfitLoss:0.#####}",
                    AverageWin = $"{avgWin:0.#####}",
                    AverageLoss = $"{avgLoss:0.#####}",
                    GeneratedAt = DateTime.UtcNow,
                    ModelThreshold = $"{metadata.Threshold:P2}",
                    ModelWinRate = $"{metadata.WinRate:P2}"
                },
                Signals = results.Select(r => new
                {
                    WindowIndex = r.WindowIndex,
                    EntryTime = r.EntryTime.ToString("yyyy-MM-dd HH:mm:ss"),
                    EntryPrice = r.EntryPrice?.ToString("0.#####", CultureInfo.InvariantCulture),
                    ExitTime = r.ExitTime?.ToString("yyyy-MM-dd HH:mm:ss"),
                    ExitPrice = r.ExitPrice?.ToString("0.#####", CultureInfo.InvariantCulture),
                    StopLoss = r.StopLoss?.ToString("0.#####", CultureInfo.InvariantCulture),
                    TakeProfit = r.TakeProfit?.ToString("0.#####", CultureInfo.InvariantCulture),
                    Outcome = r.Outcome,
                    ProfitLoss = r.ProfitLoss?.ToString("0.#####", CultureInfo.InvariantCulture),
                    Probability = $"{r.Probability:P2}",
                    Threshold = $"{r.Threshold:P2}",
                    Confidence = r.Confidence,
                    AtrM5 = r.AtrM5?.ToString("0.#####", CultureInfo.InvariantCulture),
                    SlAtrMultiplier = r.SlAtrMultiplier?.ToString("0.###", CultureInfo.InvariantCulture),
                    TpAtrMultiplier = r.TpAtrMultiplier?.ToString("0.###", CultureInfo.InvariantCulture),
                    RiskRewardRatio = r.RiskRewardRatio?.ToString("0.###", CultureInfo.InvariantCulture),
                    ExpectedWinRate = r.ExpectedWinRate?.ToString("P2")
                }).ToList()
            };

            var json = System.Text.Json.JsonSerializer.Serialize(output, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(outputFile, json);
            logger.LogInformation($"Results saved to {outputFile}");

            // Print summary to console
            PrintSummary(winCount, lossCount, pendingCount, totalProfitLoss, winRate, avgWin, avgLoss);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Failed to save results to {outputFile}");
        }
    }

    static void PrintSummary(int wins, int losses, int pending, double totalPnL, double winRate, double avgWin, double avgLoss)
    {
        Console.WriteLine();
        Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                    TRADING SUMMARY                             ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"  Total Signals:        {wins + losses + pending}");
        
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  Won Trades:           {wins}");
        Console.ResetColor();
        
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"  Lost Trades:          {losses}");
        Console.ResetColor();
        
        Console.WriteLine($"  Pending Trades:       {pending}");
        Console.WriteLine($"  Win Rate:             {winRate:P2}");
        Console.WriteLine();
        
        Console.ForegroundColor = totalPnL >= 0 ? ConsoleColor.Green : ConsoleColor.Red;
        Console.WriteLine($"  Total P&L:            {totalPnL:0.#####}");
        Console.ResetColor();
        
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  Average Win:          {avgWin:0.#####}");
        Console.ResetColor();
        
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"  Average Loss:         {avgLoss:0.#####}");
        Console.ResetColor();
        
        Console.WriteLine();
        Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();
    }

    static void SaveResults(
        PredictionResult result,
        ModelMetadata metadata,
        List<Candle> candles,
        string outputFile,
        ConsoleLogger<Program> logger)
    {
        try
        {
            var output = new
            {
                result.SignalType,
                result.Probability,
                result.Threshold,
                result.IsSignal,
                result.Prediction,
                result.CandlesUsed,
                result.CandlesProvided,
                result.EffectiveInputCandlesCount,
                result.EffectiveInputFromUtc,
                result.EffectiveInputToUtc,
                result.EffectiveInputFingerprint,
                result.EntryPrice,
                result.AtrM5,
                result.StopLoss,
                result.TakeProfit,
                result.SlAtrMultiplier,
                result.TpAtrMultiplier,
                result.RiskRewardRatio,
                result.ExpectedWinRate,
                result.Confidence,
                result.PredictionTime,
                ModelWinRate = metadata.WinRate,
                ModelThreshold = metadata.Threshold,
                FirstCandleTime = candles.First().Timestamp,
                LastCandleTime = candles.Last().Timestamp
            };

            var json = System.Text.Json.JsonSerializer.Serialize(output, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(outputFile, json);
            logger.LogInformation($"Results saved to {outputFile}");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Failed to save results to {outputFile}");
        }
    }

    static List<Candle> GenerateSampleCandles(int count)
    {
        var candles = new List<Candle>();
        var random = new Random(42); // Fixed seed for reproducibility
        var basePrice = 2000.0;
        // Generate candles over 7 days to match Python's "last 7 days" expectation
        var baseTime = DateTime.UtcNow.AddDays(-7);

        for (int i = 0; i < count; i++)
        {
            var open = basePrice + random.NextDouble() * 10 - 5;
            var close = open + random.NextDouble() * 10 - 5;
            var high = Math.Max(open, close) + Math.Abs(random.NextDouble() * 5);
            var low = Math.Min(open, close) - Math.Abs(random.NextDouble() * 5);
            var volume = random.NextDouble() * 1000000;

            candles.Add(new Candle
            {
                Timestamp = baseTime.AddMinutes(i),
                Open = open,
                High = high,
                Low = low,
                Close = close,
                Volume = volume
            });

            basePrice = close;
        }

        return candles;
    }

    static string GetDefaultModelsDirectory()
    {
        // Try to find the ML models directory relative to this executable
        var current = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(current))
        {
            var modelsPath = Path.Combine(current, "ml", "outputs", "models");
            if (Directory.Exists(modelsPath))
                return modelsPath;

            current = Path.GetDirectoryName(current);
        }

        // Fallback to absolute path
        return Path.Combine("c:\\Users\\Arek\\Documents\\Repos\\Traiding\\Trading-ML\\ml\\outputs\\models");
    }
}

/// <summary>
/// Command-line argument parser.
/// </summary>
class ArgumentParser
{
    public class Options
    {
        public bool ShowHelp { get; set; }
        public string? CandlesFile { get; set; }
        public string? ModelsDirectory { get; set; }
        public string? OutputFile { get; set; }
        public string? PythonPath { get; set; }
        public int SampleCandleCount { get; set; }
        // New: allow passing --skip-regime to instruct Python to bypass regime filters
        public bool SkipRegime { get; set; }
    }

    public Options Parse(string[] args)
    {
        var options = new Options();

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--help" or "-h" or "-?":
                    options.ShowHelp = true;
                    break;

                case "--candles-file":
                    options.CandlesFile = i + 1 < args.Length ? args[++i] : null;
                    break;

                case "--models-dir" or "--models-directory":
                    options.ModelsDirectory = i + 1 < args.Length ? args[++i] : null;
                    break;

                case "--output" or "--output-file":
                    options.OutputFile = i + 1 < args.Length ? args[++i] : null;
                    break;

                case "--python":
                    options.PythonPath = i + 1 < args.Length ? args[++i] : null;
                    break;

                case "--skip-regime":
                    // Flag - no argument consumed
                    options.SkipRegime = true;
                    break;

                case "--sample":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out var count))
                        options.SampleCandleCount = count;
                    break;
            }
        }

        return options;
    }

    public void PrintHelp()
    {
        Console.WriteLine();
        Console.WriteLine("TradingML Model Prediction Console");
        Console.WriteLine("==================================");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  ModelPrediction.exe [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --help, -h              Show this help message");
        Console.WriteLine("  --candles-file <path>   Path to CSV file with candles (OHLCV format)");
        Console.WriteLine("  --sample <count>        Generate sample candles (default: none)");
        Console.WriteLine("  --models-dir <path>     Path to model artifacts (default: auto-detect)");
        Console.WriteLine("  --output <path>         Save results to JSON file (optional)");
        Console.WriteLine("  --python <path>         Python executable path (default: python)");
        Console.WriteLine("  --skip-regime            Skip ML regime filters when calling Python (sets SKIP_REGIME_FILTER=1)");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  # Use sample candles");
        Console.WriteLine("  ModelPrediction.exe --sample 1000");
        Console.WriteLine();
        Console.WriteLine("  # Use CSV file");
        Console.WriteLine("  ModelPrediction.exe --candles-file data.csv --output result.json");
        Console.WriteLine();
        Console.WriteLine("CSV Format:");
        Console.WriteLine("  Timestamp,Open,High,Low,Close,Volume");
        Console.WriteLine("  2025-01-01 00:00:00,2000.0,2010.5,1995.3,2005.2,100000");
        Console.WriteLine();
    }
}

/// <summary>
/// Represents a single BUY signal from sliding window analysis.
/// </summary>
class SlidingWindowResult
{
    public int WindowIndex { get; set; }
    public int EntryCandelIndex { get; set; } // Index of the candle where trade was opened
    public DateTime EntryTime { get; set; }
    public double? EntryPrice { get; set; }
    public double? StopLoss { get; set; }
    public double? TakeProfit { get; set; }
    public double Probability { get; set; }
    public double Threshold { get; set; }
    public string? Confidence { get; set; }
    public double? AtrM5 { get; set; }
    public double? SlAtrMultiplier { get; set; }
    public double? TpAtrMultiplier { get; set; }
    public double? RiskRewardRatio { get; set; }
    public double? ExpectedWinRate { get; set; }
    public string Outcome { get; set; } = "Pending"; // Win, Loss, Pending
    public double? ProfitLoss { get; set; }
    public DateTime? ExitTime { get; set; }
    public double? ExitPrice { get; set; }
}
