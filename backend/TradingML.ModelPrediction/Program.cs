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
                new ConsoleLogger<PredictionService>()
            );

            logger.LogInformation("\n" + new string('=', 80));
            logger.LogInformation("RUNNING MODEL PREDICTION");
            logger.LogInformation(new string('=', 80));

            var result = await predictionService.PredictAsync(candles);

            // Output results
            PrintResults(result, metadata, logger);

            // Save output if requested
            if (!string.IsNullOrEmpty(options.OutputFile))
            {
                SaveResults(result, metadata, candles, options.OutputFile, logger);
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
        var baseTime = DateTime.UtcNow.AddMinutes(-count);

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
