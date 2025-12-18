using System.Diagnostics;
using TradingML.ModelPrediction.Models;

namespace TradingML.ModelPrediction.Services;

/// <summary>
/// Service for running model inference.
/// Integrates with Python ML model via subprocess or external process.
/// </summary>
public class PredictionService
{
    private readonly string _modelsDirectory;
    private readonly string _pythonPath;
    private readonly ModelMetadata _metadata;
    private readonly ILogger<PredictionService> _logger;

    public PredictionService(
        string modelsDirectory,
        string pythonPath,
        ModelMetadata metadata,
        ILogger<PredictionService> logger)
    {
        _modelsDirectory = modelsDirectory ?? throw new ArgumentNullException(nameof(modelsDirectory));
        _pythonPath = pythonPath ?? throw new ArgumentNullException(nameof(pythonPath));
        _metadata = metadata ?? throw new ArgumentNullException(nameof(metadata));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Predicts using the ML model based on candles.
    /// Calls a Python inference script to load and run the XGBoost model.
    /// </summary>
    public async Task<PredictionResult> PredictAsync(List<Candle> candles)
    {
        if (candles == null || candles.Count == 0)
            throw new ArgumentException("Candles list cannot be empty", nameof(candles));

        if (candles.Count < _metadata.RecommendedMinCandles)
        {
            _logger.LogWarning($"Insufficient candles: {candles.Count} < {_metadata.RecommendedMinCandles}");
        }

        _logger.LogInformation($"Starting prediction with {candles.Count} candles");

        // Take the last window_size candles
        var recentCandles = candles.TakeLast(_metadata.WindowSize).ToList();
        if (recentCandles.Count < _metadata.WindowSize)
        {
            _logger.LogWarning($"Requested window size {_metadata.WindowSize} but only {recentCandles.Count} candles available");
        }

        try
        {
            // Call Python prediction script
            var (probability, prediction) = await CallPythonPredictionAsync(recentCandles);

            var isSignal = probability >= _metadata.Threshold;
            var signalType = prediction == 1
                ? (isSignal ? "BUY" : "BUY_LOW_CONFIDENCE")
                : "SELL";

            var result = new PredictionResult
            {
                Probability = probability,
                Prediction = prediction,
                IsSignal = isSignal,
                SignalType = signalType,
                PredictionTime = DateTime.UtcNow,
                CandlesUsed = candles.Count,
                Threshold = _metadata.Threshold
            };

            _logger.LogInformation($"Prediction: {result.SignalType} (prob={result.Probability:P2}, threshold={_metadata.Threshold:P2})");

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during prediction");
            throw;
        }
    }

    /// <summary>
    /// Calls a Python subprocess to perform inference.
    /// Returns (probability, prediction_class).
    /// </summary>
    private async Task<(double probability, int prediction)> CallPythonPredictionAsync(List<Candle> candles)
    {
        // For now, this is a stub. In production:
        // 1. Write candle data to a temp file (JSON/CSV)
        // 2. Call Python inference script: python predict.py --input-file <temp> --models-dir <models>
        // 3. Read JSON output with probability and prediction
        // 4. Clean up temp file

        var tempInputFile = Path.Combine(Path.GetTempPath(), $"predict_input_{Guid.NewGuid()}.json");
        var tempOutputFile = Path.Combine(Path.GetTempPath(), $"predict_output_{Guid.NewGuid()}.json");

        try
        {
            // Write candle data to JSON
            WriteCandlesToJson(tempInputFile, candles);

            // Call Python prediction script
            var arguments = $"\"{Path.Combine(GetProjectRoot(), "ml/scripts/predict_single.py")}\" --input-file \"{tempInputFile}\" --models-dir \"{_modelsDirectory}\" --output-file \"{tempOutputFile}\"";

            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = _pythonPath,
                    Arguments = arguments,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await Task.Run(() => process.WaitForExit(30000)); // 30 second timeout

            if (process.ExitCode != 0)
            {
                _logger.LogError($"Python prediction failed: {error}");
                throw new InvalidOperationException($"Prediction script failed: {error}");
            }

            // Read output
            if (!File.Exists(tempOutputFile))
            {
                throw new FileNotFoundException("Prediction output file not created");
            }

            var outputJson = File.ReadAllText(tempOutputFile);
            // Parse JSON and extract probability and prediction
            // For now, return dummy values - you'll implement actual parsing
            return (probability: 0.75, prediction: 1);
        }
        finally
        {
            // Cleanup temp files
            try
            {
                if (File.Exists(tempInputFile)) File.Delete(tempInputFile);
                if (File.Exists(tempOutputFile)) File.Delete(tempOutputFile);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to clean up temporary files");
            }
        }
    }

    /// <summary>
    /// Writes candles to a JSON file for Python script consumption.
    /// </summary>
    private void WriteCandlesToJson(string filePath, List<Candle> candles)
    {
        var json = System.Text.Json.JsonSerializer.Serialize(
            candles.Select(c => new
            {
                c.Timestamp,
                c.Open,
                c.High,
                c.Low,
                c.Close,
                c.Volume
            }),
            new System.Text.Json.JsonSerializerOptions { WriteIndented = true }
        );

        File.WriteAllText(filePath, json);
        _logger.LogDebug($"Wrote {candles.Count} candles to {filePath}");
    }

    /// <summary>
    /// Gets the project root directory.
    /// </summary>
    private string GetProjectRoot()
    {
        var current = new DirectoryInfo(_modelsDirectory);
        while (current.Parent != null)
        {
            if (File.Exists(Path.Combine(current.FullName, "README.md")))
                return current.FullName;
            current = current.Parent;
        }
        return AppContext.BaseDirectory;
    }
}
