using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
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

     //   _logger.LogInformation($"Starting prediction with {candles.Count} candles");



        try
        {
            // Call Python prediction script (pass full context; Python uses last window internally)
            var py = await CallPythonPredictionAsync(candles);

            var isSignal = py.Probability >= _metadata.Threshold;
            var signalType = py.Prediction == 1
                ? (isSignal ? "BUY" : "BUY_LOW_CONFIDENCE")
                : "SELL";

            var result = new PredictionResult
            {
                Probability = py.Probability,
                Prediction = py.Prediction,
                IsSignal = isSignal,
                SignalType = signalType,
                PredictionTime = DateTime.UtcNow,
                CandlesUsed = py.M1CandlesAnalyzed ?? candles.Count,
                CandlesProvided = candles.Count,
                EntryPrice = py.EntryPrice,
                AtrM5 = py.AtrM5,
                StopLoss = py.StopLoss,
                TakeProfit = py.TakeProfit,
                SlAtrMultiplier = py.SlAtrMultiplier,
                TpAtrMultiplier = py.TpAtrMultiplier,
                RiskRewardRatio = py.RiskRewardRatio,
                ExpectedWinRate = py.ExpectedWinRate,
                Confidence = py.Confidence,
                PythonOutputJson = py.CompactJson,
                Threshold = _metadata.Threshold
            };

            if (py.ScriptThreshold.HasValue && Math.Abs(py.ScriptThreshold.Value - _metadata.Threshold) > 1e-9)
                _logger.LogWarning($"Threshold mismatch: python={py.ScriptThreshold.Value:P4} metadata={_metadata.Threshold:P4}");

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
    /// Returns the parsed python result (and a compact JSON string for easy diffing).
    /// </summary>
    private sealed record PythonPrediction(
        double Probability,
        int Prediction,
        double? ScriptThreshold,
        int? M1CandlesAnalyzed,
        double? EntryPrice,
        double? AtrM5,
        double? StopLoss,
        double? TakeProfit,
        double? SlAtrMultiplier,
        double? TpAtrMultiplier,
        double? RiskRewardRatio,
        double? ExpectedWinRate,
        string? Confidence,
        string RawJson,
        string CompactJson);

    private static double? TryGetNullableDouble(JsonElement root, string propertyName)
    {
        if (!root.TryGetProperty(propertyName, out var el))
            return null;

        return el.ValueKind switch
        {
            JsonValueKind.Number => el.GetDouble(),
            JsonValueKind.Null => null,
            _ => null
        };
    }

    private static string? TryGetNullableString(JsonElement root, string propertyName)
    {
        if (!root.TryGetProperty(propertyName, out var el))
            return null;

        return el.ValueKind switch
        {
            JsonValueKind.String => el.GetString(),
            JsonValueKind.Null => null,
            _ => null
        };
    }

    private async Task<PythonPrediction> CallPythonPredictionAsync(List<Candle> candles)
    {
        // We call the real sequence predictor pipeline:
        //   python ml/src/scripts/predict_sequence.py --input-csv <temp.csv> --models-dir <models>
        // It prints human-readable text + a JSON object (after the "JSON Output:" marker) to stdout.

        var tempInputFile = Path.Combine(Path.GetTempPath(), $"predict_input_{Guid.NewGuid()}.csv");

        try
        {
            // Write candle data to CSV expected by predict_sequence.py
            WriteCandlesToSemicolonCsv(tempInputFile, candles);

            // Call Python prediction script
            var projectRoot = GetProjectRoot();
            var scriptPath = Path.Combine(projectRoot, "ml", "src", "scripts", "predict_sequence.py");
            if (!File.Exists(scriptPath))
                throw new FileNotFoundException($"Prediction script not found: {scriptPath}");

            var arguments = $"\"{scriptPath}\" --input-csv \"{tempInputFile}\" --models-dir \"{_modelsDirectory}\"";

            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = _pythonPath,
                    Arguments = arguments,
                    WorkingDirectory = projectRoot,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                }
            };

            // Ensure Python can import the repository's 'ml' package (some modules use absolute imports like 'ml.src...').
            var existingPythonPath = process.StartInfo.Environment.ContainsKey("PYTHONPATH")
                ? process.StartInfo.Environment["PYTHONPATH"]
                : null;
            process.StartInfo.Environment["PYTHONPATH"] = string.IsNullOrWhiteSpace(existingPythonPath)
                ? projectRoot
                : projectRoot + Path.PathSeparator + existingPythonPath;

            // Ensure Python prints Unicode safely on Windows consoles.
            process.StartInfo.Environment["PYTHONUTF8"] = "1";
            process.StartInfo.Environment["PYTHONIOENCODING"] = "utf-8";

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            var exited = await Task.Run(() => process.WaitForExit(30000)); // 30 second timeout

            if (!exited)
            {
                try
                {
                    process.Kill(entireProcessTree: true);
                }
                catch
                {
                    // best-effort
                }

                throw new TimeoutException("Python prediction timed out after 30 seconds");
            }

            if (process.ExitCode != 0)
            {
                _logger.LogError($"Python prediction failed: {error}");
                throw new InvalidOperationException($"Prediction script failed: {error}");
            }

            if (!string.IsNullOrWhiteSpace(error))
                _logger.LogDebug($"Python stderr (exit=0): {error}");

            var outputJson = ExtractJsonObjectFromStdout(output);
            var compactJson = CompactJson(outputJson);
            _logger.LogDebug($"Python extracted JSON (compact): {compactJson}");
            try
            {
                using var doc = JsonDocument.Parse(outputJson);
                var root = doc.RootElement;

                if (!root.TryGetProperty("probability", out var probEl) || probEl.ValueKind != JsonValueKind.Number)
                    throw new FormatException("Python output JSON missing numeric 'probability'");

                var probability = probEl.GetDouble();

                int prediction;
                if (root.TryGetProperty("prediction", out var predEl) && predEl.ValueKind == JsonValueKind.Number)
                {
                    prediction = predEl.GetInt32();
                }
                else
                {
                    _logger.LogWarning("Python output JSON missing 'prediction'; deriving from probability and metadata threshold");
                    prediction = probability >= _metadata.Threshold ? 1 : 0;
                }

                double? scriptThreshold = null;
                if (root.TryGetProperty("threshold", out var thrEl) && thrEl.ValueKind == JsonValueKind.Number)
                    scriptThreshold = thrEl.GetDouble();

                int? candlesUsedByPython = null;
                if (root.TryGetProperty("m1_candles_analyzed", out var m1El) && m1El.ValueKind == JsonValueKind.Number)
                    candlesUsedByPython = m1El.GetInt32();

                var entryPrice = TryGetNullableDouble(root, "entry_price");
                var atrM5 = TryGetNullableDouble(root, "atr_m5");
                var stopLoss = TryGetNullableDouble(root, "sl");
                var takeProfit = TryGetNullableDouble(root, "tp");
                var slAtrMultiplier = TryGetNullableDouble(root, "sl_atr_multiplier");
                var tpAtrMultiplier = TryGetNullableDouble(root, "tp_atr_multiplier");
                var rr = TryGetNullableDouble(root, "rr");
                var expectedWinRate = TryGetNullableDouble(root, "expected_win_rate");
                var confidence = TryGetNullableString(root, "confidence");

                return new PythonPrediction(
                    Probability: probability,
                    Prediction: prediction,
                    ScriptThreshold: scriptThreshold,
                    M1CandlesAnalyzed: candlesUsedByPython,
                    EntryPrice: entryPrice,
                    AtrM5: atrM5,
                    StopLoss: stopLoss,
                    TakeProfit: takeProfit,
                    SlAtrMultiplier: slAtrMultiplier,
                    TpAtrMultiplier: tpAtrMultiplier,
                    RiskRewardRatio: rr,
                    ExpectedWinRate: expectedWinRate,
                    Confidence: confidence,
                    RawJson: outputJson,
                    CompactJson: compactJson);
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, $"Failed to parse python output JSON. Stdout: {output}. Stderr: {error}. ExtractedJson: {outputJson}");
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Failed to read python prediction result. Stdout: {output}. Stderr: {error}. ExtractedJson: {outputJson}");
                throw;
            }
        }
        finally
        {
            // Cleanup temp files
            try
            {
                if (File.Exists(tempInputFile)) File.Delete(tempInputFile);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to clean up temporary files");
            }
        }
    }

    /// <summary>
    /// Writes candles to a semicolon-separated CSV file for predict_sequence.py consumption.
    /// </summary>
    private void WriteCandlesToSemicolonCsv(string filePath, List<Candle> candles)
    {
        // predict_sequence.py expects: sep=';' and a 'Date' column.
        using var writer = new StreamWriter(
            filePath,
            append: false,
            encoding: new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        writer.WriteLine("Date;Open;High;Low;Close;Volume");
        foreach (var c in candles)
        {
            var dt = c.Timestamp.ToUniversalTime();
            writer.Write(dt.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
            writer.Write(';');
            writer.Write(c.Open.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(';');
            writer.Write(c.High.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(';');
            writer.Write(c.Low.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(';');
            writer.Write(c.Close.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(';');
            writer.WriteLine(c.Volume.ToString("R", CultureInfo.InvariantCulture));
        }

        _logger.LogDebug($"Wrote {candles.Count} candles to {filePath}");
    }

    private static string ExtractJsonObjectFromStdout(string stdout)
    {
        if (string.IsNullOrWhiteSpace(stdout))
            throw new FormatException("Python stdout is empty; cannot extract JSON");

        var marker = "JSON Output:";
        var idx = stdout.LastIndexOf(marker, StringComparison.OrdinalIgnoreCase);
        var candidate = idx >= 0 ? stdout[(idx + marker.Length)..] : stdout;
        candidate = candidate.Trim();

        // Try to isolate a trailing JSON object.
        var match = Regex.Match(candidate, "\\{[\\s\\S]*\\}\\s*$");
        if (match.Success)
            return match.Value.Trim();

        // Fallback: attempt to parse from last '{' to end.
        var brace = candidate.LastIndexOf('{');
        if (brace >= 0)
            return candidate[brace..].Trim();

        throw new FormatException($"Could not locate JSON object in python stdout. Stdout was: {stdout}");
    }

    private static string CompactJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return JsonSerializer.Serialize(doc.RootElement, new JsonSerializerOptions
        {
            WriteIndented = false
        });
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

    private static string ComputeCandlesFingerprint(List<Candle> candles)
    {
        // Deterministic, lightweight fingerprint to confirm the effective input changed.
        // We hash only the window that is actually used for prediction.
        using var sha = SHA256.Create();

        // Build a stable string with invariant formatting.
        var sb = new StringBuilder(capacity: candles.Count * 64);
        foreach (var c in candles)
        {
            sb.Append(c.Timestamp.ToUniversalTime().Ticks).Append('|')
              .Append(c.Open.ToString("R", CultureInfo.InvariantCulture)).Append('|')
              .Append(c.High.ToString("R", CultureInfo.InvariantCulture)).Append('|')
              .Append(c.Low.ToString("R", CultureInfo.InvariantCulture)).Append('|')
              .Append(c.Close.ToString("R", CultureInfo.InvariantCulture)).Append('|')
              .Append(c.Volume.ToString("R", CultureInfo.InvariantCulture)).Append('\n');
        }

        var bytes = Encoding.UTF8.GetBytes(sb.ToString());
        var hash = sha.ComputeHash(bytes);
        return Convert.ToHexString(hash);
    }

    private void LogCandleStats(string label, List<Candle> candles)
    {
        if (candles.Count == 0)
            return;

        var first = candles[0];
        var last = candles[^1];

        var openMin = candles.Min(c => c.Open);
        var openMax = candles.Max(c => c.Open);
        var highMin = candles.Min(c => c.High);
        var highMax = candles.Max(c => c.High);
        var lowMin = candles.Min(c => c.Low);
        var lowMax = candles.Max(c => c.Low);
        var closeMin = candles.Min(c => c.Close);
        var closeMax = candles.Max(c => c.Close);
        var volMin = candles.Min(c => c.Volume);
        var volMax = candles.Max(c => c.Volume);

        _logger.LogDebug(
            $"Stats[{label}]: count={candles.Count}, " +
            $"first={first.Timestamp:o} O={first.Open.ToString("0.#####", CultureInfo.InvariantCulture)} H={first.High.ToString("0.#####", CultureInfo.InvariantCulture)} L={first.Low.ToString("0.#####", CultureInfo.InvariantCulture)} C={first.Close.ToString("0.#####", CultureInfo.InvariantCulture)} V={first.Volume.ToString("0.#####", CultureInfo.InvariantCulture)}, " +
            $"last={last.Timestamp:o} O={last.Open.ToString("0.#####", CultureInfo.InvariantCulture)} H={last.High.ToString("0.#####", CultureInfo.InvariantCulture)} L={last.Low.ToString("0.#####", CultureInfo.InvariantCulture)} C={last.Close.ToString("0.#####", CultureInfo.InvariantCulture)} V={last.Volume.ToString("0.#####", CultureInfo.InvariantCulture)}, " +
            $"range: O={openMin.ToString("0.#####", CultureInfo.InvariantCulture)}..{openMax.ToString("0.#####", CultureInfo.InvariantCulture)} " +
            $"H={highMin.ToString("0.#####", CultureInfo.InvariantCulture)}..{highMax.ToString("0.#####", CultureInfo.InvariantCulture)} " +
            $"L={lowMin.ToString("0.#####", CultureInfo.InvariantCulture)}..{lowMax.ToString("0.#####", CultureInfo.InvariantCulture)} " +
            $"C={closeMin.ToString("0.#####", CultureInfo.InvariantCulture)}..{closeMax.ToString("0.#####", CultureInfo.InvariantCulture)} " +
            $"V={volMin.ToString("0.#####", CultureInfo.InvariantCulture)}..{volMax.ToString("0.#####", CultureInfo.InvariantCulture)}");
    }
}
