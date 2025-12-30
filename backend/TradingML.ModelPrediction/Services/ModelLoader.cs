using System.Text.Json;
using TradingML.ModelPrediction.Models;

namespace TradingML.ModelPrediction.Services;

/// <summary>
/// Service for loading and managing ML model artifacts from disk.
/// </summary>
public class ModelLoader
{
    private readonly string _modelsDirectory;
    private readonly ILogger<ModelLoader> _logger;

    public ModelLoader(string modelsDirectory, ILogger<ModelLoader> logger)
    {
        _modelsDirectory = modelsDirectory ?? throw new ArgumentNullException(nameof(modelsDirectory));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Loads model metadata (features, threshold, etc.) from JSON files.
    /// </summary>
    /// <returns>Loaded ModelMetadata with all configuration.</returns>
    /// <exception cref="FileNotFoundException">Thrown if model files are not found.</exception>
    /// <exception cref="JsonException">Thrown if JSON parsing fails.</exception>
    public ModelMetadata LoadModelMetadata()
    {
        _logger.LogInformation($"Loading model metadata from {_modelsDirectory}");

        var featureColumnsPath = Path.Combine(_modelsDirectory, "sequence_feature_columns.json");
        var thresholdPath = Path.Combine(_modelsDirectory, "sequence_threshold.json");
        var featureImportancePath = Path.Combine(_modelsDirectory, "sequence_feature_importance.json");

        if (!File.Exists(featureColumnsPath))
            throw new FileNotFoundException($"Feature columns file not found: {featureColumnsPath}");
        if (!File.Exists(thresholdPath))
            throw new FileNotFoundException($"Threshold file not found: {thresholdPath}");

        try
        {
            // Load feature columns
            var featureColumnsJson = File.ReadAllText(featureColumnsPath);
            var featureColumns = JsonSerializer.Deserialize<List<string>>(featureColumnsJson)
                ?? throw new JsonException("Failed to parse feature columns");

            // Load threshold and metadata
            var thresholdJson = File.ReadAllText(thresholdPath);
            using var thresholdDoc = JsonDocument.Parse(thresholdJson);
            var root = thresholdDoc.RootElement;

            // Validate required and optional properties safely
            var missingRequired = new List<string>();

            if (!root.TryGetProperty("threshold", out var thresholdProp) || thresholdProp.ValueKind == JsonValueKind.Null)
                missingRequired.Add("threshold");
            if (!root.TryGetProperty("window_size", out var windowSizeProp) || windowSizeProp.ValueKind == JsonValueKind.Null)
                missingRequired.Add("window_size");
            if (!root.TryGetProperty("n_features_per_candle", out var featuresPerCandleProp) || featuresPerCandleProp.ValueKind == JsonValueKind.Null)
                missingRequired.Add("n_features_per_candle");
            if (!root.TryGetProperty("total_features", out var totalFeaturesProp) || totalFeaturesProp.ValueKind == JsonValueKind.Null)
                missingRequired.Add("total_features");

            if (missingRequired.Any())
            {
                var msg = $"Missing required model metadata properties: {string.Join(", ", missingRequired)}";
                _logger.LogError(msg);
                throw new JsonException(msg);
            }

            var threshold = thresholdProp.GetDouble();
            var windowSize = windowSizeProp.GetInt32();
            var featuresPerCandle = featuresPerCandleProp.GetInt32();
            var totalFeatures = totalFeaturesProp.GetInt32();

            // Optional properties - log and provide sensible defaults when absent
            int recommendedMinCandles = 0;
            if (root.TryGetProperty("recommended_min_candles", out var recommendedProp) && recommendedProp.ValueKind != JsonValueKind.Null)
            {
                recommendedMinCandles = recommendedProp.GetInt32();
            }
            else
            {
                _logger.LogWarning("Optional property 'recommended_min_candles' not found in threshold JSON; defaulting to 0");
            }

            double winRate = 0.0;
            if (root.TryGetProperty("win_rate", out var winRateProp) && winRateProp.ValueKind != JsonValueKind.Null)
            {
                winRate = winRateProp.GetDouble();
            }
            else
            {
                _logger.LogWarning("Optional property 'win_rate' not found in threshold JSON; defaulting to 0.0");
            }

            // Load feature importance if available
            var featureImportance = new Dictionary<string, double>();
            if (File.Exists(featureImportancePath))
            {
                try
                {
                    var importanceJson = File.ReadAllText(featureImportancePath);
                    using var importanceDoc = JsonDocument.Parse(importanceJson);
                    foreach (var prop in importanceDoc.RootElement.EnumerateObject())
                    {
                        if (prop.Value.TryGetDouble(out var value))
                        {
                            featureImportance[prop.Name] = value;
                        }
                    }
                    _logger.LogInformation($"Loaded {featureImportance.Count} feature importance values");
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to load feature importance, continuing without it");
                }
            }

            var metadata = new ModelMetadata
            {
                FeatureColumns = featureColumns,
                Threshold = threshold,
                WindowSize = windowSize,
                FeaturesPerCandle = featuresPerCandle,
                TotalFeatures = totalFeatures,
                RecommendedMinCandles = recommendedMinCandles,
                WinRate = winRate,
                FeatureImportance = featureImportance
            };

            _logger.LogInformation($"Model metadata loaded: {totalFeatures} features, threshold={threshold}, window={windowSize}");

            return metadata;
        }
        catch (JsonException ex)
        {
            _logger.LogError(ex, "Failed to parse JSON model files");
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error loading model metadata");
            throw;
        }
    }

    /// <summary>
    /// Loads the trained XGBoost model from pickle file.
    /// Note: In production, you would need to call Python via IPC or use ONNX format.
    /// For now, this returns a placeholder. You'll need XGBoost.NET or ONNX Runtime.
    /// </summary>
    /// <returns>Path to the model file for later use.</returns>
    public string LoadModelPath()
    {
        var modelPath = Path.Combine(_modelsDirectory, "sequence_xgb_model.pkl");
        if (!File.Exists(modelPath))
        {
            var scalerPath = Path.Combine(_modelsDirectory, "sequence_scaler.pkl");
            _logger.LogWarning($"Model file not found at {modelPath}. Checked for scaler at {scalerPath}");
            throw new FileNotFoundException($"XGBoost model not found at {modelPath}");
        }

        _logger.LogInformation($"Model file found at {modelPath}");
        return modelPath;
    }

    /// <summary>
    /// Validates that model artifacts exist and are accessible.
    /// </summary>
    /// <returns>True if all required files exist.</returns>
    public bool ValidateModelArtifacts()
    {
        var requiredFiles = new[]
        {
            "sequence_feature_columns.json",
            "sequence_threshold.json",
            "sequence_xgb_model.pkl"
        };

        var missingFiles = requiredFiles
            .Where(f => !File.Exists(Path.Combine(_modelsDirectory, f)))
            .ToList();

        if (missingFiles.Any())
        {
            _logger.LogError($"Missing model artifacts: {string.Join(", ", missingFiles)}");
            return false;
        }

        _logger.LogInformation("All model artifacts validated successfully");
        return true;
    }
}
