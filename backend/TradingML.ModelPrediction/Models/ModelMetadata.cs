namespace TradingML.ModelPrediction.Models;

/// <summary>
/// Metadata about the trained ML model.
/// </summary>
public class ModelMetadata
{
    /// <summary>
    /// List of feature column names in the order expected by the model.
    /// </summary>
    public List<string> FeatureColumns { get; set; } = new();

    /// <summary>
    /// Decision threshold for BUY signal classification.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Expected window size (number of historical candles).
    /// </summary>
    public int WindowSize { get; set; }

    /// <summary>
    /// Number of features per candle.
    /// </summary>
    public int FeaturesPerCandle { get; set; }

    /// <summary>
    /// Total number of features expected by the model.
    /// </summary>
    public int TotalFeatures { get; set; }

    /// <summary>
    /// Minimum recommended number of candles for prediction.
    /// </summary>
    public int RecommendedMinCandles { get; set; }

    /// <summary>
    /// Win rate of the model (for reference).
    /// </summary>
    public double WinRate { get; set; }

    /// <summary>
    /// Feature importance mapping (feature name -> importance value).
    /// </summary>
    public Dictionary<string, double> FeatureImportance { get; set; } = new();
}
