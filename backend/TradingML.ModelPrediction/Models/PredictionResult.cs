namespace TradingML.ModelPrediction.Models;

/// <summary>
/// Represents the result of a model prediction.
/// </summary>
public class PredictionResult
{
    /// <summary>
    /// The predicted probability of a BUY signal (0-1).
    /// </summary>
    public double Probability { get; set; }

    /// <summary>
    /// The predicted class (0 for SELL, 1 for BUY).
    /// </summary>
    public int Prediction { get; set; }

    /// <summary>
    /// Whether the prediction meets the confidence threshold.
    /// </summary>
    public bool IsSignal { get; set; }

    /// <summary>
    /// Signal type: "BUY", "SELL", or "NEUTRAL".
    /// </summary>
    public string SignalType { get; set; } = "NEUTRAL";

    /// <summary>
    /// Timestamp of the prediction.
    /// </summary>
    public DateTime PredictionTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Number of candles used for the prediction.
    /// </summary>
    public int CandlesUsed { get; set; }

    /// <summary>
    /// The decision threshold used for the prediction.
    /// </summary>
    public double Threshold { get; set; }
}
