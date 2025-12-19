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
    /// Number of candles provided to the predictor (pre-filtering/selection).
    /// </summary>
    public int CandlesProvided { get; set; }

    /// <summary>
    /// Python's default effective input window (typically last N days) candle count.
    /// </summary>
    public int? EffectiveInputCandlesCount { get; set; }

    /// <summary>
    /// Python's default effective input window start time (UTC).
    /// </summary>
    public DateTime? EffectiveInputFromUtc { get; set; }

    /// <summary>
    /// Python's default effective input window end time (UTC).
    /// </summary>
    public DateTime? EffectiveInputToUtc { get; set; }

    /// <summary>
    /// Fingerprint of the effective input window to make changes obvious.
    /// </summary>
    public string? EffectiveInputFingerprint { get; set; }

    /// <summary>
    /// Entry price used by the Python script (typically last M5 close).
    /// </summary>
    public double? EntryPrice { get; set; }

    /// <summary>
    /// ATR computed on M5 candles by the Python pipeline (may be null if not available).
    /// </summary>
    public double? AtrM5 { get; set; }

    /// <summary>
    /// Stop-loss price returned by the Python pipeline (may be null for no-trade / filtered cases).
    /// </summary>
    public double? StopLoss { get; set; }

    /// <summary>
    /// Take-profit price returned by the Python pipeline (may be null for no-trade / filtered cases).
    /// </summary>
    public double? TakeProfit { get; set; }

    /// <summary>
    /// SL ATR multiplier used by Python (expected to be consistent with risk configuration).
    /// </summary>
    public double? SlAtrMultiplier { get; set; }

    /// <summary>
    /// TP ATR multiplier used by Python (expected to be consistent with risk configuration).
    /// </summary>
    public double? TpAtrMultiplier { get; set; }

    /// <summary>
    /// Risk/reward ratio (TP distance / SL distance).
    /// </summary>
    public double? RiskRewardRatio { get; set; }

    /// <summary>
    /// Expected win rate returned by Python (training/test estimate).
    /// </summary>
    public double? ExpectedWinRate { get; set; }

    /// <summary>
    /// Confidence label returned by the Python pipeline (e.g. high/medium/low or filter reason).
    /// </summary>
    public string? Confidence { get; set; }

    /// <summary>
    /// Optional compact JSON returned by the Python predictor (single line).
    /// </summary>
    public string? PythonOutputJson { get; set; }

    /// <summary>
    /// The decision threshold used for the prediction.
    /// </summary>
    public double Threshold { get; set; }
}
