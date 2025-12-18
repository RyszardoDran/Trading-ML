namespace TradingML.ModelPrediction.Models;

/// <summary>
/// Represents a single candle (OHLCV) for a financial instrument.
/// </summary>
public class Candle
{
    /// <summary>
    /// Timestamp of the candle opening.
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Opening price.
    /// </summary>
    public double Open { get; set; }

    /// <summary>
    /// Highest price during the candle.
    /// </summary>
    public double High { get; set; }

    /// <summary>
    /// Lowest price during the candle.
    /// </summary>
    public double Low { get; set; }

    /// <summary>
    /// Closing price.
    /// </summary>
    public double Close { get; set; }

    /// <summary>
    /// Volume traded during the candle.
    /// </summary>
    public double Volume { get; set; }
}
