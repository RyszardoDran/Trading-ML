using TradingML.ModelPrediction.Models;

namespace TradingML.ModelPrediction.Services;

/// <summary>
/// Service for parsing candle data from CSV/JSON input.
/// </summary>
public class CandleParser
{
    private readonly ILogger<CandleParser> _logger;

    public CandleParser(ILogger<CandleParser> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Parses candles from a CSV file (OHLCV format).
    /// Expected columns: Timestamp/Date, Open, High, Low, Close, Volume
    /// Supports both comma and semicolon delimiters
    /// </summary>
    public List<Candle> ParseFromCsv(string csvPath)
    {
        if (!File.Exists(csvPath))
            throw new FileNotFoundException($"CSV file not found: {csvPath}");

        var candles = new List<Candle>();

        try
        {
            var lines = File.ReadAllLines(csvPath);
            if (lines.Length < 2)
            {
                _logger.LogWarning("CSV file has no data rows");
                return candles;
            }

            // Detect delimiter (comma or semicolon)
            var delimiter = lines[0].Contains(';') ? ';' : ',';

            // Skip header
            for (int i = 1; i < lines.Length; i++)
            {
                var parts = lines[i].Split(delimiter);
                if (parts.Length < 6)
                    continue;

                try
                {
                    var candle = new Candle
                    {
                        Timestamp = DateTime.ParseExact(parts[0].Trim(), "yyyy.MM.dd HH:mm", System.Globalization.CultureInfo.InvariantCulture),
                        Open = double.Parse(parts[1].Trim()),
                        High = double.Parse(parts[2].Trim()),
                        Low = double.Parse(parts[3].Trim()),
                        Close = double.Parse(parts[4].Trim()),
                        Volume = double.Parse(parts[5].Trim())
                    };
                    candles.Add(candle);
                }
                catch (FormatException)
                {
                    // Try standard DateTime format as fallback
                    try
                    {
                        var candle = new Candle
                        {
                            Timestamp = DateTime.Parse(parts[0].Trim()),
                            Open = double.Parse(parts[1].Trim()),
                            High = double.Parse(parts[2].Trim()),
                            Low = double.Parse(parts[3].Trim()),
                            Close = double.Parse(parts[4].Trim()),
                            Volume = double.Parse(parts[5].Trim())
                        };
                        candles.Add(candle);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning($"Failed to parse line {i}: {lines[i]}");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning($"Failed to parse line {i}: {lines[i]}");
                }
            }

            _logger.LogInformation($"Parsed {candles.Count} candles from CSV");
            return candles;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading CSV file");
            throw;
        }
    }

    /// <summary>
    /// Parses candles from a list of OHLCV tuples.
    /// </summary>
    public List<Candle> ParseFromOhlcv(List<(DateTime timestamp, double open, double high, double low, double close, double volume)> ohlcvData)
    {
        return ohlcvData
            .Select(x => new Candle
            {
                Timestamp = x.timestamp,
                Open = x.open,
                High = x.high,
                Low = x.low,
                Close = x.close,
                Volume = x.volume
            })
            .ToList();
    }

    /// <summary>
    /// Validates candle data quality.
    /// </summary>
    public bool ValidateCandles(List<Candle> candles, int minimumRequired = 260)
    {
        if (candles == null || candles.Count == 0)
        {
            _logger.LogError("No candles provided for validation");
            return false;
        }

        if (candles.Count < minimumRequired)
        {
            _logger.LogError($"Insufficient candles: {candles.Count} < {minimumRequired}");
            return false;
        }

        // Check for invalid OHLC relationships
        var invalidCount = candles.Count(c => c.High < c.Low || c.Open < 0 || c.Close < 0);
        if (invalidCount > 0)
        {
            _logger.LogError($"Found {invalidCount} candles with invalid OHLC values");
            return false;
        }

        _logger.LogInformation($"Candle validation passed: {candles.Count} candles");
        return true;
    }
}
