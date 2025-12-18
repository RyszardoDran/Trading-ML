using Xunit;
using TradingML.ModelPrediction.Models;
using TradingML.ModelPrediction.Services;

namespace TradingML.ModelPrediction.Tests;

/// <summary>
/// Unit tests for CandleParser service.
/// </summary>
public class CandleParserTests
{
    private readonly CandleParser _parser;
    private readonly ConsoleLogger<CandleParser> _logger;

    public CandleParserTests()
    {
        _logger = new ConsoleLogger<CandleParser>();
        _parser = new CandleParser(_logger);
    }

    [Fact]
    public void ParseFromOhlcv_WithValidData_ReturnsCandleList()
    {
        // Arrange
        var ohlcvData = new List<(DateTime timestamp, double open, double high, double low, double close, double volume)>
        {
            (DateTime.UtcNow, 2000, 2010, 1990, 2005, 100000),
            (DateTime.UtcNow.AddMinutes(1), 2005, 2015, 1995, 2008, 110000)
        };

        // Act
        var candles = _parser.ParseFromOhlcv(ohlcvData);

        // Assert
        Assert.NotNull(candles);
        Assert.Equal(2, candles.Count);
        Assert.Equal(2000, candles[0].Open);
        Assert.Equal(2005, candles[0].Close);
        Assert.Equal(100000, candles[0].Volume);
    }

    [Fact]
    public void ValidateCandles_WithInsufficientCandles_ReturnsFalse()
    {
        // Arrange
        var candles = new List<Candle>
        {
            new() { Timestamp = DateTime.UtcNow, Open = 2000, High = 2010, Low = 1990, Close = 2005, Volume = 100000 }
        };

        // Act
        var isValid = _parser.ValidateCandles(candles, minimumRequired: 260);

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public void ValidateCandles_WithValidCandles_ReturnsTrue()
    {
        // Arrange
        var candles = GenerateSampleCandles(300);

        // Act
        var isValid = _parser.ValidateCandles(candles, minimumRequired: 260);

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public void ValidateCandles_WithInvalidOHLC_ReturnsFalse()
    {
        // Arrange
        var candles = new List<Candle>();
        for (int i = 0; i < 300; i++)
        {
            if (i == 100)
            {
                // Invalid: High < Low
                candles.Add(new Candle
                {
                    Timestamp = DateTime.UtcNow.AddMinutes(i),
                    Open = 2000,
                    High = 1990,  // Invalid: higher should be higher than low
                    Low = 2010,
                    Close = 2005,
                    Volume = 100000
                });
            }
            else
            {
                candles.Add(new Candle
                {
                    Timestamp = DateTime.UtcNow.AddMinutes(i),
                    Open = 2000 + i * 0.01,
                    High = 2010 + i * 0.01,
                    Low = 1990 + i * 0.01,
                    Close = 2005 + i * 0.01,
                    Volume = 100000
                });
            }
        }

        // Act
        var isValid = _parser.ValidateCandles(candles, minimumRequired: 260);

        // Assert
        Assert.False(isValid);
    }

    private List<Candle> GenerateSampleCandles(int count)
    {
        var candles = new List<Candle>();
        for (int i = 0; i < count; i++)
        {
            candles.Add(new Candle
            {
                Timestamp = DateTime.UtcNow.AddMinutes(i),
                Open = 2000 + i * 0.01,
                High = 2010 + i * 0.01,
                Low = 1990 + i * 0.01,
                Close = 2005 + i * 0.01,
                Volume = 100000 + i * 10
            });
        }
        return candles;
    }
}
