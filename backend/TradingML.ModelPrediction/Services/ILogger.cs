namespace TradingML.ModelPrediction.Services;

/// <summary>
/// Logger interface stub for services.
/// In a real project, this would be injected from Microsoft.Extensions.Logging
/// </summary>
public interface ILogger<T>
{
    void LogInformation(string message, params object[] args);
    void LogWarning(string message, params object[] args);
    void LogWarning(Exception ex, string message, params object[] args);
    void LogError(string message, params object[] args);
    void LogError(Exception ex, string message, params object[] args);
    void LogDebug(string message, params object[] args);
}

/// <summary>
/// Default console logger implementation.
/// </summary>
public class ConsoleLogger<T> : ILogger<T>
{
    private readonly string _category = typeof(T).Name;

    public void LogInformation(string message, params object[] args)
    {
        var formattedMessage = args.Length > 0 ? string.Format(message, args) : message;
        Console.WriteLine($"[INFO] [{_category}] {formattedMessage}");
    }

    public void LogWarning(string message, params object[] args)
    {
        var formattedMessage = args.Length > 0 ? string.Format(message, args) : message;
        Console.WriteLine($"[WARN] [{_category}] {formattedMessage}");
    }

    public void LogWarning(Exception ex, string message, params object[] args)
    {
        var formattedMessage = args.Length > 0 ? string.Format(message, args) : message;
        Console.WriteLine($"[WARN] [{_category}] {formattedMessage}");
        Console.WriteLine($"       Exception: {ex.Message}");
    }

    public void LogError(string message, params object[] args)
    {
        var formattedMessage = args.Length > 0 ? string.Format(message, args) : message;
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"[ERROR] [{_category}] {formattedMessage}");
        Console.ResetColor();
    }

    public void LogError(Exception ex, string message, params object[] args)
    {
        var formattedMessage = args.Length > 0 ? string.Format(message, args) : message;
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"[ERROR] [{_category}] {formattedMessage}");
        Console.WriteLine($"        Exception: {ex.Message}");
        Console.ResetColor();
    }

    public void LogDebug(string message, params object[] args)
    {
        var formattedMessage = args.Length > 0 ? string.Format(message, args) : message;
        Console.WriteLine($"[DEBUG] [{_category}] {formattedMessage}");
    }
}
