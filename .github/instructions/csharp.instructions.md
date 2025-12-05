---
applyTo: "**/*.cs"
---

<!-- The above section is called 'frontmatter' and is used to define metadata for the document -->
<!-- The main content of the markdown file starts here -->

# C#/.NET Backend Development Guidelines

## General Guidelines

Follow idiomatic C# practices and Microsoft C# coding conventions. Prioritize writing clean, maintainable, and secure code suitable for a production trading platform.

- **Consistency**: Adhere to the existing code style and patterns in the project.
- **Security**: Implement security best practices, including input validation, parameterized queries, secure authentication/authorization, and protection against injection attacks.
- **Error Handling**: Implement robust error handling with structured logging to ensure system stability and ease of debugging.
- **Configuration Management**: Externalize configuration from code using `appsettings.json` and environment-specific variants. Do not commit secrets to version control.
- **Testing**: Write unit tests for business logic and integration tests for critical paths. Aim for high test coverage, especially on new code.
- **Async/Await**: Use `async` and `await` for non-blocking I/O operations to maximize performance and responsiveness.

## ASP.NET Core Web API Standards

### Project Structure

Follow the standard .NET project structure:

```
src/
├── YourProject.Api/                 # Entry point (Program.cs)
├── YourProject.Application/         # Business logic, DTOs, services
├── YourProject.Domain/              # Domain models and interfaces
├── YourProject.Infrastructure/      # Database, external service integrations
└── YourProject.Persistence/         # Entity Framework Core configuration

tests/
├── YourProject.UnitTests/
├── YourProject.IntegrationTests/
└── YourProject.FunctionalTests/
```

### Configuration Management

1. **appsettings.json**: Contains default configuration.
2. **appsettings.{Environment}.json**: Environment-specific overrides (Development, Staging, Production).
3. **User Secrets**: Use `dotnet user-secrets` for local development secrets (database credentials, API keys).
4. **Environment Variables**: Production secrets managed via CI/CD or cloud providers (Azure Key Vault, AWS Secrets Manager).

Example configuration structure:
```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=...;Database=...;"
  },
  "Logging": {
    "LogLevel": {
      "Default": "Information"
    }
  },
  "Trading": {
    "ApiBaseUrl": "https://api.example.com",
    "MaxRetries": 3,
    "TimeoutSeconds": 30
  }
}
```

### Dependency Injection

1. **Use Built-in DI Container**: Configure dependencies in `Program.cs` using extension methods.
2. **Constructor Injection**: Inject dependencies through constructors, never use service locator pattern.
3. **Interface-Based Design**: Define interfaces for all services to enable testability.

Example:
```csharp
// Program.cs
services.AddScoped<ITradingService, TradingService>();
services.AddScoped<IStockDataRepository, StockDataRepository>();
services.AddHttpClient<IExternalApiClient, ExternalApiClient>();
```

### Database Access with Entity Framework Core

1. **DbContext**: Create a dedicated DbContext inheriting from `DbContext`.
2. **Repositories Pattern**: Use repository pattern for data access abstraction.
3. **Migrations**: Use `dotnet ef migrations` for schema management.
4. **Query Optimization**: Use `.AsNoTracking()` for read-only queries; use `.Include()` for eager loading.

Example:
```csharp
public class TradingDbContext : DbContext
{
    public DbSet<Stock> Stocks { get; set; }
    public DbSet<PriceHistory> PriceHistories { get; set; }
    
    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // Configure fluent API
        modelBuilder.Entity<Stock>()
            .HasMany(s => s.PriceHistories)
            .WithOne(p => p.Stock)
            .HasForeignKey(p => p.StockId);
    }
}
```

### API Endpoint Design

1. **RESTful Conventions**: Follow REST principles for endpoint design.
2. **Route Organization**: Group related endpoints using `[ApiController]` and `[Route("api/[controller]")]`.
3. **HTTP Status Codes**: Return appropriate status codes (200, 201, 400, 401, 404, 500, etc.).
4. **Request/Response DTOs**: Use DTOs to decouple API contracts from domain models.

Example:
```csharp
[ApiController]
[Route("api/[controller]")]
public class StocksController : ControllerBase
{
    private readonly ITradingService _tradingService;
    
    public StocksController(ITradingService tradingService)
    {
        _tradingService = tradingService;
    }
    
    [HttpGet("{id}")]
    public async Task<ActionResult<StockDto>> GetStock(int id)
    {
        var stock = await _tradingService.GetStockAsync(id);
        if (stock == null)
            return NotFound();
        
        return Ok(new StockDto { Id = stock.Id, Symbol = stock.Symbol });
    }
    
    [HttpPost]
    public async Task<ActionResult<StockDto>> CreateStock(CreateStockRequest request)
    {
        var stock = await _tradingService.CreateStockAsync(request.Symbol);
        return CreatedAtAction(nameof(GetStock), new { id = stock.Id }, stock);
    }
}
```

### Authentication & Authorization

1. **JWT Tokens**: Use JWT for API authentication.
2. **Authorization Policies**: Define custom authorization policies for role-based access control.
3. **Secure Headers**: Implement security headers (CORS, CSP, HSTS).

Example:
```csharp
// Program.cs
services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.Authority = configuration["Auth:Authority"];
        options.Audience = configuration["Auth:Audience"];
    });

services.AddAuthorization(options =>
{
    options.AddPolicy("Admin", policy => 
        policy.RequireRole("Admin"));
});
```

<a name="csharp-error-handling"></a>

## Error Handling & Logging

1. **Exception Handling**: Use try-catch for recoverable errors; use middleware for global error handling.
2. **Structured Logging**: Use structured logging (Serilog) with context information.
3. **Problem Details**: Return problem details (RFC 7807) for API errors.

Example:
```csharp
// Middleware for global exception handling
app.UseExceptionHandler(errorApp =>
{
    errorApp.Run(async context =>
    {
        var exception = context.Features.Get<IExceptionHandlerFeature>();
        var problemDetails = new ProblemDetails
        {
            Title = "An error occurred",
            Status = StatusCodes.Status500InternalServerError,
            Detail = exception?.Error.Message
        };
        
        context.Response.StatusCode = StatusCodes.Status500InternalServerError;
        await context.Response.WriteAsJsonAsync(problemDetails);
    });
});
```

## Naming Conventions

- **Classes**: PascalCase (e.g., `StockService`, `PriceCalculator`)
- **Methods**: PascalCase (e.g., `GetStockAsync`, `CalculateMovingAverage`)
- **Properties**: PascalCase (e.g., `CurrentPrice`, `LastUpdated`)
- **Local Variables**: camelCase (e.g., `currentPrice`, `stockList`)
- **Constants**: UPPER_SNAKE_CASE or PascalCase (e.g., `MAX_RETRIES` or `DefaultTimeout`)
- **Interfaces**: Start with 'I' (e.g., `IStockService`, `IPriceRepository`)
- **Private Fields**: _camelCase or m_camelCase (e.g., `_logger`, `_dbContext`)

## Testing Standards

### Unit Tests

- Test business logic in service classes
- Use a mocking framework (Moq, NSubstitute) to isolate dependencies
- Follow Arrange-Act-Assert pattern
- Test both happy path and error scenarios

Example:
```csharp
[Fact]
public async Task GetStock_WithValidId_ReturnsStock()
{
    // Arrange
    var mockRepository = new Mock<IStockRepository>();
    var stock = new Stock { Id = 1, Symbol = "AAPL" };
    mockRepository.Setup(r => r.GetByIdAsync(1))
        .ReturnsAsync(stock);
    
    var service = new StockService(mockRepository.Object);
    
    // Act
    var result = await service.GetStockAsync(1);
    
    // Assert
    Assert.NotNull(result);
    Assert.Equal("AAPL", result.Symbol);
}
```

### Integration Tests

- Test interactions between layers (API → Service → Repository → Database)
- Use in-memory or test database
- Test API endpoints with real request/response flow

Example:
```csharp
[Fact]
public async Task GetStock_Endpoint_ReturnsOk()
{
    // Arrange
    using var client = _factory.CreateClient();
    
    // Act
    var response = await client.GetAsync("/api/stocks/1");
    
    // Assert
    Assert.Equal(System.Net.HttpStatusCode.OK, response.StatusCode);
}
```

### Test Organization

- One test file per class being tested
- Use nested classes or separate files for related test groups
- Descriptive test names: `MethodName_Condition_ExpectedResult`

## Performance Considerations

1. **Async All the Way**: Make database calls, API calls, and I/O operations asynchronous.
2. **Caching**: Implement caching for frequently accessed data (trading prices, reference data).
3. **Database Indexing**: Ensure appropriate indexes on frequently queried columns.
4. **Pagination**: Implement pagination for large result sets.
5. **Connection Pooling**: Configure appropriate connection pool sizes.

## Security Checklist

- [ ] Input validation on all user inputs and API parameters
- [ ] SQL injection prevention (parameterized queries via EF Core)
- [ ] XSS protection (sanitize outputs)
- [ ] CSRF protection
- [ ] Secure authentication and authorization
- [ ] Secrets management (never hardcode credentials)
- [ ] HTTPS enforcement
- [ ] Rate limiting
- [ ] Audit logging for sensitive operations
- [ ] Dependency vulnerability scanning

<!-- © Capgemini 2025 -->
