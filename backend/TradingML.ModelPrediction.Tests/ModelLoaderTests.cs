using Xunit;
using TradingML.ModelPrediction.Models;
using TradingML.ModelPrediction.Services;

namespace TradingML.ModelPrediction.Tests;

/// <summary>
/// Unit tests for ModelLoader service.
/// </summary>
public class ModelLoaderTests
{
    private readonly string _modelsDirectory;
    private readonly ModelLoader _loader;

    public ModelLoaderTests()
    {
        // Use the actual models directory from the ML project
        _modelsDirectory = @"c:\Users\Arek\Documents\Repos\Traiding\Trading-ML\ml\outputs\models";
        _loader = new ModelLoader(_modelsDirectory, new ConsoleLogger<ModelLoader>());
    }

    [Fact]
    public void ValidateModelArtifacts_WithValidModels_ReturnsTrue()
    {
        // Act & Assert
        if (!Directory.Exists(_modelsDirectory))
        {
            return; // Skip test if models directory doesn't exist
        }

        var result = _loader.ValidateModelArtifacts();
        Assert.True(result, "Model artifacts should be valid");
    }

    [Fact]
    public void LoadModelMetadata_WithValidFiles_ReturnsMetadata()
    {
        // Arrange & Act & Assert
        if (!Directory.Exists(_modelsDirectory))
        {
            return; // Skip test if models directory doesn't exist
        }

        var metadata = _loader.LoadModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.NotEmpty(metadata.FeatureColumns);
        Assert.True(metadata.Threshold > 0);
        Assert.True(metadata.WindowSize > 0);
        Assert.True(metadata.TotalFeatures > 0);
    }

    [Fact]
    public void LoadModelPath_WithExistingModel_ReturnsValidPath()
    {
        // Arrange & Act & Assert
        if (!Directory.Exists(_modelsDirectory))
        {
            return; // Skip test if models directory doesn't exist
        }

        var path = _loader.LoadModelPath();

        // Assert
        Assert.NotEmpty(path);
        Assert.EndsWith(".pkl", path);
        Assert.True(File.Exists(path), "Model file should exist");
    }

    [Fact]
    public void Constructor_WithNullDirectory_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ModelLoader(null!, new ConsoleLogger<ModelLoader>())
        );
    }

    [Fact]
    public void Constructor_WithNullLogger_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ModelLoader(_modelsDirectory, null!)
        );
    }
}
