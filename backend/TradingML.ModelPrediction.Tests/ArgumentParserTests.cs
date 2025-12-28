using Xunit;

namespace TradingML.ModelPrediction.Tests
{
    public class ArgumentParserTests
    {
        [Fact]
        public void Parse_SkipRegimeFlag_SetsOption()
        {
            var parser = new ArgumentParser();
            var opts = parser.Parse(new[] { "--skip-regime" });
            Assert.True(opts.SkipRegime);
        }
    }
}
