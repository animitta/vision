using Serilog;
using Serilog.Events;
namespace VisionProxy;

public class Program
{
    public async static Task<int> Main(string[] args)
    {
        var loggerConfiguration = new LoggerConfiguration()
#if DEBUG
            .MinimumLevel.Verbose()
#else
            .MinimumLevel.Information()
#endif
            .MinimumLevel.Override("Microsoft", LogEventLevel.Information)
            .MinimumLevel.Override("Microsoft.EntityFrameworkCore", LogEventLevel.Warning)
            .Enrich.FromLogContext()
            .WriteTo.Async(c => c.File("logs/logs.txt"))
            .WriteTo.Async(c => c.Console());

        Log.Logger = loggerConfiguration.CreateLogger();
        try
        {
            var builder = WebApplication.CreateBuilder(args);
            builder.Host.AddAppSettingsSecretsJson()
                .UseAutofac()
                .UseSerilog();
            await builder.AddApplicationAsync<VisionProxyModule>();
            var app = builder.Build();
            await app.InitializeApplicationAsync();

            Log.Information("Starting VisionProxy.");
            await app.RunAsync();
            return 0;
        }
        catch (Exception ex)
        {
            if (ex.GetType().Name.Equals("StopTheHostException", StringComparison.Ordinal))
            {
                throw;
            }

            Log.Fatal(ex, "VisionProxy terminated unexpectedly!");
            return 1;
        }
        finally
        {
            Log.CloseAndFlush();
        }
    }
}
