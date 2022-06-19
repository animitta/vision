using Volo.Abp;
using Volo.Abp.Autofac;
using Volo.Abp.Modularity;
using Volo.Abp.AspNetCore.Mvc;
using Volo.Abp.AspNetCore.Serilog;
using Volo.Abp.AspNetCore.SignalR;
namespace VisionProxy;

[DependsOn(typeof(AbpAutofacModule))]
[DependsOn(typeof(AbpAspNetCoreMvcModule))]
[DependsOn(typeof(AbpAspNetCoreSignalRModule))]
[DependsOn(typeof(AbpAspNetCoreSerilogModule))]
public class VisionProxyModule : AbpModule
{
    public override void ConfigureServices(ServiceConfigurationContext context)
    {
        //var configuration = context.Services.GetConfiguration();
        //var hostingEnvironment = context.Services.GetHostingEnvironment();

        context.Services.AddSignalR(options =>
        {
            // 最大的单个消息10MB
            options.MaximumReceiveMessageSize = 10 * 1024 * 1024;
        });

    }

    public override void OnApplicationInitialization(ApplicationInitializationContext context)
    {
        var app = context.GetApplicationBuilder();
        var env = context.GetEnvironment();

        if (env.IsDevelopment())
        {
            app.UseDeveloperExceptionPage();
        }

        app.UseAbpRequestLocalization();
        app.UseCorrelationId();
        app.UseStaticFiles();
        app.UseRouting();
        app.UseUnitOfWork();

        app.UseAuditing();
        app.UseAbpSerilogEnrichers();
        app.UseConfiguredEndpoints();
    }
}
