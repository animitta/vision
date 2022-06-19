using Microsoft.AspNetCore.SignalR;
using Volo.Abp.AspNetCore.SignalR;
namespace VisionProxy;

[HubRoute("/hubs/vision")]
public class VisionHub : AbpHub
{
    public VisionHub()
    {
    }

    public override Task OnConnectedAsync()
    {
        Logger.LogWarning("用户上线:{Time}", DateTime.Now);
        return base.OnConnectedAsync();
    }

    public override Task OnDisconnectedAsync(Exception exception)
    {
        Logger.LogError(exception, "用户离线:{Time}", DateTime.Now);
        return base.OnDisconnectedAsync(exception);
    }

    public Task SendData(VisionMessage message)
    {
        Logger.LogInformation("渲染请求:{Method}, 时间:{Time}, 内容字节:{Size}", message.Method, DateTime.Now, message.Arguments.Length);
        return Clients.All.SendAsync("ReceiveMessage", message);
    }
}
