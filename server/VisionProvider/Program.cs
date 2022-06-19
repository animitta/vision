using Microsoft.AspNetCore.SignalR.Client;
namespace VisionProvider;

public static class Program
{
    private static HubConnection _connection;

    static Program()
    {
        var endpoint = "http://127.0.0.1:21806/hubs/vision";
        _connection = new HubConnectionBuilder().WithUrl(endpoint).Build();
        _connection.Closed += async (error) =>
        {
            await Task.Delay(new Random().Next(0, 10) * 1000);
            await _connection.StartAsync();
        };
    }

    public static async Task Main()
    {
        AddEventListener();
        try
        {
            await _connection.StartAsync();

            while (true)
            {
                var value = Guid.NewGuid();
                var message = new VisionMessage
                {
                    Method = "Test:" + value.ToString(),
                    Arguments = value.ToByteArray()
                };

                if (_connection.State == HubConnectionState.Connected)
                {
                    await _connection.SendAsync("SendData", message);
                }

                await Task.Delay(5 * 1000);
            }
        }
        catch (Exception ex)
        {
            Console.Write(ex.Message);
        }
        Console.WriteLine("运行结束了...");
    }

    private static void AddEventListener()
    {
        _connection.On<VisionMessage>("ReceiveMessage", (message) =>
        {
            Console.WriteLine($"收到了新消息: 方法:{message.Method}, 参数:${new Guid(message.Arguments)}");
        });
    }
}