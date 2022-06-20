import logging
from signalrcore.hub_connection_builder import HubConnectionBuilder
# from signalrcore.protocol.messagepack_protocol import MessagePackHubProtocol

class SignalRClient:
    def __init__(self, handler, url=None):
        url = url or 'http://127.0.0.1:21806/hubs/vision'  # wss

        # .with_hub_protocol(MessagePackHubProtocol())\
        connection = HubConnectionBuilder()\
            .with_url(url)\
            .configure_logging(logging.INFO)\
            .with_automatic_reconnect({
                "type": "raw",
                "keep_alive_interval": 10,
                "reconnect_interval": 5,
                "max_attempts": 5
            }).build()
        connection.on("ReceiveMessage", handler)
        self.connection = connection

    def __enter__(self):
        self.connection.on_open(self.on_opened)
        self.connection.on_close(self.on_close)
        self.connection.start()
        return self

    def __exit__(self, types, values, trace):
        if self.connection:
            self.connection.stop()

    def on_opened():
        print(">> SignalR 连接成功!")

    def on_close():
        print(">> SignalR 连接关闭!")

    def send_message(self, method, arguments):
        self.connection.send("SendData", [method, arguments])
