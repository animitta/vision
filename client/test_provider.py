import base64
from datetime import datetime
from signalrclient import SignalRClient

if __name__ == '__main__':
    with open('data/pc.bin', 'rb') as file:
        data = file.read()

    with SignalRClient(None) as client:
        while True:
            message = input('>> 输入e退出, 输入任意其它键继续...')
            if message == 'e':
                break
            content = base64.b64encode(data).decode('utf-8')
            client.send_message('visualize_point_cloud', content)
            print(f'>> 请求渲染:visualize_point_cloud, 参数大小:{len(data)}, 时间:{datetime.now()}')

time.time