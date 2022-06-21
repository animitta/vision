import time
import base64
from datetime import datetime
from signalrclient import SignalRClient

if __name__ == '__main__':
    with open('data/pc.bin', 'rb') as file:
        data = file.read()

    i = 0
    method = 'visualize_point_cloud'
    content = base64.b64encode(data).decode('utf-8')
    with SignalRClient(None) as client:
        while True:
            i = i+1
            print('>> 20秒后将发送下一次渲染请求')
            time.sleep(20)

            print(f'>> 请求渲染:{method}:{i}, 参数大小:{len(content)}, 时间:{datetime.now()}')
            client.send_message(f'{method}:{i}', content)
