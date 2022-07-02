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
        time.sleep(20)
        print('>> 20秒后将发送第一次渲染请求')
        while True:
            i = i+1
            print(f'>> 请求渲染:{method}:{i}, 参数大小:{len(content)}, 时间:{datetime.now()}')
            client.send_message(f'{method}:{i}', content)

            input(">> 输入任意键继续发送下一次消息")

# signalrcore socket(封装)
