import time
import torch
import base64
import numpy as np

from io import BytesIO
from datetime import datetime
from msilib.schema import Error
from signalr_client import SignalRClient

client = None
send_count = 0


def start():
    global client
    client = SignalRClient(None)
    client.start()


def close():
    if client != None:
        client.close()


def send(method=None, **args):
    global send_count

    if client == None:
        raise Error('模块未调用start初始化')

    if not method:
        method = 'visualize_point_cloud'

    if len(args) == 0:
        raise Error('渲染需要至少一个有效的参数')

    # 对数据进行打包
    buffer = BytesIO()
    torch.save(args, buffer)
    content = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 记录日志

    send_count += 1
    print(f'>> 请求渲染: [{send_count}] - [{datetime.now()}]: 参数大小: {len(content)}')

    # 发送请求
    client.send_message(f'{method}', content)


if __name__ == '__main__':
    point_cloud = np.fromfile('data/pc.bin', dtype=np.float32).reshape(-1, 4)
    point_cloud = point_cloud[:, :3]
    point_cloud = point_cloud[:3000]

    color = np.repeat(np.array([0.8, 0.1, 0.1]), 3000, 0)

    start()

    time.sleep(5)
    send('visualize_point_cloud', point_cloud=point_cloud, color = color)
    time.sleep(10)

    close()
