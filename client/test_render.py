import time
import base64
import queue
import numpy as np

from datetime import datetime
from signalrclient import SignalRClient
from multiprocessing import Process, Manager

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mayavi import mlab
from utils import draw_bbox, draw_colored_pointcloud

def visualize_point_cloud(point_cloud, color=None, gt_3dbox=None, pred_3dbox=None, scores=None, labels=None, point_size=0.05, save_name=None):
    # point_cloud: (N, 3), color: (N, 3), gt_3dbox: (7, ), pred_3dbox: (7, )
    # scores:
    if save_name is not None:
        mlab.options.offscreen = True
    else:
        mlab.options.offscreen = False

    fig = draw_colored_pointcloud(point_cloud, colors=color, point_size=point_size)
    if gt_3dbox is not None:
        draw_bbox(fig, gt_boxes=gt_3dbox.unsqueeze(0))
    if pred_3dbox is not None:
        draw_bbox(fig, ref_boxes=pred_3dbox.unsqueeze(0), ref_scores=scores, ref_labels=labels)

    # set default view point
    mlab.view(azimuth=-179, elevation=70.0, distance=85.0,
              roll=90.0, focalpoint=(35, 0, 0))

    if save_name is not None:
        mlab.savefig(save_name)
        mlab.close()
    else:
        return mlab


def display_plotting(fig, azimuth=-179, elevation=70.0, distance=85.0,
                     roll=90.0, focalpoint=(35, 0, 0)):
    mlab.view(azimuth=azimuth,
              elevation=elevation,
              distance=distance,
              focalpoint=focalpoint,
              roll=roll)
    mlab.show(stop=True)


def visualize_object_3d(sub_cloud,
                        image,
                        sub_cloud2d,
                        gt_3dbox=None,
                        pred_3dbox=None,
                        pred_scores=None,
                        pred_class=None,
                        point_size=0.05):
    # sub_cloud: (N, 4), image(C, H, W), sub_cloud2d: (N, 2)
    # gt_3dbox: [x, y, z, dx, dy, dz, heading]
    # pred_3dbox: [x, y, z, dx, dy, dz, heading]
    x, y = sub_cloud2d[:, 0], sub_cloud2d[:, 1]
    color = image[:, y, x]*255
    color = color.transpose(0, 1).numpy().astype(np.uint8)
    fig = visualize_point_cloud(sub_cloud[:, :3], color, gt_3dbox, pred_3dbox, pred_scores, pred_class, point_size)
    return fig


def visualize_image(image, box2d=None, mask=None, depth_points=None, alpha=0.5):
    # image: (3, H, W); box2d: [l, t, r, b]; mask: (1, Hm, Wm); depth_points: (N, 3) - X, Y, Depth
    img = torchvision.transforms.ToPILImage()(image)
    fig, ax = plt.subplots()
    ax.imshow(img)
    if box2d is not None:
        left, top, right, bottom = box2d
        width = right - left
        height = bottom - top
        rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if mask is not None:
        mask = torch.nn.Upsample(image.size()[-2:], mode='bilinear')(mask.unsqueeze(0)).squeeze(0)
        mask = torchvision.transforms.ToPILImage()(mask)
        ax.imshow(mask, cmap='viridis', alpha=alpha, interpolation='bilinear')  # interpolation='none'

    if depth_points is not None:
        x, y, depth = depth_points[:, 0], depth_points[:, 1], depth_points[:, 2]
        depth = -depth
        ax.scatter(x=x, y=y, c=depth, cmap='spring', s=4)

    plt.show()


def inv_norm_image(image):
    invTrans = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0, 0, 0), (1/0.229, 1/0.224, 1/0.225)),
        torchvision.transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1)),
    ])
    return invTrans(image)


# 这个进程帮我们接受数据, 是数据的提供者
def start_signalr(queue):
    def handle(package):
        recvied_time = datetime.now()
        if queue.full():
            print(f'[signalr] - [{recvied_time}]: 等待原来的消息被消费, 当前接受数据被抛弃')
        else:
            print(f'[signalr] - [{recvied_time}]: 消费队列未蛮, 正在压入数据,等待渲染线程获取数据...')
            queue.put_nowait(package[0])
            queue.put_nowait(package[1])

    print('[signalr]: 启动通讯监听')
    with SignalRClient(handle):
        while True:
            time.sleep(1)


def render_vision():
    method = queue.get(block=True)
    arguments = queue.get(block=True)
    print('>> 正在启动一个新的mayavi窗口')

    buffer = base64.b64decode(arguments)
    clouds = np.frombuffer(buffer, dtype=np.float32).reshape(-1, 4)[:, :3]
    mlab = visualize_point_cloud(clouds, point_size=0.02)
    return mlab


if __name__ == '__main__':
    # 多进程队列
    queue = Manager().Queue(100)
    p = Process(target=start_signalr, args=(queue,))
    p.start()

    while True:
        mlab = render_vision()
        flag = input('>> 渲染完成,输入回车绘制下一个请求,输入e结束程序\n')
        print(flag)

        try:
            mlab.close()
        except:
            pass
        if flag == 'e':
            break
    p.terminate()
