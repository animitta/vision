import base64
import time
import numpy as np
from signalrclient import SignalRClient
from multiprocessing import Process

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mayavi import mlab
from utils import draw_bbox, draw_colored_pointcloud

render = False


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
        return fig


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


def read_point_cloud(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)  # (N, 3)


def handle_message(message):
    global render
    method = message[0]
    arguments = base64.b64decode(message[1])
    print(f'>> 接受渲染:{method}, 参数大小:{len(arguments)}')
    render = True
    # clouds = np.frombuffer(arguments, dtype=np.float32).reshape(-1, 4)
    # visualize_point_cloud(clouds[:, :3], point_size=0.01)


if __name__ == '__main__':
    print('准备启动连接')
    with SignalRClient(handle_message):
        print('准备渲染图像')
        clouds = read_point_cloud('data/pc.bin')[:, :3]
        fig = visualize_point_cloud(clouds, point_size=0.01)

        while True:
            time.sleep(1)
