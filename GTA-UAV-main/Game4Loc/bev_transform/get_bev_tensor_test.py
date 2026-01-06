import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
import random
import sys

sys.path.append('/data3/czh_code/dress')
from orthographic_corrector import OrthographicCorrector
from camera_config import get_camera_params

import time
import math


def get_BEV_tensor_new(img, Ho, Wo, Fov=170, dty=-20, dx=0, dy=0, dataset=False, camera_type="street_view",
                       device='cpu'):
    """
    新的BEV张量转换函数 - 使用正交校正器

    Args:
        img: 输入图像
        Ho: 输出高度
        Wo: 输出宽度
        Fov: 视野角度
        dty: 高度调整
        dx, dy: 平移调整
        dataset: 是否为数据集模式
        camera_type: 相机类型
        device: 计算设备

    Returns:
        BEV图像
    """
    device = device

    t0 = time.time()
    Hp, Wp = img.shape[0], img.shape[1]

    # 非标准全景图像补全
    if dty != 0 or Wp != 2 * Hp:
        ty = (Wp / 2 - Hp) / 2 + dty
        matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
        img = cv2.warpPerspective(img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))

    t1 = time.time()
    frame = torch.from_numpy(img.copy()).to(device)
    t2 = time.time()

    # 使用新的正交校正器计算变换
    camera_params = get_camera_params(camera_type)
    camera_params = update_camera_size(camera_params, Wp, Hp)

    corrector = OrthographicCorrector(camera_params)

    # 计算姿态角 - 根据Fov和调整参数计算
    roll_rad = np.radians(dy)  # dy对应滚转
    pitch_rad = -np.radians(Fov / 2)  # Fov对应俯仰
    yaw_rad = np.radians(dx)  # dx对应偏航

    # 计算单应性矩阵
    H_matrix = corrector.compute_homography_matrix(roll_rad, pitch_rad, yaw_rad)

    # 转换为UV坐标
    uv = homography_to_uv_torch(H_matrix, (Wo, Ho), device)

    t3 = time.time()

    # 应用网格采样
    BEV = F.grid_sample(frame.permute(2, 0, 1).unsqueeze(0).float(), uv.unsqueeze(0), align_corners=True)
    t4 = time.time()

    if dataset:
        return BEV.squeeze(0)
    else:
        if device == 'cpu':
            bev_img = BEV.permute(0, 2, 3, 1).squeeze(0).int()
            bev_img = np.array(bev_img).astype(np.uint8)
            return bev_img
        else:
            return BEV.permute(0, 2, 3, 1).squeeze(0).int()


def get_BEV_tensor_old(img, Ho, Wo, Fov=170, dty=-20, dx=0, dy=0, dataset=False, out=None, device='cpu'):
    """
    保留旧的BEV张量转换函数用于对比
    """
    device = device

    t0 = time.time()
    Hp, Wp = img.shape[0], img.shape[1]
    if dty != 0 or Wp != 2 * Hp:
        ty = (Wp / 2 - Hp) / 2 + dty
        matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
        img = cv2.warpPerspective(img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))

    t1 = time.time()
    frame = torch.from_numpy(img.copy()).to(device)
    t2 = time.time()

    if out is None:
        Fov = Fov * torch.pi / 180
        center = torch.tensor([Wp / 2 + dx, Hp + dy]).to(device)

        anglex = torch.tensor(dx).to(device) * 2 * torch.pi / Wp
        angley = -torch.tensor(dy).to(device) * torch.pi / Hp
        anglez = torch.tensor(0).to(device)

        euler_angles = (anglex, angley, anglez)
        euler_angles = torch.stack(euler_angles, -1)

        R02 = euler_angles_to_matrix(euler_angles, "XYZ")
        R20 = torch.inverse(R02)

        f = Wo / 2 / torch.tan(torch.tensor(Fov / 2))
        out = torch.zeros((Wo, Ho, 2)).to(device)
        f0 = torch.zeros((Wo, Ho, 3)).to(device)
        f0[:, :, 0] = Ho / 2 - (torch.ones((Ho, Wo)).to(device) * (torch.arange(Ho)).to(device)).T
        f0[:, :, 1] = Wo / 2 - torch.ones((Ho, Wo)).to(device) * torch.arange(Wo).to(device)
        f0[:, :, 2] = -torch.ones((Wo, Ho)).to(device) * f
        f1 = R20 @ f0.reshape((-1, 3)).T

        f1_0 = torch.sqrt(torch.sum(f1 ** 2, 0))
        f1_1 = torch.sqrt(torch.sum(f1[:2, :] ** 2, 0))
        theta = torch.arctan2(f1[2, :], f1_1) + torch.pi / 2
        phi = torch.arctan2(f1[1, :], f1[0, :])
        phi = phi + torch.pi

        i_p = 1 - theta / torch.pi
        j_p = 1 - phi / (2 * torch.pi)
        out[:, :, 0] = j_p.reshape((Ho, Wo))
        out[:, :, 1] = i_p.reshape((Ho, Wo))
        out[:, :, 0] = (out[:, :, 0] - 0.5) / 0.5
        out[:, :, 1] = (out[:, :, 1] - 0.5) / 0.5

    t3 = time.time()
    BEV = F.grid_sample(frame.permute(2, 0, 1).unsqueeze(0).float(), out.unsqueeze(0), align_corners=True)
    t4 = time.time()

    if dataset:
        return BEV.squeeze(0)
    else:
        if device == 'cpu':
            bev_img = BEV.permute(0, 2, 3, 1).squeeze(0).int()
            bev_img = np.array(bev_img).astype(np.uint8)
            return bev_img
        else:
            return BEV.permute(0, 2, 3, 1).squeeze(0).int()


def test_street_to_bev_new(input_image_path, output_image_path, dataset="CVACT", method="new"):
    """
    测试新的街景到BEV转换
    """
    if dataset == "CVACT":
        Ho = 384
        Wo = 384
        Fov = 135
        dty = 0
        camera_type = "cvact"
    elif dataset == "CVUSA":
        Ho = 616
        Wo = 1232
        Fov = 170
        dty = -20
        camera_type = "cvusa"
    else:
        raise ValueError("数据集必须是 'CVACT' 或 'CVUSA'")

    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {input_image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if method == "new":
        bev_image = get_BEV_tensor_new(
            img, Ho, Wo, Fov=Fov, dty=dty, dx=0, dy=0,
            camera_type=camera_type, device='cpu'
        )
    else:
        bev_image = get_BEV_tensor_old(
            img, Ho, Wo, Fov=Fov, dty=dty, dx=0, dy=0,
            device='cpu'
        )

    Image.fromarray(bev_image).save(output_image_path)
    print(f"鸟瞰图已保存至: {output_image_path}")


if __name__ == "__main__":
    input_image_path = "/data3/czh_code/gta/save/corrected_image.jpg"
    output_image_path = "/data3/czh_code/gta/save/get_bev_tensor_new.jpg"
    dataset = "CVACT"

    # 测试新方法
    test_street_to_bev_new(input_image_path, output_image_path, dataset, method="new")

    # 测试旧方法对比
    test_street_to_bev_new(input_image_path, output_image_path.replace(".jpg", "_old.jpg"), dataset, method="old")