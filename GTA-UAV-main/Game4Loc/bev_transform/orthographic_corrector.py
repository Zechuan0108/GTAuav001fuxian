import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import math


class OrthographicCorrector:
    """
    倾斜图像变换正射图像系统
    基于投影变换的几何校正方法
    """

    def __init__(self, camera_parameters: Dict):
        """
        初始化校正系统

        Args:
            camera_parameters: 相机参数字典
        """
        self.camera_params = camera_parameters

        # 计算相机内参矩阵
        self.K = self._compute_camera_matrix()
        self.K_inv = np.linalg.inv(self.K)

    def _compute_camera_matrix(self) -> np.ndarray:
        """计算相机内参矩阵"""
        width = self.camera_params["width"]
        height = self.camera_params["height"]
        focal_x = self.camera_params["focal_x"]
        focal_y = self.camera_params["focal_y"]
        c_x = self.camera_params["c_x"]
        c_y = self.camera_params["c_y"]

        return np.array([
            [focal_x * width, 0, (c_x + 0.5) * width],
            [0, focal_y * height, (c_y + 0.5) * height],
            [0, 0, 1]
        ])

    def euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        将滚转、俯仰、偏航角转换为旋转矩阵

        Args:
            roll: 滚转角 (绕X轴)
            pitch: 俯仰角 (绕Y轴)
            yaw: 偏航角 (绕Z轴)

        Returns:
            3x3旋转矩阵
        """
        # 滚转矩阵 (绕x轴)
        R_roll = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])

        # 俯仰矩阵 (绕y轴)
        R_pitch = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        # 航偏矩阵 (绕z轴)
        R_yaw = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵: R = R_yaw · R_pitch · R_roll
        R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        return R

    def compute_homography_matrix(self, roll: float, pitch: float, yaw: float,
                                  dx: float = 0, dy: float = 0, dz: float = 0) -> np.ndarray:
        """
        计算投影变换矩阵（单应性矩阵）
        基于投影变换的几何校正方法

        Args:
            roll: 滚转角
            pitch: 俯仰角
            yaw: 偏航角
            dx, dy, dz: 平移参数

        Returns:
            3x3单应性矩阵
        """
        w = self.camera_params["width"]
        h = self.camera_params["height"]

        c_x = (self.camera_params["c_x"] + 0.5) * w
        c_y = (self.camera_params["c_y"] + 0.5) * h
        f_x = self.camera_params["focal_x"] * w
        f_y = self.camera_params["focal_y"] * h

        # 使用平均焦距
        f = (f_x + f_y) / 2

        # 投影 2D -> 3D 矩阵
        A1 = np.array([
            [1, 0, -c_x],
            [0, 1, -c_y],
            [0, 0, 1],
            [0, 0, 1]
        ])

        # 绕X、Y、Z轴的旋转矩阵
        RX = np.array([
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll), np.cos(roll), 0],
            [0, 0, 0, 1]
        ])

        RY = np.array([
            [np.cos(pitch), 0, -np.sin(pitch), 0],
            [0, 1, 0, 0],
            [np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1]
        ])

        RZ = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 组合旋转矩阵
        R = np.dot(np.dot(RX, RY), RZ)

        # 平移矩阵
        T = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])

        # 投影 3D -> 2D 矩阵
        A2 = np.array([
            [f, 0, c_x, 0],
            [0, f, c_y, 0],
            [0, 0, 1, 0]
        ])

        # 最终变换矩阵: H = A2 · T · R · A1
        H = np.dot(A2, np.dot(T, np.dot(R, A1)))

        return H

    def correct_image(self, image: np.ndarray, roll: float, pitch: float, yaw: float,
                      dx: float = 0, dy: float = 0, dz: float = 0) -> np.ndarray:
        """
        校正图像的主函数

        Args:
            image: 输入图像
            roll: 滚转角
            pitch: 俯仰角
            yaw: 偏航角
            dx, dy, dz: 平移参数

        Returns:
            校正后的图像
        """
        height, width = image.shape[:2]

        # 获取投影变换矩阵
        H = self.compute_homography_matrix(roll, pitch, yaw, dx, dy, dz)

        # 应用透视变换
        corrected_image = cv2.warpPerspective(
            image, H, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return corrected_image

    def homography_to_uv(self, H: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """
        将单应性矩阵转换为UV坐标格式

        Args:
            H: 单应性矩阵
            output_size: 输出尺寸 (height, width)

        Returns:
            UV坐标网格
        """
        h, w = output_size

        # 创建网格坐标
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        xx, yy = np.meshgrid(x, y)

        # 齐次坐标
        ones = np.ones_like(xx)
        coords = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3).T

        # 应用变换
        transformed = H @ coords
        transformed = transformed / transformed[2]  # 齐次坐标归一化

        # 转换为UV格式 (归一化到[-1, 1])
        uv = transformed[:2].T.reshape(h, w, 2)
        uv_normalized = (uv / np.array([w - 1, h - 1]) * 2 - 1)

        return uv_normalized

    def visualize_correction(self, original_image: np.ndarray, corrected_image: np.ndarray,
                             roll: float, pitch: float, yaw: float):
        """可视化校正结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 显示原始图像
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始倾斜图像')
        axes[0].axis('off')

        # 显示校正图像
        axes[1].imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'校正后的正射图像\n(roll:{roll:.2f}, pitch:{pitch:.2f}, yaw:{yaw:.2f})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()


def homography_to_uv_torch(H: np.ndarray, output_size: Tuple[int, int], device='cpu'):
    """
    PyTorch版本的homography_to_uv
    """
    import torch
    h, w = output_size

    # 创建网格坐标
    x = torch.linspace(0, w - 1, w, device=device)
    y = torch.linspace(0, h - 1, h, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # 齐次坐标
    ones = torch.ones_like(xx)
    coords = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3).T

    # 转换为numpy计算单应性变换
    coords_np = coords.cpu().numpy()
    transformed_np = H @ coords_np
    transformed_np = transformed_np / transformed_np[2]

    # 转换回torch
    transformed = torch.from_numpy(transformed_np[:2]).to(device)
    uv = transformed.T.reshape(h, w, 2)

    # 归一化到[-1, 1]
    uv_normalized = (uv / torch.tensor([w - 1, h - 1], device=device)) * 2 - 1

    return uv_normalized