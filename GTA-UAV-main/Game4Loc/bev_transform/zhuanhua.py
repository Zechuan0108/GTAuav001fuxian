import cv2
import numpy as np
from orthographic_corrector import OrthographicCorrector
from camera_config import get_camera_params


def correct_tilted_image_new(image, roll=0.0, pitch=0.0, yaw=0.0,
                             camera_type="uav", dx=0, dy=0, dz=0):
    """
    使用新的正交校正器进行图像校正

    Args:
        image: 输入图像 (numpy数组)
        roll: 滚转角 (弧度)
        pitch: 俯仰角 (弧度)
        yaw: 偏航角 (弧度)
        camera_type: 相机类型
        dx, dy, dz: 平移参数

    Returns:
        校正后的图像
    """
    # 获取相机参数
    camera_params = get_camera_params(camera_type)

    # 如果提供了图像，更新尺寸参数
    if image is not None:
        height, width = image.shape[:2]
        camera_params = update_camera_size(camera_params, width, height)

    # 创建校正器
    corrector = OrthographicCorrector(camera_params)

    # 应用校正
    corrected_image = corrector.correct_image(image, roll, pitch, yaw, dx, dy, dz)

    return corrected_image


def correct_tilted_image_batch(images, rolls, pitches, yaws, camera_type="uav"):
    """
    批量校正图像

    Args:
        images: 图像列表
        rolls: 滚转角列表
        pitches: 俯仰角列表
        yaws: 偏航角列表
        camera_type: 相机类型

    Returns:
        校正后的图像列表
    """
    corrected_images = []

    for i, image in enumerate(images):
        roll = rolls[i] if i < len(rolls) else 0.0
        pitch = pitches[i] if i < len(pitches) else 0.0
        yaw = yaws[i] if i < len(yaws) else 0.0

        corrected = correct_tilted_image_new(image, roll, pitch, yaw, camera_type)
        corrected_images.append(corrected)

    return corrected_images


# 兼容旧接口的函数
def correct_tilted_image(image, pitch_angle=5, roll_angle=0, focal_length=None, sensor_size=None):
    """
    兼容旧接口的校正函数

    Args:
        image: 输入图像
        pitch_angle: 俯仰角 (度)
        roll_angle: 滚转角 (度)
        focal_length: 保留参数，用于兼容
        sensor_size: 保留参数，用于兼容

    Returns:
        校正后的图像
    """
    # 角度转换：度->弧度
    roll_rad = np.radians(roll_angle)
    pitch_rad = np.radians(pitch_angle)
    yaw_rad = 0.0  # 默认无偏航

    return correct_tilted_image_new(image, roll_rad, pitch_rad, yaw_rad, "uav")


# 测试函数
if __name__ == "__main__":
    # 读取原始图像
    original_image = cv2.imread("/data3/czh_code/gta/GTA-UAV-LR/drone/images/200_0001_0000006197.png")

    if original_image is not None:
        # 假设俯仰角为30度（无人机向下倾斜30度）
        pitch_angle = 5
        roll_angle = 0

        # 使用新方法校正图像
        corrected_image = correct_tilted_image_new(original_image, pitch_angle, roll_angle)

        # 保存结果
        cv2.imwrite("/data3/czh_code/gta/save/corrected_image_new.jpg", corrected_image)
        print("图像校正完成，结果已保存")
    else:
        print("无法读取图像文件")