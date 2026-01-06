"""
新的测试主程序 - 用于测试所有新功能
"""

import cv2
import numpy as np
from orthographic_corrector import OrthographicCorrector
from camera_config import get_camera_params
from zhuanhua import correct_tilted_image_new
from cruvedbev import test_CruvedBEV_new
from get_bev_tensor import test_street_to_bev_new


def test_all_functions():
    """测试所有新功能"""

    # 测试图像路径
    test_image_path = "/data3/czh_code/gta/GTA-UAV-LR/drone/images/200_0001_0000006197.png"
    output_dir = "/data3/czh_code/gta/save/"

    print("开始测试所有新功能...")

    # 1. 测试倾斜校正
    print("\n1. 测试倾斜校正...")
    try:
        img = cv2.imread(test_image_path)
        if img is not None:
            corrected = correct_tilted_image_new(
                img,
                roll=0.1,  # 滚转角
                pitch=-0.3,  # 俯仰角
                yaw=0.05,  # 偏航角
                camera_type="uav"
            )
            cv2.imwrite(output_dir + "tilt_corrected_new.jpg", corrected)
            print("  倾斜校正测试完成")
        else:
            print("  无法读取测试图像")
    except Exception as e:
        print(f"  倾斜校正测试失败: {e}")

    # 2. 测试BEV转换
    print("\n2. 测试BEV转换...")
    try:
        test_CruvedBEV_new(
            test_image_path,
            output_dir + "bev_new.jpg",
            dataset="CVACT",
            method="new"
        )
        print("  BEV转换测试完成")
    except Exception as e:
        print(f"  BEV转换测试失败: {e}")

    # 3. 测试张量BEV转换
    print("\n3. 测试张量BEV转换...")
    try:
        test_street_to_bev_new(
            test_image_path,
            output_dir + "tensor_bev_new.jpg",
            dataset="CVACT",
            method="new"
        )
        print("  张量BEV转换测试完成")
    except Exception as e:
        print(f"  张量BEV转换测试失败: {e}")

    print("\n所有测试完成！")


def compare_methods():
    """对比新旧方法效果"""
    test_image_path = "/data3/czh_code/gta/GTA-UAV-LR/drone/images/200_0001_0000006197.png"
    output_dir = "/data3/czh_code/gta/save/"

    print("开始对比新旧方法...")

    # 对比BEV转换
    test_CruvedBEV_new(test_image_path, output_dir + "bev_old.jpg", "CVACT", "old")
    test_CruvedBEV_new(test_image_path, output_dir + "bev_new.jpg", "CVACT", "new")

    # 对比张量BEV转换
    test_street_to_bev_new(test_image_path, output_dir + "tensor_bev_old.jpg", "CVACT", "old")
    test_street_to_bev_new(test_image_path, output_dir + "tensor_bev_new.jpg", "CVACT", "new")

    print("对比完成！请查看输出图像对比效果")


if __name__ == "__main__":
    # 测试所有功能
    test_all_functions()

    # 对比新旧方法
    compare_methods()