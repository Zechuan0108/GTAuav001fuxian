import numpy as np
import cv2
import math
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class GeometricCorrectionSystem:
    """
    基于姿态参数的无人机视图几何校正系统
    输入：无人机图像 + 姿态角
    输出：鸟瞰视图
    """

    def __init__(self, img_width: int, img_height: int, focal_length: float = 800):
        """
        初始化几何校正系统

        Args:
            img_width: 图像宽度
            img_height: 图像高度
            focal_length: 焦距参数
        """
        self.img_width = img_width
        self.img_height = img_height
        self.focal_length = focal_length

        # 相机内参矩阵
        self.K = np.array([
            [focal_length, 0, img_width / 2],
            [0, focal_length, img_height / 2],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(self.K)

        # 缓存变换矩阵以提高性能
        self._H_cache = {}

    def euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        将欧拉角转换为旋转矩阵

        Args:
            roll: 滚转角 (弧度)
            pitch: 俯仰角 (弧度)
            yaw: 航偏角 (弧度)

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
                                  use_cache: bool = True) -> np.ndarray:
        """
        计算单应性矩阵

        Args:
            roll: 滚转角
            pitch: 俯仰角
            yaw: 航偏角
            use_cache: 是否使用缓存

        Returns:
            3x3单应性矩阵
        """
        # 检查缓存
        cache_key = (round(roll, 4), round(pitch, 4), round(yaw, 4))
        if use_cache and cache_key in self._H_cache:
            return self._H_cache[cache_key]

        # 获取旋转矩阵
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)

        # 计算单应性矩阵: H = K · R · K⁻¹
        H = np.dot(self.K, np.dot(R, self.K_inv))

        # 缓存结果
        if use_cache:
            self._H_cache[cache_key] = H

        return H

    def correct_to_bird_view(self, drone_image: np.ndarray,
                             roll: float, pitch: float, yaw: float,
                             output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        将无人机图像校正为鸟瞰视图

        Args:
            drone_image: 无人机图像
            roll: 滚转角
            pitch: 俯仰角
            yaw: 航偏角
            output_size: 输出图像尺寸 (宽, 高)，默认为输入尺寸

        Returns:
            校正后的鸟瞰图
        """
        if output_size is None:
            output_size = (self.img_width, self.img_height)

        # 计算单应性矩阵
        H = self.compute_homography_matrix(roll, pitch, yaw)

        # 应用透视变换
        bird_view = cv2.warpPerspective(
            drone_image,
            H,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # 黑色边框
        )

        return bird_view

    def create_grid_image(self, width: int, height: int,
                          grid_size: int = 50, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        创建网格图像用于可视化变换效果

        Args:
            width: 图像宽度
            height: 图像高度
            grid_size: 网格大小
            color: 网格线颜色 (B, G, R)

        Returns:
            网格图像
        """
        grid_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 绘制垂直线
        for x in range(0, width, grid_size):
            cv2.line(grid_img, (x, 0), (x, height), color, 1)

        # 绘制水平线
        for y in range(0, height, grid_size):
            cv2.line(grid_img, (0, y), (width, y), color, 1)

        # 绘制中心十字
        center_x, center_y = width // 2, height // 2
        cv2.line(grid_img, (center_x, 0), (center_x, height), (255, 0, 0), 2)
        cv2.line(grid_img, (0, center_y), (width, center_y), (255, 0, 0), 2)

        return grid_img

    def visualize_correction(self, drone_image: np.ndarray,
                             roll: float, pitch: float, yaw: float,
                             show_grid: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        可视化校正过程

        Args:
            drone_image: 无人机图像
            roll: 滚转角
            pitch: 俯仰角
            yaw: 航偏角
            show_grid: 是否显示网格

        Returns:
            (鸟瞰图, 变换后的网格图像)
        """
        # 生成鸟瞰图
        bird_view = self.correct_to_bird_view(drone_image, roll, pitch, yaw)

        # 生成网格可视化（可选）
        grid_transformed = None
        if show_grid:
            grid_img = self.create_grid_image(self.img_width, self.img_height)
            H = self.compute_homography_matrix(roll, pitch, yaw)
            grid_transformed = cv2.warpPerspective(
                grid_img, H, (self.img_width, self.img_height)
            )

        return bird_view, grid_transformed

    def batch_correct(self, drone_images: list, rolls: list, pitches: list, yaws: list) -> list:
        """
        批量校正多张图像

        Args:
            drone_images: 无人机图像列表
            rolls: 滚转角列表
            pitches: 俯仰角列表
            yaws: 航偏角列表

        Returns:
            校正后的鸟瞰图列表
        """
        assert len(drone_images) == len(rolls) == len(pitches) == len(yaws), "输入列表长度必须一致"

        bird_views = []
        for img, roll, pitch, yaw in zip(drone_images, rolls, pitches, yaws):
            bird_view = self.correct_to_bird_view(img, roll, pitch, yaw)
            bird_views.append(bird_view)

        return bird_views

    def get_correction_parameters(self, roll: float, pitch: float, yaw: float) -> dict:
        """
        获取校正参数信息

        Args:
            roll: 滚转角
            pitch: 俯仰角
            yaw: 航偏角

        Returns:
            包含校正参数的字典
        """
        H = self.compute_homography_matrix(roll, pitch, yaw)
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)

        return {
            'homography_matrix': H,
            'rotation_matrix': R,
            'camera_matrix': self.K,
            'roll_deg': math.degrees(roll),
            'pitch_deg': math.degrees(pitch),
            'yaw_deg': math.degrees(yaw)
        }


class BirdViewAnalyzer:
    """
    鸟瞰图分析器 - 用于评估校正效果
    """

    @staticmethod
    def calculate_image_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算两幅图像的相似度

        Args:
            img1: 图像1
            img2: 图像2

        Returns:
            相似度分数 (0-1)
        """
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算结构相似性
        try:
            from skimage.metrics import structural_similarity as ssim
            score, _ = ssim(gray1, gray2, full=True)
            return max(0, min(1, score))
        except ImportError:
            # 如果skimage不可用，使用简单的MSE方法
            mse = np.mean((gray1 - gray2) ** 2)
            return max(0, 1 - mse / 255.0)

    @staticmethod
    def analyze_geometric_properties(bird_view: np.ndarray) -> dict:
        """
        分析鸟瞰图的几何属性

        Args:
            bird_view: 鸟瞰图

        Returns:
            几何属性字典
        """
        # 转换为灰度图
        gray = cv2.cvtColor(bird_view, cv2.COLOR_BGR2GRAY)

        # 计算边缘
        edges = cv2.Canny(gray, 50, 150)

        # 计算直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=10)

        # 分析直线角度分布
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1)
                angles.append(angle)

        return {
            'edge_density': np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]),
            'line_count': len(angles) if angles else 0,
            'angle_variance': np.var(angles) if angles else 0
        }


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    加载并预处理图像

    Args:
        image_path: 图像路径
        target_size: 目标尺寸 (宽, 高)

    Returns:
        预处理后的图像
    """
    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 调整尺寸
    img_resized = cv2.resize(img, target_size)

    return img_resized


def demo_geometric_correction_with_real_image(image_path: str):
    """
    使用真实图像演示几何校正系统

    Args:
        image_path: 图像文件路径
    """
    print("=" * 60)
    print("无人机视图几何校正系统演示 (真实图像)")
    print("=" * 60)

    try:
        # 加载图像
        drone_image = load_and_preprocess_image(image_path)
        img_height, img_width = drone_image.shape[:2]

        # 创建校正系统
        corrector = GeometricCorrectionSystem(img_width=img_width, img_height=img_height)
        analyzer = BirdViewAnalyzer()

        # 测试不同的姿态角
        test_cases = [
            # (roll, pitch, yaw, 描述)
            (0.0, 0.0, 0.0, "水平飞行"),
            (0.1, 0.0, 0.0, "向右滚转"),
            (-0.1, 0.0, 0.0, "向左滚转"),
            (0.0, -0.2, 0.0, "俯仰向下"),
            (0.0, 0.2, 0.0, "俯仰向上"),
            (0.0, 0.0, 0.3, "向右航偏"),
            (0.05, -0.15, 0.1, "综合姿态")
        ]

        print(f"\n图像尺寸: {drone_image.shape}")
        print(f"相机内参矩阵:\n{corrector.K}")

        results = []

        for roll, pitch, yaw, description in test_cases:
            print(f"\n--- {description} ---")
            print(f"姿态角: roll={roll:.2f}rad, pitch={pitch:.2f}rad, yaw={yaw:.2f}rad")
            print(
                f"姿态角: roll={math.degrees(roll):.1f}°, pitch={math.degrees(pitch):.1f}°, yaw={math.degrees(yaw):.1f}°")

            # 执行几何校正
            bird_view, grid_viz = corrector.visualize_correction(drone_image, roll, pitch, yaw)

            # 分析校正结果
            properties = analyzer.analyze_geometric_properties(bird_view)

            print(f"鸟瞰图尺寸: {bird_view.shape}")
            print(f"几何属性: {properties}")

            results.append({
                'description': description,
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'bird_view': bird_view,
                'grid_viz': grid_viz,
                'properties': properties
            })

        # 可视化结果
        visualize_results(drone_image, results)

        return corrector, analyzer, results

    except Exception as e:
        print(f"错误: {e}")
        print("使用模拟图像进行演示...")
        return demo_geometric_correction_with_synthetic_image()


def demo_geometric_correction_with_synthetic_image():
    """
    使用合成图像演示几何校正系统
    """
    print("=" * 60)
    print("无人机视图几何校正系统演示 (合成图像)")
    print("=" * 60)

    # 创建校正系统
    corrector = GeometricCorrectionSystem(img_width=640, img_height=480)
    analyzer = BirdViewAnalyzer()

    # 创建合成测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 浅灰色背景

    # 添加一些特征用于可视化
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)  # 红色矩形
    cv2.circle(test_image, (400, 300), 50, (0, 255, 0), -1)  # 绿色圆形
    cv2.putText(test_image, "DRONE VIEW", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 测试不同的姿态角
    test_cases = [
        # (roll, pitch, yaw, 描述)
        (0.0, 0.0, 0.0, "水平飞行"),
        (0.1, 0.0, 0.0, "向右滚转"),
        (0.0, -0.2, 0.0, "俯仰向下"),
        (0.05, -0.15, 0.1, "综合姿态")
    ]

    print(f"\n图像尺寸: {test_image.shape}")
    print(f"相机内参矩阵:\n{corrector.K}")

    results = []

    for roll, pitch, yaw, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"姿态角: roll={roll:.2f}rad, pitch={pitch:.2f}rad, yaw={yaw:.2f}rad")

        # 执行几何校正
        bird_view, grid_viz = corrector.visualize_correction(test_image, roll, pitch, yaw)

        # 分析校正结果
        properties = analyzer.analyze_geometric_properties(bird_view)

        print(f"鸟瞰图尺寸: {bird_view.shape}")
        print(f"几何属性: {properties}")

        results.append({
            'description': description,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'bird_view': bird_view,
            'grid_viz': grid_viz,
            'properties': properties
        })

    # 可视化结果
    visualize_results(test_image, results)

    return corrector, analyzer, results


def visualize_results(original_img, results):
    """
    可视化校正结果
    """
    n = len(results)
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 8))

    # 显示原始图像
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("原始无人机视图")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')  # 第二行第一列留空

    # 显示校正结果
    for i, result in enumerate(results):
        # 显示鸟瞰图
        axes[0, i + 1].imshow(cv2.cvtColor(result['bird_view'], cv2.COLOR_BGR2RGB))
        axes[0, i + 1].set_title(f"{result['description']}\n鸟瞰图")
        axes[0, i + 1].axis('off')

        # 显示网格变换（如果有）
        if result['grid_viz'] is not None:
            axes[1, i + 1].imshow(cv2.cvtColor(result['grid_viz'], cv2.COLOR_BGR2RGB))
            axes[1, i + 1].set_title("网格变换")
            axes[1, i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def save_correction_results(results, output_dir="output"):
    """
    保存校正结果到文件

    Args:
        results: 校正结果列表
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        # 保存鸟瞰图
        bird_view_path = os.path.join(output_dir, f"bird_view_{i + 1}_{result['description']}.jpg")
        cv2.imwrite(bird_view_path, result['bird_view'])

        # 保存网格变换图（如果有）
        if result['grid_viz'] is not None:
            grid_path = os.path.join(output_dir, f"grid_{i + 1}_{result['description']}.jpg")
            cv2.imwrite(grid_path, result['grid_viz'])

    print(f"结果已保存到 {output_dir} 目录")


# 简单使用示例
def simple_usage_example():
    """
    简单使用示例
    """
    print("=" * 60)
    print("简单使用示例")
    print("=" * 60)

    # 1. 创建校正系统
    corrector = GeometricCorrectionSystem(img_width=800, img_height=600)

    # 2. 加载或创建无人机图像
    # 这里创建一个简单的测试图像
    drone_image = np.ones((600, 800, 3), dtype=np.uint8) * 150
    cv2.rectangle(drone_image, (200, 150), (400, 300), (0, 0, 255), -1)
    cv2.circle(drone_image, (600, 400), 80, (0, 255, 0), -1)

    # 3. 定义姿态参数
    roll = 0.1  # 弧度
    pitch = -0.2  # 弧度
    yaw = 0.05  # 弧度

    # 4. 执行几何校正
    bird_view = corrector.correct_to_bird_view(drone_image, roll, pitch, yaw)

    # 5. 显示结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(drone_image, cv2.COLOR_BGR2RGB))
    plt.title("原始无人机视图")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(bird_view, cv2.COLOR_BGR2RGB))
    plt.title("校正后的鸟瞰图")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("简单示例完成!")
    return bird_view


if __name__ == "__main__":
    # 选项1: 使用真实图像（替换为您的图像路径）
    # image_path = "path/to/your/drone_image.jpg"
    # corrector, analyzer, results = demo_geometric_correction_with_real_image(image_path)

    # 选项2: 使用合成图像
    corrector, analyzer, results = demo_geometric_correction_with_synthetic_image()

    # 选项3: 简单使用示例
    # bird_view = simple_usage_example()

    # 保存结果（可选）
    save_correction_results(results)

    print("\n" + "=" * 60)
    print("几何校正系统演示完成")
    print("=" * 60)