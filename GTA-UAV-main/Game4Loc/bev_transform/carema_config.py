"""
相机参数配置文件
"""

# 相机参数配置
DEFAULT_CAMERA_PARAMS = {
    # 街景相机参数
    "street_view": {
        "width": 2048,
        "height": 1024,
        "focal_x": 0.8,
        "focal_y": 0.8,
        "c_x": 0.0,
        "c_y": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0
    },
    # 卫星图像参数  
    "satellite": {
        "width": 512,
        "height": 512,
        "focal_x": 1.0,
        "focal_y": 1.0,
        "c_x": 0.0,
        "c_y": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0
    },
    # UAV无人机参数
    "uav": {
        "width": 640,
        "height": 512,
        "focal_x": 0.8645801943826472,
        "focal_y": 0.8645801943826472,
        "c_x": -0.005689562165298415,
        "c_y": -0.0037807145404085553,
        "k1": -0.2967016615133813,
        "k2": 0.08663139457756042,
        "p1": 0.0008407777901525121,
        "p2": 8.959107030007158e-05,
        "k3": 0.023342862058067047
    },
    # CVACT数据集参数
    "cvact": {
        "width": 1664,
        "height": 832,
        "focal_x": 0.8,
        "focal_y": 0.8,
        "c_x": 0.0,
        "c_y": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0
    },
    # CVUSA数据集参数
    "cvusa": {
        "width": 1232,
        "height": 616,
        "focal_x": 0.8,
        "focal_y": 0.8,
        "c_x": 0.0,
        "c_y": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0
    }
}


def get_camera_params(camera_type="street_view", custom_params=None):
    """
    获取指定类型的相机参数

    Args:
        camera_type: 相机类型
        custom_params: 自定义参数，会覆盖默认参数

    Returns:
        相机参数字典
    """
    params = DEFAULT_CAMERA_PARAMS.get(camera_type, DEFAULT_CAMERA_PARAMS["street_view"]).copy()

    if custom_params:
        params.update(custom_params)

    return params


def update_camera_size(params, width, height):
    """
    更新相机参数的尺寸

    Args:
        params: 原始相机参数
        width: 新宽度
        height: 新高度

    Returns:
        更新后的相机参数
    """
    new_params = params.copy()
    new_params["width"] = width
    new_params["height"] = height
    return new_params