import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import ImageOps
from orthographic_corrector import OrthographicCorrector, homography_to_uv_torch
from camera_config import get_camera_params


def grid_sample(image, optical, jac=None):
    """双线性插值采样函数 - 保持不变"""
    C, IH, IW = image.shape
    image = image.unsqueeze(0)

    _, H, W, _ = optical.shape

    ix = optical[..., 0].view(1, 1, H, W)
    iy = optical[..., 1].view(1, 1, H, W)

    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)
        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)
        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)
        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask = mask_x * mask_y

    assert torch.sum(mask) > 0

    nw = (ix_se - ix) * (iy_se - iy) * mask
    ne = (ix - ix_sw) * (iy_sw - iy) * mask
    sw = (ix_ne - ix) * (iy - iy_ne) * mask
    se = (ix - ix_nw) * (iy - iy_nw) * mask

    image = image.view(1, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(1, 1, H * W).repeat(1, C, 1)).view(1, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(1, 1, H * W).repeat(1, C, 1)).view(1, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(1, 1, H * W).repeat(1, C, 1)).view(1, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(1, 1, H * W).repeat(1, C, 1)).view(1, C, H, W)

    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:
        dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
                    sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
        dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
                    sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
        dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)

        jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
        jac_new1 = torch.sum(jac_new, dim=-1)

        return out_val, jac_new1
    else:
        return out_val, None


def BEV_transform_new(rot, S, H, W, meter_per_pixel, Camera_height, camera_type="cvact", device='cpu'):
    """
    新的BEV变换函数 - 使用正交校正器

    Args:
        rot: 旋转角度
        S: 输出尺寸
        H: 输入高度
        W: 输入宽度
        meter_per_pixel: 米每像素
        Camera_height: 相机高度
        camera_type: 相机类型
        device: 计算设备

    Returns:
        UV坐标
    """
    # 获取相机参数
    camera_params = get_camera_params(camera_type)
    camera_params = update_camera_size(camera_params, W, H)

    # 创建校正器
    corrector = OrthographicCorrector(camera_params)

    # 计算姿态角 - 模拟俯视效果
    roll = 0.0  # 无滚转
    pitch = -np.pi / 2  # -90度俯仰，完全俯视
    yaw = rot * np.pi / 180  # 将输入旋转转换为弧度

    # 计算单应性矩阵
    H_matrix = corrector.compute_homography_matrix(roll, pitch, yaw)

    # 转换为UV坐标
    uv = homography_to_uv_torch(H_matrix, (S, S), device)

    return uv


def BEV_transform_old(rot, S, H, W, meter_per_pixel, Camera_height):
    """
    保留旧的BEV变换函数用于对比
    """
    ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=rot.device),
                            torch.arange(0, S, dtype=torch.float32, device=rot.device), indexing='ij')
    ii = ii.unsqueeze(dim=0).repeat(1, 1, 1)
    jj = jj.unsqueeze(dim=0).repeat(1, 1, 1)

    max_dist = torch.sqrt(torch.tensor(2 * (S / 2) ** 2, dtype=torch.float32, device=rot.device))
    dist_to_center = torch.sqrt((ii - S / 2) ** 2 + (jj - S / 2) ** 2)
    normalized_dist = dist_to_center / max_dist
    h = 3 * (normalized_dist ** 4)

    radius = torch.sqrt((ii - (S / 2 - 0.5)) ** 2 + (jj - (S / 2 - 0.5)) ** 2)
    theta = torch.atan2(ii - (S / 2 - 0.5), jj - (S / 2 - 0.5))
    theta = (-np.pi / 2 + theta % (2 * np.pi)) % (2 * np.pi)

    theta = (theta + rot[:, None, None] * np.pi) % (2 * np.pi)
    theta = theta / (2 * np.pi) * W

    meter_per_pixel_tensor = torch.full((1, 1, 1), meter_per_pixel, device=radius.device)
    phimin = torch.atan2(radius * meter_per_pixel_tensor, torch.tensor(Camera_height, device=radius.device) + h)
    phimin = phimin / np.pi * H

    uv = torch.stack([theta, phimin], dim=-1)

    return uv


def resize_and_pad_image(image, target_height, target_width):
    """调整和填充图像 - 保持不变"""
    current_width, current_height = image.size

    if current_height == target_height:
        padded_image = image
    else:
        top_padding = (target_height - current_height) // 2
        bottom_padding = target_height - current_height - top_padding
        padded_image = ImageOps.expand(image, (0, top_padding, 0, bottom_padding), fill='black')

    resized_image = padded_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized_image


def Transformation_SVI_new(img_path, uv, H, W, resize_and_pad=False):
    """使用新方法的街景到鸟瞰图转换"""
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])

    image = Image.open(img_path)

    if resize_and_pad:
        image = resize_and_pad_image(image, H, W)

    image_tensor = transform(image)

    transformed_image, _ = grid_sample(image_tensor, uv)

    return np.transpose(transformed_image.detach().cpu().numpy()[0], (1, 2, 0))


def Transformation_SVI_img_new(image, uv, H, W, resize_and_pad=False):
    """图像版本的转换函数"""
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))

    if resize_and_pad:
        image = resize_and_pad_image(image, H, W)

    image_tensor = transform(image)

    transformed_image, _ = grid_sample(image_tensor, uv)

    return np.transpose(transformed_image.detach().cpu().numpy()[0], (1, 2, 0))


def CruvedBEV_new(input_path, dataset="CVACT", method="new"):
    """
    新的BEV转换主函数

    Args:
        input_path: 输入路径
        dataset: 数据集类型
        method: 方法选择，"new"或"old"
    """
    if dataset == "CVACT":
        S = 512
        H = 832
        W = 1664
        Camera_height = -1.5
        camera_type = "cvact"

    elif dataset == "CVUSA":
        S = 512
        H = 616
        W = 1232
        Camera_height = -1.5
        camera_type = "cvusa"

    else:
        raise ValueError("The dataset must be one of [CVACT, CVUSA, VIGOR, G2A-3]")

    rot = torch.tensor([90], dtype=torch.float32)
    meter_per_pixel = 0.06

    # 选择变换方法
    if method == "new":
        uv = BEV_transform_new(rot, S, H, W, meter_per_pixel, Camera_height, camera_type)
    else:
        uv = BEV_transform_old(rot, S, H, W, meter_per_pixel, Camera_height)

    Transformation_SVI_new(input_path, uv, H, W)


def CruvedBEV_img_new(image, method="new"):
    """图像版本的BEV转换"""
    S = 384
    H = 1024
    W = 2048
    Camera_height = -1.5

    rot = torch.tensor([90], dtype=torch.float32)
    meter_per_pixel = 0.06

    if method == "new":
        uv = BEV_transform_new(rot, S, H, W, meter_per_pixel, Camera_height, "street_view")
    else:
        uv = BEV_transform_old(rot, S, H, W, meter_per_pixel, Camera_height)

    bev_image = Transformation_SVI_img_new(image, uv, H, W)
    return bev_image


def test_CruvedBEV_new(input_image_path, output_image_path, dataset="CVACT", method="new"):
    """
    测试新的BEV转换函数
    """
    if dataset == "CVACT":
        S = 384
        H = 1024
        W = 2048
        Camera_height = -1.5
        camera_type = "cvact"
    elif dataset == "CVUSA":
        S = 512
        H = 616
        W = 1232
        Camera_height = -1.5
        camera_type = "cvusa"
    else:
        raise ValueError("数据集必须是 'CVACT' 或 'CVUSA'")

    rot = torch.tensor([90], dtype=torch.float32)
    meter_per_pixel = 0.06

    if method == "new":
        uv = BEV_transform_new(rot, S, H, W, meter_per_pixel, Camera_height, camera_type)
    else:
        uv = BEV_transform_old(rot, S, H, W, meter_per_pixel, Camera_height)

    bev_image = Transformation_SVI_new(
        input_image_path,
        uv,
        H,
        W,
        resize_and_pad=(dataset == "CVUSA")
    )

    bev_image = (bev_image * 255).astype(np.uint8)
    Image.fromarray(bev_image).save(output_image_path)
    print(f"鸟瞰图已保存至: {output_image_path}")


if __name__ == "__main__":
    input_image_path = "/data3/czh_code/dress/dataset/DReSS/Chicago/panorama/CHG,--pEtVASvNbkr8hW_WMXwQ,41.94138489,-87.83194741,192,2022.jpg"
    output_image_path = "/data3/jcx_code/dress/save/bev_output_new.jpg"
    dataset = "CVACT"

    # 测试新方法
    test_CruvedBEV_new(input_image_path, output_image_path, dataset, method="new")

    # 测试旧方法对比
    test_CruvedBEV_new(input_image_path, output_image_path.replace(".jpg", "_old.jpg"), dataset, method="old")