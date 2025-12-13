import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.ndimage import rotate

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_2d_gaussian(size, params, add_background=True):
    """
    生成一个2D高斯分布
    size: 网格大小 (size x size)
    params: 高斯参数 (A, mu_x, mu_y, sigma_x, sigma_y, theta)
            其中theta是旋转角度（弧度）
    add_background: 是否添加背景基座
    """
    A, mu_x, mu_y, sigma_x, sigma_y, theta = params
    
    # 创建网格
    grid_range = torch.linspace(-1, 1, size)
    x, y = torch.meshgrid(grid_range, grid_range, indexing='ij')
    
    # 中心化坐标
    x_centered = x - mu_x
    y_centered = y - mu_y
    
    # 应用旋转
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    x_rot = x_centered * cos_theta - y_centered * sin_theta
    y_rot = x_centered * sin_theta + y_centered * cos_theta
    
    # 计算高斯分布
    exponent = -0.5 * ((x_rot / sigma_x)**2 + (y_rot / sigma_y)**2)
    gaussian = A * torch.exp(exponent)
    
    # 添加背景基座
    if add_background:
        # 创建一个更大的高斯作为背景
        background_sigma = 0.8
        background_strength = 0.2
        background = background_strength * torch.exp(
            -0.5 * ((x**2 + y**2) / background_sigma**2)
        )
        gaussian += background
    
    return gaussian

def create_double_gaussian_2d(size, params1, params2, add_background=True):
    """
    创建2D双高斯分布
    params1, params2: 两个高斯的参数
    """
    gaussian1 = generate_2d_gaussian(size, params1, add_background=False)
    gaussian2 = generate_2d_gaussian(size, params2, add_background=False)
    
    # 合并两个高斯
    double_gaussian = gaussian1 + gaussian2
    
    # 添加共同的背景
    if add_background:
        grid_range = torch.linspace(-1, 1, size)
        x, y = torch.meshgrid(grid_range, grid_range, indexing='ij')
        
        # 背景基座（更大、更平缓的高斯）
        background_sigma = 1.2
        background_strength = 0.3
        
        # 计算两个高斯中心的中间点
        center_x = (params1[1] + params2[1]) / 2
        center_y = (params1[2] + params2[2]) / 2
        
        background = background_strength * torch.exp(
            -0.5 * (
                ((x - center_x) / background_sigma)**2 + 
                ((y - center_y) / background_sigma)**2
            )
        )
        
        double_gaussian += background
    
    # 归一化
    if double_gaussian.max() > 0:
        double_gaussian = double_gaussian / double_gaussian.max()
    
    return double_gaussian

def generate_random_gaussian_params(seed=None):
    """生成随机高斯参数"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 随机参数
    A = np.random.uniform(0.7, 1.3)  # 振幅
    mu_x = np.random.uniform(-0.4, 0.4)  # x中心
    mu_y = np.random.uniform(-0.4, 0.4)  # y中心
    sigma_x = np.random.uniform(0.15, 0.35)  # x方向标准差
    sigma_y = np.random.uniform(0.15, 0.35)  # y方向标准差
    theta = np.random.uniform(0, np.pi)  # 旋转角度
    
    return (A, mu_x, mu_y, sigma_x, sigma_y, theta)

def create_3d_surface_from_2d(height_map, thickness=1.0):
    """
    将2D高度图转换为3D体数据
    height_map: 2D高度图 (H, W)
    thickness: 在z方向上的厚度
    """
    H, W = height_map.shape
    
    # 创建3D网格
    grid_range = torch.linspace(-1, 1, H)
    x, y, z = torch.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    
    # 将高度图扩展到3D
    # 对于每个(x,y)位置，z值小于高度的地方设为高度值
    height_map_3d = height_map.unsqueeze(2).repeat(1, 1, H)
    
    # 创建一个3D掩码，只保留高度以下的区域
    mask = z <= height_map_3d
    mask = mask & (z >= (height_map_3d - thickness))
    
    # 应用掩码
    volume = torch.zeros_like(mask, dtype=torch.float32)
    volume[mask] = height_map_3d[mask]
    
    return volume

def project_3d_volume(volume, angle_deg):
    """
    对3D体数据进行投影（沿Z轴旋转后沿X轴投影）
    volume: 3D体数据 (D, H, W)
    angle_deg: 旋转角度
    """
    # 将PyTorch张量转换为NumPy数组
    volume_np = volume.cpu().numpy() if torch.is_tensor(volume) else volume
    
    # 对体数据进行旋转（绕Z轴）
    rotated_volume = rotate(volume_np, angle_deg, axes=(1,2), reshape=False, order=1)
    
    # 沿X轴投影（求和）
    projection = np.sum(rotated_volume, axis=0)
    
    # 归一化
    if projection.max() > 0:
        projection = projection / projection.max()
    
    return torch.tensor(projection, dtype=torch.float32)

def create_dataset(num_samples, grid_size, num_projections, use_random_angles=True, thickness=0.2):
    """创建数据集"""
    inputs = []  # 2D 投影图像
    targets = [] # 3D 真实模型（高度图）
    
    for i in range(num_samples):
        # 设置随机种子
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 生成两个随机高斯参数
        params1 = generate_random_gaussian_params(seed=seed)
        params2 = generate_random_gaussian_params(seed=seed+100)
        
        # print(f"Sample {i+1}:")
        # print(f"  Gaussian 1: center=({params1[1]:.2f}, {params1[2]:.2f})")
        # print(f"  Gaussian 2: center=({params2[1]:.2f}, {params2[2]:.2f})")

        # 生成2D双高斯高度图
        height_map = create_double_gaussian_2d(
            grid_size, params1, params2, add_background=True
        )
        
        # 将2D高度图转换为3D体数据
        volume_3d = create_3d_surface_from_2d(height_map, thickness=thickness)
        
        # 存储3D图作为目标
        targets.append(volume_3d)

        # 生成多角度 2D 投影
        projections = []
        
        if use_random_angles:
            # 随机生成投影角度
            angles = np.random.uniform(0, 180, num_projections)
        else:
            # 固定角度（均匀分布）
            angles = np.linspace(0, 180, num_projections, endpoint=False)
            
        for angle in angles:
            proj = project_3d_volume(volume_3d, angle)
            projections.append(proj)
        
        # 将多个投影图像堆叠
        inputs.append(torch.stack(projections))

    return torch.stack(inputs), torch.stack(targets)



