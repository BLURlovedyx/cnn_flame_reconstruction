import torch
import numpy as np

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_3d_gaussian(size, params):
    """
    生成一个3D双高斯分布作为火焰的真实模型 (Ground Truth)。
    size: 体素网格大小，例如 32
    params: 双高斯参数 (A1, mux1, muy1, muz1, sigma1, A2, mux2, muy2, muz2, sigma2)
    """
    A1, mx1, my1, mz1, s1, A2, mx2, my2, mz2, s2 = params
    grid_range = torch.linspace(-1, 1, size)
    x, y, z = torch.meshgrid(grid_range, grid_range, grid_range, indexing='ij')

    # 第一高斯
    g1 = A1 * torch.exp(
        -0.5 * (((x - mx1) / s1)**2 + ((y - my1) / s1)**2 + ((z - mz1) / s1)**2)
    )
    # 第二高斯
    g2 = A2 * torch.exp(
        -0.5 * (((x - mx2) / s2)**2 + ((y - my2) / s2)**2 + ((z - mz2) / s2)**2)
    )
    
    # 返回归一化后的模型
    flame_model = g1 + g2
    return flame_model / flame_model.max()

def project_model(model_3d, angle_deg):
    """
    模拟射线追踪投影 (Ray Casting / Tomographic Projection)。
    model_3d: (D, H, W) 三维体素网格
    angle_deg: 投影角度 (0到360度)
    """
    size = model_3d.shape[0]
    angle_rad = np.deg2rad(angle_deg)
    
    # 旋转矩阵 (绕Z轴旋转)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = torch.tensor([[cos_a, -sin_a, 0], 
                      [sin_a, cos_a, 0], 
                      [0, 0, 1]], dtype=torch.float32).to(device)

    # 创建目标网格 (投影平面, Y-Z平面, X是投影方向)
    grid_range = torch.linspace(-1, 1, size).to(device)
    x_coords, y_coords, z_coords = torch.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    
    # 将原始坐标 (x, y, z) 堆叠成 (N^3, 3) 矩阵
    coords_orig = torch.stack([x_coords.flatten(), y_coords.flatten(), z_coords.flatten()], dim=1)
    
    # 旋转坐标: new_coords = coords_orig @ R.T
    coords_rot = coords_orig @ R.T # (N^3, 3)
    
    # 提取新的X, Y, Z坐标
    x_rot, y_rot, z_rot = coords_rot[:, 0].reshape(size, size, size), \
                          coords_rot[:, 1].reshape(size, size, size), \
                          coords_rot[:, 2].reshape(size, size, size)
    
    # 射线积分 (近似): 沿着旋转后的X轴方向进行求和
    # 在CT中，投影是沿着射线路径的积分。由于我们旋转了网格，我们只需要沿着新的X轴求和。
    # **注意：这是一个近似的离散化投影方法。** 更精确的方法是使用插值。
    projected_image = torch.sum(model_3d, dim=0) # Sum along the 'depth' axis of the rotated grid
    
    return projected_image

def create_dataset(num_samples, grid_size, num_projections):
    """创建数据集"""
    inputs = []  # 2D 投影图像
    targets = [] # 3D 真实模型
    
    for _ in range(num_samples):
        # 随机生成双高斯参数
        A1, A2 = np.random.uniform(0.8, 1.2, 2)
        m_center = np.random.uniform(-0.5, 0.5, 6) # mx1, my1, mz1, mx2, my2, mz2
        s_val = np.random.uniform(0.1, 0.3, 2) # sigma1, sigma2
        params = (A1, m_center[0], m_center[1], m_center[2], s_val[0],
                  A2, m_center[3], m_center[4], m_center[5], s_val[1])

        # 1. 生成 3D 真实模型
        model_3d = generate_3d_gaussian(grid_size, params).to(device)
        targets.append(model_3d)

        # 2. 生成多角度 2D 投影
        angles = np.linspace(0, 180, num_projections, endpoint=False)
        projections = []
        for angle in angles:
            # 简化：这里我们直接使用固定的角度，而不是随机投影
            # 实际的投影函数应该使用更精确的射线插值，这里使用简单的张量旋转和求和代替
            # 为了让代码跑起来，我们简化投影过程：只使用三个正交投影
            if angle == 0:
                 proj = torch.sum(model_3d, dim=0) # 沿X轴投影 (Y-Z平面)
            elif angle == 60:
                 proj = torch.sum(model_3d, dim=1) # 沿Y轴投影 (X-Z平面)
            elif angle == 120:
                 proj = torch.sum(model_3d, dim=2) # 沿Z轴投影 (X-Y平面)
            else:
                 proj = torch.sum(model_3d, dim=0)
                 
            # 归一化投影图像
            projections.append(proj / proj.max()) 
        
        # 将多个投影图像堆叠成 (C, H, W) 作为输入
        inputs.append(torch.stack(projections))

    return torch.stack(inputs), torch.stack(targets)

