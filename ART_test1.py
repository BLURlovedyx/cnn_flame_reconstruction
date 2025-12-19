import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# 1. 设置字体族
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 ['SimHei']
# 2. 解决负号‘-’显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

from dataset.gaijin_double_g import *

import numpy as np
from scipy.ndimage import rotate
import torch

import time

def project_3d_volume_with_weights(volume, angle_deg):
    """
    对3D体数据进行投影，并计算投影系数（权重矩阵）
    
    参数：
    volume: 3D体数据（D, H, W）
    angle_deg: 旋转角度
    detector_indices: 可选，指定哪些探测器像素需要计算权重
    
    返回：
    projection: 2D投影图像
    weight_matrices: 投影系数列表，每个元素是(体素索引, 权重)的元组列表
    """
    # 转换为NumPy数组
    volume_np = volume.cpu().numpy() if torch.is_tensor(volume) else volume
    
    # 对体数据进行旋转（绕Z轴）
    rotated_volume = rotate(volume_np, angle_deg, axes=(0, 1), reshape=False, order=1)
    
    # 获取尺寸
    D, H, W = rotated_volume.shape
    
    # 沿X轴投影（求和）
    projection = np.sum(rotated_volume, axis=0)
    
    # 归一化
    if projection.max() > 0:
        projection = projection / projection.max()
    

    return torch.tensor(projection, dtype=torch.float32)

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.transform import Rotation as R

def get_weight_matrix_parallel(volume, angle, detector_shape=None):
    """
    获取平行束几何下的权重矩阵
    
    参数：
    volume_shape: 体积形状 (D, H, W)
    angles: 投影角度列表（度）
    detector_shape: 探测器形状 (detector_h, detector_w)，默认与H,W相同
    
    返回：
    A: 稀疏权重矩阵，形状为 (num_rays, num_voxels)
    """
    D, H, W = volume.shape
    num_voxels = D * H * W
    
    if detector_shape is None:
        detector_h, detector_w = H, W
    else:
        detector_h, detector_w = detector_shape
    
    num_rays =  detector_h * detector_w
    
    # 使用LIL格式构建稀疏矩阵
    A = lil_matrix((num_rays, num_voxels))
    
    # 射线方向（平行束，沿X轴方向）
    ray_dir = np.array([1.0, 0.0, 0.0])
    
    ray_idx = 0
    
    angle_rad = np.radians(angle)
    
    # 绕Z轴的旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    
    # 逆旋转矩阵（从旋转后坐标转回原始坐标）
    inv_rotation = rotation_matrix.T
    
    for det_y in range(detector_h):
        for det_z in range(detector_w):
            # 探测器像素在旋转后坐标系中的位置
            # 在旋转后坐标系中，探测器像素的Y,Z坐标固定
            det_y_pos = (det_y - detector_h/2 + 0.5)
            det_z_pos = (det_z - detector_w/2 + 0.5)
            
            # 射线穿过旋转后体积的路径
            for x in range(D):
                # 在旋转后坐标系中的坐标
                rotated_pos = np.array([x - D/2 + 0.5, det_y_pos, det_z_pos])
                
                # 转换回原始坐标系
                original_pos = inv_rotation @ rotated_pos
                
                # 计算原始坐标系中的体素索引
                voxel_x = int(np.round(original_pos[0] + D/2 - 0.5))
                voxel_y = int(np.round(original_pos[1] + H/2 - 0.5))
                voxel_z = int(np.round(original_pos[2] + W/2 - 0.5))
                
                # 检查是否在体积范围内
                if 0 <= voxel_x < D and 0 <= voxel_y < H and 0 <= voxel_z < W:
                    voxel_idx = voxel_x * (H * W) + voxel_y * W + voxel_z
                    A[ray_idx, voxel_idx] = 1.0
            
            ray_idx += 1
    
    return A.tocsr()

def ART_reconstruction_3d(projections, angles,grid_size, num_angles, 
                          lambda_art=0.1, num_iterations=50, 
                          use_exact_weights=True, verbose=True):
    """
    ART实现,使用正确的权重计算
    """
    if isinstance(projections, torch.Tensor):
        projections = projections.numpy()
    
    if len(projections.shape) == 3:
        projections = projections[np.newaxis, ...]
        single_sample = True
    else:
        single_sample = False
    
    batch_size = projections.shape[0]
    
    reconstruction = np.zeros((batch_size, grid_size, grid_size, grid_size))
    
    for batch_idx in range(batch_size):
        if verbose:
            print(f"重建样本 {batch_idx+1}/{batch_size}")
        
        x = np.zeros((grid_size, grid_size, grid_size))
        proj_data = projections[batch_idx]
        
        # 预处理：归一化投影数据
        proj_data = proj_data / np.max(proj_data) if np.max(proj_data) > 0 else proj_data
        
        for iteration in range(num_iterations):
            if verbose and iteration % 10 == 0:
                print(f"  迭代 {iteration+1}/{num_iterations}")
            
            for angle_idx in range(num_angles):
                angle = angles[angle_idx]
                p_meas = proj_data[angle_idx]
                
                # 1. 前向投影：计算模拟投影
                proj= project_3d_volume_with_weights(x, angle)  
                weight=get_weight_matrix_parallel(x,angle,detector_shape=None)
                # 2. 计算误差
                print(proj.shape,weight.shape)
                error = proj - p_meas

                # ART更新公式
                update = lambda_art *error / (weight**2)*weight
                
                x -= update
                
                # 应用非负约束
                x = np.maximum(x, 0)
                
                # 可选：应用总变分平滑
                if iteration > 5 and iteration % 5 == 0:
                    # 简单的3x3x3均值滤波
                    from scipy.ndimage import uniform_filter
                    x = uniform_filter(x, size=3, mode='constant')
        
        reconstruction[batch_idx] = x
    
    if single_sample:
        return reconstruction[0]
    else:
        return reconstruction


def visualize_reconstruction_comparison(ground_truth, reconstruction, threshold=0.1):
    """
    可视化对比真实模型和重建结果
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 真实模型
    ax1 = fig.add_subplot(131, projection='3d')
    x, y, z = np.where(ground_truth > threshold)
    values = ground_truth[x, y, z]
    scatter1 = ax1.scatter(x, y, z, c=values, cmap='viridis', s=10, alpha=0.6)
    ax1.set_title('真实模型 (Ground Truth)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # 重建结果
    ax2 = fig.add_subplot(132, projection='3d')
    x, y, z = np.where(reconstruction > threshold)
    values = reconstruction[x, y, z]
    scatter2 = ax2.scatter(x, y, z, c=values, cmap='viridis', s=10, alpha=0.6)
    ax2.set_title('ART重建结果')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # 切片对比
    slice_idx = ground_truth.shape[0] // 2
    ax3 = fig.add_subplot(133)
    
    # 创建一个包含两个子图的复合图
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    im1 = ax3.imshow(ground_truth[slice_idx], cmap='hot', alpha=0.7, 
                    extent=[0, 1, 0, 1], origin='lower')
    im2 = ax3.imshow(reconstruction[slice_idx], cmap='cool', alpha=0.7, 
                    extent=[0, 1, 0, 1], origin='lower')
    ax3.set_title(f'中间切片对比 (Z={slice_idx})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='真实模型'),
                      Patch(facecolor='blue', alpha=0.7, label='重建结果')]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def calculate_reconstruction_metrics(ground_truth, reconstruction):
    """
    计算重建质量指标
    """
    # 确保输入是numpy数组
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.numpy()
    if isinstance(reconstruction, torch.Tensor):
        reconstruction = reconstruction.numpy()
    
    # 归一化
    gt_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
    recon_norm = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
    
    # 计算均方误差
    mse = np.mean((gt_norm - recon_norm) ** 2)
    
    # 计算结构相似性指数
    from skimage.metrics import structural_similarity as ssim
    ssim_value = 0
    for z in range(ground_truth.shape[0]):
        ssim_slice = ssim(gt_norm[z], recon_norm[z], 
                         data_range=1.0, 
                         gaussian_weights=True,
                         sigma=1.5,
                         use_sample_covariance=False)
        ssim_value += ssim_slice
    ssim_value /= ground_truth.shape[0]
    
    return {
        'MSE': mse,
        'SSIM': ssim_value,
        'PSNR': 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    }




# 主程序：使用ART重建您的双高斯分布
if __name__ == "__main__":
    # 设置参数
    GRID_SIZE = 32
    NUM_PROJECTIONS = 3
    NUM_SAMPLES = 10
    
    # 生成数据
    print("生成数据集...")
    X_train, Y_train,angles= create_dataset(NUM_SAMPLES, GRID_SIZE, NUM_PROJECTIONS, use_random_angles=True)
    print(f"输入形状 (2D 投影): {X_train.shape}")
    print(f"输出形状 (3D 模型): {Y_train.shape}")
    
    # 选择一个样本进行重建
    sample_idx = 5
    projections_sample = X_train[sample_idx]  # 形状: (3, 32, 32)
    ground_truth_sample = Y_train[sample_idx]  # 形状: (32, 32, 32)
    
    print(f"\n对样本 {sample_idx} 进行ART重建...")
    print(f"投影数据形状: {projections_sample.shape}")
    print(f"真实模型形状: {ground_truth_sample.shape}")
    
    # 进行ART重建
    reconstruction = ART_reconstruction_3d(
        projections=projections_sample,
        angles=angles,
        num_angles=NUM_PROJECTIONS,
        grid_size=GRID_SIZE,
        lambda_art=0.1,  # 较小的松弛因子更稳定
        num_iterations=50,  # 增加迭代次数
        verbose=True
    )
    
    print(f"\n重建完成！重建形状: {reconstruction.shape}")
    
    # 计算重建质量指标
    metrics = calculate_reconstruction_metrics(ground_truth_sample, reconstruction)
    print("\n重建质量评估:")
    print(f"均方误差 (MSE): {metrics['MSE']:.6f}")
    print(f"结构相似性 (SSIM): {metrics['SSIM']:.4f}")
    print(f"峰值信噪比 (PSNR): {metrics['PSNR']:.2f} dB")
    
    # 可视化对比
    print("\n生成可视化对比图...")
    fig = visualize_reconstruction_comparison(
        ground_truth=ground_truth_sample,
        reconstruction=reconstruction,
        threshold=0.1
    )
    plt.show()
    
