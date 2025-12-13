import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    double_gaussian = double_gaussian / double_gaussian.max()
    
    return double_gaussian, gaussian1, gaussian2

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

def visualize_2d_gaussian(gaussian, title="2D Gaussian Distribution"):
    """可视化2D高斯分布"""
    data = gaussian.numpy() if torch.is_tensor(gaussian) else gaussian
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2D热图
    im1 = axes[0].imshow(data, cmap='hot', origin='lower',
                        extent=[-1, 1, -1, 1], aspect='auto')
    axes[0].set_title(f'{title} - Heatmap')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # 等高线图
    contour = axes[1].contourf(data, levels=20, cmap='hot',
                              extent=[-1, 1, -1, 1])
    axes[1].set_title(f'{title} - Contour')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(contour, ax=axes[1])
    
    # 3D曲面图
    ax3d = axes[2] = fig.add_subplot(133, projection='3d')
    X, Y = np.meshgrid(np.linspace(-1, 1, data.shape[0]),
                      np.linspace(-1, 1, data.shape[1]))
    surf = ax3d.plot_surface(X, Y, data, cmap='hot', alpha=0.8,
                            linewidth=0, antialiased=True)
    ax3d.set_title(f'{title} - 3D Surface')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z (Value)')
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    return fig

def visualize_double_gaussian_3d(gaussian, title="Double Gaussian Distribution", 
                                 filename=None, show_individual=False):
    """可视化双高斯分布的三维曲面图"""
    data = gaussian.numpy() if torch.is_tensor(gaussian) else gaussian
    
    # 创建网格
    x = np.linspace(-1, 1, data.shape[0])
    y = np.linspace(-1, 1, data.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 创建图形
    fig = plt.figure(figsize=(16, 6))
    
    if show_individual:
        # 显示三个子图：组合、高斯1、高斯2
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        
        # 组合分布
        surf1 = ax1.plot_surface(X, Y, data, cmap='hot', alpha=0.8,
                                linewidth=0, antialiased=True)
        ax1.set_title(f'{title} - Combined')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z (Value)')
        ax1.set_zlim(0, 1)
        fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10)
        
        # 假设我们只有组合数据，没有单独的
        # 这里可以添加单独的高斯可视化
        ax2.set_title("Individual Gaussian 1")
        ax3.set_title("Individual Gaussian 2")
        
    else:
        # 只显示组合分布的大图
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制曲面
        surf = ax.plot_surface(X, Y, data, cmap='hot', alpha=0.8,
                             linewidth=0, antialiased=True, rstride=1, cstride=1)
        
        # 设置标签和标题
        ax.set_xlabel('X', fontsize=12, labelpad=10)
        ax.set_ylabel('Y', fontsize=12, labelpad=10)
        ax.set_zlabel('Value', fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=16, pad=20)
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        # 添加颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=20, pad=0.1)
        cbar.set_label('Value', fontsize=12)
        
        # 设置z轴范围
        ax.set_zlim(0, 1.1)
        
        # 设置坐标轴刻度
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 设置坐标轴标签字体大小
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='z', which='major', labelsize=10)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")
    
    return fig

# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("2D Gaussian Distribution Generator")
    print("=" * 60)
    
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. 生成并可视化单个随机高斯
    print("\n1. Generating a single random 2D Gaussian...")
    size = 200
    
    # 生成随机参数
    params_single = generate_random_gaussian_params(seed=seed)
    print(f"Single Gaussian parameters:")
    print(f"  Amplitude: {params_single[0]:.3f}")
    print(f"  Center (x, y): ({params_single[1]:.3f}, {params_single[2]:.3f})")
    print(f"  Sigma (x, y): ({params_single[3]:.3f}, {params_single[4]:.3f})")
    print(f"  Rotation: {params_single[5]:.3f} rad")
    
    # 2. 生成并可视化双高斯分布
    print("\n2. Generating a double 2D Gaussian distribution...")
    
    # 生成两个随机高斯参数
    params1 = generate_random_gaussian_params(seed=seed+1)
    params2 = generate_random_gaussian_params(seed=seed+2)
    
    print(f"First Gaussian parameters:")
    print(f"  Amplitude: {params1[0]:.3f}")
    print(f"  Center (x, y): ({params1[1]:.3f}, {params1[2]:.3f})")
    print(f"  Sigma (x, y): ({params1[3]:.3f}, {params1[4]:.3f})")
    print(f"  Rotation: {params1[5]:.3f} rad")
    
    print(f"\nSecond Gaussian parameters:")
    print(f"  Amplitude: {params2[0]:.3f}")
    print(f"  Center (x, y): ({params2[1]:.3f}, {params2[2]:.3f})")
    print(f"  Sigma (x, y): ({params2[3]:.3f}, {params2[4]:.3f})")
    print(f"  Rotation: {params2[5]:.3f} rad")
    
    # 生成双高斯
    double_gaussian, gauss1, gauss2 = create_double_gaussian_2d(
        size, params1, params2, add_background=True
    )
    
    # 可视化双高斯
    fig2 = visualize_double_gaussian_3d(
        double_gaussian, 
        "Double 2D Gaussian Distribution (3D Surface)",
        filename="double_gaussian_3d.png",
        show_individual=False
    )
    
    plt.suptitle('Multiple Random Double Gaussian Examples', fontsize=16)
    plt.tight_layout()
    plt.savefig("multiple_gaussian_examples.png", dpi=150, bbox_inches='tight')
    
    # 显示所有图形
    print("\nAll visualizations have been generated!")
    print("Figures saved:")
    print("  - single_2d_gaussian.png")
    print("  - double_gaussian_3d.png")
    print("  - complex_flame_gaussian.png")
    print("  - gaussian_comparison.png")
    print("  - multiple_gaussian_examples.png")
    
    plt.show()
    
    # 打印参数总结
    print("\n" + "=" * 60)
    print("Parameter Summary")
    print("=" * 60)
    print(f"Grid size: {size}x{size}")
    print(f"Single Gaussian created with random parameters")
    print(f"Double Gaussian created with 2 random Gaussians + background")
    print(f"Complex flame-like Gaussian created for realistic flame simulation")