import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main import*

# 假设模型已经训练完成，并且我们有了 X_train, Y_train, model, device 等变量

def visualize_reconstruction(model, X_data, Y_data, sample_index, slice_dim=0, slice_idx=None):
    """
    可视化真实模型和重建模型的一个切片。
    
    Args:
        model (nn.Module): 训练好的模型。
        X_data (Tensor): 输入的 2D 投影数据集。
        Y_data (Tensor): 真实的 3D 模型数据集。
        sample_index (int): 要可视化的样本索引。
        slice_dim (int): 切片维度 (0=X, 1=Y, 2=Z)。
        slice_idx (int): 切片索引 (例如，如果 size=32，取 16 为中心切片)。
    """
    
    model.eval() # 切换到评估模式
    
    # 提取输入和真实目标
    input_projections = X_data[sample_index:sample_index+1].to(device)
    true_3d = Y_data[sample_index].cpu().numpy()
    
    # 获取重建结果
    with torch.no_grad():
        reconstructed_3d = model(input_projections).squeeze(0).cpu().numpy()

    size = true_3d.shape[0]
    
    # 确定切片索引，默认取中心切片
    if slice_idx is None:
        slice_idx = size // 2
    
    # 提取切片
    if slice_dim == 0:
        true_slice = true_3d[slice_idx, :, :]
        reconstructed_slice = reconstructed_3d[slice_idx, :, :]
        dim_label = 'X'
    elif slice_dim == 1:
        true_slice = true_3d[:, slice_idx, :]
        reconstructed_slice = reconstructed_3d[:, slice_idx, :]
        dim_label = 'Y'
    else: # slice_dim == 2
        true_slice = true_3d[:, :, slice_idx]
        reconstructed_slice = reconstructed_3d[:, :, slice_idx]
        dim_label = 'Z'

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 真实模型切片
    im1 = axes[0].imshow(true_slice, cmap='hot', origin='lower')
    axes[0].set_title(f'Ground Truth (True Flame Model) - {dim_label}-Slice {slice_idx}')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. 重建模型切片
    im2 = axes[1].imshow(reconstructed_slice, cmap='hot', origin='lower')
    axes[1].set_title(f'Reconstruction (CNN Output) - {dim_label}-Slice {slice_idx}')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

# --- 运行可视化 ---

# 假设我们在训练集中选取第一个样本进行可视化
SAMPLE_TO_VISUALIZE = 0 

# 可视化 Z 轴的中心切片 (slice_dim=2)
print("\n--- 可视化 Z 轴中心切片 ---")
visualize_reconstruction(model, X_train, Y_train, 
                         sample_index=SAMPLE_TO_VISUALIZE, 
                         slice_dim=2, 
                         slice_idx=GRID_SIZE // 2)

# 可视化 X 轴的中心切片 (slice_dim=0)
print("\n--- 可视化 X 轴中心切片 ---")
visualize_reconstruction(model, X_train, Y_train, 
                         sample_index=SAMPLE_TO_VISUALIZE, 
                         slice_dim=0, 
                         slice_idx=GRID_SIZE // 2)