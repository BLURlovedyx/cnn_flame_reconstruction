import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from dataset.gaijin_double_g import *
from model.c_net_1 import *
from dataset.dataload  import FlameDatasetManager

# --- 运行数据生成 ---
GRID_SIZE = 32
NUM_PROJECTIONS = 3
NUM_SAMPLES = 1000
TRAIN_RATIO = 0.8
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


manager = FlameDatasetManager("data/flame_dataset")
data = manager.load_dataset("flame_small")
X_all = torch.cat([data['X_train'], data['X_val']], dim=0)
Y_all = torch.cat([data['Y_train'], data['Y_val']], dim=0)


# --- 划分训练集和验证集 ---
X_train, X_val, Y_train, Y_val = train_test_split(
    X_all, Y_all, 
    train_size=TRAIN_RATIO, 
    random_state=42,
    shuffle=True
)

print(f"\n数据集划分:")
print(f"训练集: {len(X_train)} 个样本")
print(f"验证集: {len(X_val)} 个样本")

# --- 创建 DataLoader ---
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 实例化模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = Flame3DReconstructionNet(input_channels=NUM_PROJECTIONS, output_size=GRID_SIZE).to(device)

# --- 损失函数和优化器 ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# --- 训练和验证循环 ---
print("\n开始训练模型...")

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 10

for epoch in range(NUM_EPOCHS):
    # --- 训练阶段 ---
    model.train()
    running_train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * inputs.size(0)
    
    train_loss = running_train_loss / len(train_dataset)
    train_losses.append(train_loss)
    
    # --- 验证阶段 ---
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * inputs.size(0)
    
    val_loss = running_val_loss / len(val_dataset)
    val_losses.append(val_loss)
    
    # 学习率调度
    scheduler.step(val_loss)
    
    # --- 早停机制 ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'best_model.pth')
        print(f"✓ 保存最佳模型 (epoch {epoch+1})")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"\n早停触发! 在 {epoch+1} 轮停止训练")
            break
    
    # 打印进度
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | '
          f'训练损失: {train_loss:.6f} | '
          f'验证损失: {val_loss:.6f} | '
          f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')

print("\n训练完成！")

# --- 可视化训练过程 ---
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='训练损失', marker='o', markersize=3)
plt.plot(val_losses, label='验证损失', marker='s', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('损失 (MSE)')
plt.title('训练和验证损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 在验证集上评估模型 ---
def evaluate_model(model, val_loader, device):
    model.eval()
    all_outputs = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # 保存结果用于后续分析
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader.dataset)
    
    # 合并所有批次
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return avg_loss, all_outputs, all_targets

print("\n在验证集上评估最佳模型...")
# 加载最佳模型
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

val_loss, val_outputs, val_targets = evaluate_model(model, val_loader, device)
print(f"验证集最终损失: {val_loss:.6f}")

# --- 可视化一些样本的重建结果 ---
def visualize_samples(inputs, outputs, targets, num_samples=3):
    """可视化输入投影、预测结果和真实值"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        if i >= len(inputs):
            break
            
        # 显示输入投影 (平均所有投影通道)
        proj = inputs[i].mean(dim=0)  # 平均所有投影角度
        axes[i, 0].imshow(proj, cmap='hot')
        axes[i, 0].set_title(f'样本 {i+1}: 输入投影')
        axes[i, 0].axis('off')
        
        # 显示预测的3D模型的中心切片
        pred_center = outputs[i, GRID_SIZE//2, :, :]
        axes[i, 1].imshow(pred_center, cmap='hot')
        axes[i, 1].set_title('预测3D (中心切片)')
        axes[i, 1].axis('off')
        
        # 显示真实的3D模型的中心切片
        true_center = targets[i, GRID_SIZE//2, :, :]
        axes[i, 2].imshow(true_center, cmap='hot')
        axes[i, 2].set_title('真实3D (中心切片)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

# 从验证集中取几个样本
num_viz_samples = 3
sample_indices = np.random.choice(len(X_val), num_viz_samples, replace=False)

sample_inputs = X_val[sample_indices]
sample_targets = Y_val[sample_indices]

with torch.no_grad():
    sample_inputs_tensor = sample_inputs.to(device)
    sample_outputs = model(sample_inputs_tensor).cpu()

visualize_samples(sample_inputs, sample_outputs, sample_targets, num_viz_samples)

# # --- 保存训练结果 ---
# results = {
#     'train_losses': train_losses,
#     'val_losses': val_losses,
#     'best_val_loss': best_val_loss,
#     'final_epoch': epoch + 1,
#     'num_train_samples': len(X_train),
#     'num_val_samples': len(X_val)
# }

# torch.save(results, 'training_results.pth')
# print("\n训练结果已保存到 training_results.pth")