import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset.gaijin_double_g import*
from model.c_net_1 import*

# --- 运行数据生成 ---
GRID_SIZE = 32
NUM_PROJECTIONS = 3 # 假设我们使用3个投影角度
NUM_SAMPLES = 1000  # 训练样本数

print("正在生成数据集...")
# 
# 图示：三维体素网格中的火焰模型以及从不同角度观察到的二维投影图像。
X_train, Y_train = create_dataset(NUM_SAMPLES, GRID_SIZE, NUM_PROJECTIONS)
print(f"输入形状 (2D 投影): {X_train.shape}")
print(f"输出形状 (3D 模型): {Y_train.shape}")

# 创建 PyTorch DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# --- 实例化模型 ---
model = Flame3DReconstructionNet(input_channels=NUM_PROJECTIONS, output_size=GRID_SIZE).to(device)


# 损失函数: 均方误差 (MSE)
criterion = nn.MSELoss()
# 优化器: Adam
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# --- 训练参数 ---
num_epochs = 50

print("开始训练模型...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        # targets 形状: (B, D, H, W)
        # inputs 形状: (B, C, H, W)
        
        # 将数据移到 GPU/CPU
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

print("训练完成！")