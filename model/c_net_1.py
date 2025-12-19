import torch.nn as nn
import torch.nn.functional as F

class Flame3DReconstructionNet(nn.Module):
    def __init__(self, input_channels, output_size):
        super(Flame3DReconstructionNet, self).__init__()
        
        # 1. 2D 特征提取 (处理多张 2D 投影)
        # 输入: (B, C_in, H, W) -> (B, num_projections, size, size)
        self.conv2d_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 图像尺寸减半
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 图像尺寸再次减半 (例如 32->8)
        )
        
        # 2. 特征提升到 3D
        # 输出: (B, 128, 8, 8) -> 展平后连接全连接层
        flat_size = 128 * (output_size // 4) * (output_size // 4)
        self.fc_to_3d = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 4 * 4 * 4) # 转换为 3D 特征张量
        )
        
        # 3. 3D 反卷积 (重建体素)
        self.conv3d_decoder = nn.Sequential(
            # 将 (B, 128, 4, 4, 4) 放大到 (B, 64, 8, 8, 8)
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            # 放大到 (B, 32, 16, 16, 16)
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            # 放大到 (B, 1, 32, 32, 32)
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # 输出归一化到 [0, 1] 之间
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (Batch, Num_Projections, Size, Size)
        
        # 1. 2D 编码
        x = self.conv2d_encoder(x) # (B, 128, 8, 8) (假设 size=32)
        
        # 2. 展平并连接全连接层
        x = x.view(x.size(0), -1)
        x = self.fc_to_3d(x)
        
        # 3. 形状转换: 1D 特征 -> 3D 初始特征图 (B, 128, 4, 4, 4)
        x = x.view(x.size(0), 128, 4, 4, 4)
        
        # 4. 3D 反卷积/解码
        reconstructed_3d = self.conv3d_decoder(x) # (B, 1, 32, 32, 32)
        
        # 移除通道维度 (B, 1, D, H, W) -> (B, D, H, W)
        return reconstructed_3d.squeeze(1)
