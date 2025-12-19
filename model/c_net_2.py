import torch.nn as nn
import torch.nn.functional as F
import torch

class ImprovedFlame3DNet(nn.Module):
    def __init__(self, input_channels=3, output_size=32):
        super().__init__()
        
        # 1. 简化2D编码器
        self.encoder = nn.Sequential(
            # 第1层: 32×32 -> 16×16
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第2层: 16×16 -> 8×8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第3层: 8×8 -> 4×4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 2. 2D->3D转换 (不使用全连接瓶颈)
        # 直接将特征图重塑为3D
        self.to_3d = nn.Sequential(
            # 将2D特征 [B, 128, 4, 4] 转换为3D特征 [B, 32, 4, 4, 4]
            nn.Conv2d(128, 32 * 4, 1),  # 1x1卷积调整通道数
            nn.ReLU(),
        )
        
        # 3. 简化3D解码器
        self.decoder = nn.Sequential(
            # 4×4×4 -> 8×8×8
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
            
            # 8×8×8 -> 16×16×16
            nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
            
            # 16×16×16 -> 32×32×32
            nn.ConvTranspose3d(8, 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 4, 3, padding=1),
            nn.ReLU(),
            
            # 最终输出
            nn.Conv3d(4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 2D编码
        x = self.encoder(x)  # [B, 128, 4, 4]
        
        # 2D->3D转换
        x = self.to_3d(x)  # [B, 128, 4, 4]
        B, C, H, W = x.shape
        x = x.view(B, 32, 4, H, W)  # 重塑为 [B, 32, 4, 4, 4]
        
        # 3D解码
        x = self.decoder(x)  # [B, 1, 32, 32, 32]
        
        return x.squeeze(1)