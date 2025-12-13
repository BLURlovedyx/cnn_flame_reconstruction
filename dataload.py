import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dataset.gaijin_double_g import create_dataset
from c_net_1 import Flame3DReconstructionNet

class FlameDatasetManager:
    """数据集管理器，处理数据集的创建、保存和加载"""
    
    def __init__(self, data_dir="data/flame_dataset"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_and_save_dataset(self, dataset_name, num_samples=1000, grid_size=32, 
                                num_projections=3, train_ratio=0.8, force_create=False):
        """创建并保存数据集"""
        
        save_path = self.data_dir / f"{dataset_name}.pt"
        
        # 如果数据集已存在且不强制重新创建
        if save_path.exists() and not force_create:
            print(f"数据集 '{dataset_name}' 已存在，跳过创建...")
            return save_path
        
        print(f"正在创建数据集 '{dataset_name}'...")
        print(f"参数: 样本数={num_samples}, 网格大小={grid_size}, 投影数={num_projections}")
        
        # 生成完整数据集
        X_all, Y_all = create_dataset(
            num_samples, grid_size, num_projections, 
            use_random_angles=True
        )
        
        # 划分训练验证集
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_all, Y_all, 
            train_size=train_ratio, 
            random_state=42,
            shuffle=True
        )
        
        # 保存
        torch.save({
            'X_train': X_train,
            'Y_train': Y_train,
            'X_val': X_val,
            'Y_val': Y_val,
            'config': {
                'num_samples': num_samples,
                'grid_size': grid_size,
                'num_projections': num_projections,
                'train_ratio': train_ratio,
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            }
        }, save_path)
        
        print(f"数据集已保存到: {save_path}")
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        
        return save_path
    
    def load_dataset(self, dataset_name):
        """加载数据集"""
        
        filepath = self.data_dir / f"{dataset_name}.pt"
        
        if not filepath.exists():
            print(f"错误: 数据集 '{dataset_name}' 不存在!")
            return None
        
        data = torch.load(filepath)
        
        print(f"加载数据集: {dataset_name}")
        for key, value in data['config'].items():
            print(f"  {key}: {value}")
        
        return {
            'X_train': data['X_train'],
            'Y_train': data['Y_train'],
            'X_val': data['X_val'],
            'Y_val': data['Y_val'],
            'config': data['config']
        }
    
    def list_datasets(self):
        """列出所有可用的数据集"""
        
        datasets = list(self.data_dir.glob("*.pt"))
        
        if not datasets:
            print("没有找到任何数据集")
            return []
        
        print(f"在 {self.data_dir} 中找到的数据集:")
        for i, dataset in enumerate(datasets, 1):
            data = torch.load(dataset, map_location='cpu')
            config = data.get('config', {})
            print(f"{i}. {dataset.stem}:")
            print(f"   样本: {config.get('train_samples', '?')}+{config.get('val_samples', '?')}")
            print(f"   网格: {config.get('grid_size', '?')}")
            print(f"   投影: {config.get('num_projections', '?')}")
        
        return datasets