# flame_dataset_manager.py
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

class FlameDatasetManager:
    """数据集管理器，处理数据集的创建、保存和加载"""
    
    def __init__(self, data_dir="data/flame_dataset"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_manifest = self.data_dir / "datasets.json"
        self._load_manifest()
    
    def _load_manifest(self):
        """加载数据集清单"""
        if self.dataset_manifest.exists():
            with open(self.dataset_manifest, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
    
    def _save_manifest(self):
        """保存数据集清单"""
        with open(self.dataset_manifest, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def create_and_save_dataset(self, dataset_name, num_samples=1000, grid_size=32, 
                                num_projections=3, train_ratio=0.8, force_create=False):
        """创建并保存数据集"""
        
        save_path = self.data_dir / f"{dataset_name}.pt"
        
        # 检查数据集是否已存在
        if dataset_name in self.manifest and not force_create:
            print(f"数据集 '{dataset_name}' 已存在，跳过创建...")
            print(f"数据集信息: {self.manifest[dataset_name]}")
            return save_path
        
        print(f"正在创建数据集 '{dataset_name}'...")
        print(f"参数: 样本数={num_samples}, 网格大小={grid_size}, 投影数={num_projections}")
        
        # 这里需要导入您的create_dataset函数
        from dataset.gaijin_double_g import create_dataset
        
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
        
        # 保存到文件
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
                'val_samples': len(X_val),
                'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_shape': {
                    'X_train': X_train.shape,
                    'Y_train': Y_train.shape,
                    'X_val': X_val.shape,
                    'Y_val': Y_val.shape
                }
            }
        }, save_path)
        
        # 更新清单
        self.manifest[dataset_name] = {
            'path': str(save_path),
            'config': {
                'num_samples': num_samples,
                'grid_size': grid_size,
                'num_projections': num_projections,
                'train_ratio': train_ratio,
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            },
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_manifest()
        
        print(f"✓ 数据集已保存到: {save_path}")
        print(f"训练集: {len(X_train)} 样本, 形状: {X_train.shape} -> {Y_train.shape}")
        print(f"验证集: {len(X_val)} 样本, 形状: {X_val.shape} -> {Y_val.shape}")
        
        return save_path
    
    def load_dataset(self, dataset_name, return_config=True):
        """加载数据集 - 修复的关键函数！"""
        
        if dataset_name not in self.manifest:
            # 尝试直接查找文件
            filepath = self.data_dir / f"{dataset_name}.pt"
            if not filepath.exists():
                # 列出可用的数据集
                available = self.list_datasets(print_list=False)
                if available:
                    print(f"错误: 数据集 '{dataset_name}' 不存在!")
                    print(f"可用的数据集有: {', '.join(available)}")
                else:
                    print(f"错误: 数据集 '{dataset_name}' 不存在! 没有任何可用的数据集。")
                return None
        
        # 从清单获取路径
        dataset_info = self.manifest.get(dataset_name, {})
        filepath = Path(dataset_info.get('path', self.data_dir / f"{dataset_name}.pt"))
        
        if not filepath.exists():
            print(f"错误: 数据文件不存在: {filepath}")
            return None
        
        print(f"加载数据集: {dataset_name}")
        print(f"文件路径: {filepath}")
        
        try:
            data = torch.load(filepath)
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return None
        
        if 'config' in data:
            print("数据集配置:")
            for key, value in data['config'].items():
                if key not in ['data_shape', 'creation_date']:
                    print(f"  {key}: {value}")
        
        if return_config:
            return data
        else:
            return data['X_train'], data['Y_train'], data['X_val'], data['Y_val']
    
    def list_datasets(self, print_list=True):
        """列出所有可用的数据集"""
        
        datasets = {}
        
        # 从清单获取
        for name, info in self.manifest.items():
            datasets[name] = info
        
        # 也从文件系统查找
        for pt_file in self.data_dir.glob("*.pt"):
            if pt_file.stem not in datasets:
                datasets[pt_file.stem] = {'path': str(pt_file), 'in_manifest': False}
        
        if print_list:
            if not datasets:
                print("没有找到任何数据集")
            else:
                print(f"在 {self.data_dir} 中找到的数据集:")
                for i, (name, info) in enumerate(datasets.items(), 1):
                    print(f"{i}. {name}:")
                    if 'config' in info:
                        config = info['config']
                        print(f"   样本: {config.get('train_samples', '?')}+{config.get('val_samples', '?')}")
                        print(f"   网格: {config.get('grid_size', '?')}")
                        print(f"   投影: {config.get('num_projections', '?')}")
                    print(f"   路径: {info.get('path', '未知')}")
                    if not info.get('in_manifest', True):
                        print(f"   ⚠ 不在清单中")
        
        return list(datasets.keys())
    
    def delete_dataset(self, dataset_name):
        """删除数据集"""
        if dataset_name in self.manifest:
            filepath = Path(self.manifest[dataset_name]['path'])
            if filepath.exists():
                filepath.unlink()
                print(f"已删除文件: {filepath}")
            del self.manifest[dataset_name]
            self._save_manifest()
            print(f"已从清单中移除: {dataset_name}")
        else:
            print(f"数据集 '{dataset_name}' 不在清单中")
    
    def get_dataset_info(self, dataset_name):
        """获取数据集详细信息"""
        if dataset_name in self.manifest:
            return self.manifest[dataset_name]
        else:
            print(f"数据集 '{dataset_name}' 不存在")
            return None
        
        
# example_usage.py
def main():
    # 初始化数据集管理器
    manager = FlameDatasetManager("data/flame_dataset")
    
    # 1. 列出所有数据集
    print("当前所有数据集:")
    manager.list_datasets()
    
    print("\n" + "="*50 + "\n")
    
    # 2. 创建新数据集（如果不存在）
    manager.create_and_save_dataset(
        dataset_name="GAUSS_dataset_10000",  # 数据集名称
        num_samples=10000,             # 总样本数
        grid_size=32,                # 网格大小
        num_projections=3,           # 投影数量
        train_ratio=0.8              # 训练集比例
    )
    
    print("\n" + "="*50 + "\n")
    
    # 3. 加载数据集
    # data = manager.load_dataset("flame_small")
    
    # if data is not None:
    #     print(f"\n数据集形状:")
    #     print(f"X_train: {data['X_train'].shape}")
    #     print(f"Y_train: {data['Y_train'].shape}")
    #     print(f"X_val: {data['X_val'].shape}")
    #     print(f"Y_val: {data['Y_val'].shape}")
    
    # print("\n" + "="*50 + "\n")
    
    # # 4. 获取数据集信息
    # info = manager.get_dataset_info("flame_small")
    # if info:
    #     print("数据集详细信息:")
    #     import json
    #     print(json.dumps(info, indent=2, ensure_ascii=False))

# if __name__ == "__main__":
#     main()