import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import copy

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModalEyeDataset(Dataset):
    """眼科多模态数据集，支持IVCM、OCT和前节照多模态融合，并处理缺失模态"""
    
    def __init__(self, data_dir, multimodal_index_path, mode='train', 
                 modalities=None, split_path=None, transform=None, 
                 max_images_per_modality=5, modality_completion=True):
        """
        初始化眼科多模态分类数据集
        
        Args:
            data_dir (str): 数据根目录
            multimodal_index_path (str): 多模态数据集索引文件路径
            mode (str): 'train', 'val', 或 'test'
            modalities (List[str], optional): 使用的模态列表，默认使用所有模态['IVCM', 'OCT', '前节照']
            split_path (str, optional): 数据集分割文件路径，如果提供则使用预定义的分割
            transform (Dict[str, callable], optional): 各模态图像变换字典
            max_images_per_modality (int): 每个模态最多使用的图像数量
            modality_completion (bool): 是否进行模态补全
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.max_images_per_modality = max_images_per_modality
        self.modality_completion = modality_completion
        
        # 设置默认使用的模态
        if modalities is None:
            self.modalities = ['IVCM', 'OCT', '前节照']
        else:
            self.modalities = modalities
        
        # 加载多模态数据集索引
        with open(multimodal_index_path, 'r', encoding='utf-8') as f:
            self.multimodal_dataset = json.load(f)
        
        # 使用预定义的数据分割
        if split_path:
            with open(split_path, 'r', encoding='utf-8') as f:
                self.samples = json.load(f)
            logger.info(f"从预定义分割文件加载 {mode} 集: {len(self.samples)} 个样本")
        else:
            # 使用所有可用的多模态样本
            self.samples = self.multimodal_dataset
            logger.info(f"使用所有样本: {len(self.samples)} 个样本")
        
        # 类别映射
        self.class_mapping = {
            'FECD': 0,
            'normal': 1,
            'other': 2
        }
        
        # 图像转换
        self.transform = {}
        if transform is None:
            # 为每个模态设置默认的图像变换
            for modality in self.modalities:
                if mode == 'train':
                    self.transform[modality] = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    self.transform[modality] = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        else:
            self.transform = transform
            
        # 生成模态替代样本池，用于缺失模态补全
        if self.modality_completion:
            self._create_modality_pools()
        
    def _create_modality_pools(self):
        """为每个模态创建样本池，用于缺失模态补全"""
        self.modality_pools = {modality: [] for modality in self.modalities}
        
        # 收集每个模态的所有样本
        for sample in self.samples:
            for modality in self.modalities:
                if modality in sample['modalities']:
                    # 存储样本ID和对应的模态数据
                    self.modality_pools[modality].append({
                        'id': sample['id'],
                        'modality_data': sample['modality_data'][modality]
                    })
        
        # 记录每个模态的样本数量
        for modality, pool in self.modality_pools.items():
            logger.info(f"模态 {modality} 的样本池大小: {len(pool)}")
    
    def _get_modality_completion(self, modality):
        """获取指定模态的替代样本"""
        if not self.modality_pools[modality]:
            return None
        
        # 随机选择一个样本
        replacement = random.choice(self.modality_pools[modality])
        return replacement['modality_data']
    
    def _load_modality_images(self, modality_data, modality):
        """加载指定模态的图像"""
        image_paths = modality_data['image_paths']
        
        # 随机选择最多max_images_per_modality张图像
        if len(image_paths) > self.max_images_per_modality:
            image_paths = random.sample(image_paths, self.max_images_per_modality)
        
        # 加载图像
        images = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                
                # 应用转换
                if modality in self.transform:
                    image = self.transform[modality](image)
                else:
                    # 使用默认转换
                    image = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])(image)
                
                images.append(image)
            except Exception as e:
                logger.error(f"加载图像出错: {e}")
                logger.error(f"图像路径: {image_path}")
                continue
        
        return images
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        sample = self.samples[idx]
        
        # 目标标签
        label = self.class_mapping[sample['disease_category']]
        
        # 为每个模态加载图像
        modality_images = {}
        modality_image_counts = {}
        modality_available = {modality: False for modality in self.modalities}
        
        for modality in self.modalities:
            # 检查样本是否包含该模态
            if modality in sample['modalities']:
                modality_data = sample['modality_data'][modality]
                images = self._load_modality_images(modality_data, modality)
                if images:
                    modality_images[modality] = torch.stack(images)
                    modality_image_counts[modality] = len(images)
                    modality_available[modality] = True
            
            # 如果模态缺失或图像加载失败，进行模态补全
            if not modality_available[modality] and self.modality_completion:
                # 获取替代样本
                replacement_data = self._get_modality_completion(modality)
                if replacement_data:
                    images = self._load_modality_images(replacement_data, modality)
                    if images:
                        modality_images[modality] = torch.stack(images)
                        modality_image_counts[modality] = len(images)
                        modality_available[modality] = True
            
            # 如果仍然没有图像（模态补全失败），使用黑色图像
            if not modality_available[modality]:
                logger.warning(f"样本 {sample['id']} 模态 {modality} 没有可用图像")
                # 创建一个黑色图像
                black_img = torch.zeros(3, 224, 224)
                modality_images[modality] = black_img.unsqueeze(0)
                modality_image_counts[modality] = 1
        
        return {
            'id': sample['id'],
            'patient_id': sample['patient_id'],
            'modality_images': modality_images,
            'modality_image_counts': modality_image_counts,
            'modality_available': modality_available,
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultiModalCollator:
    """多模态数据批次整理器"""
    
    def __call__(self, batch):
        """
        将批次样本整合为模型可处理的格式
        
        Args:
            batch (List[Dict]): 样本列表
        
        Returns:
            Dict: 整理后的批次数据
        """
        ids = [item['id'] for item in batch]
        patient_ids = [item['patient_id'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        
        # 获取使用的模态列表
        modalities = list(batch[0]['modality_images'].keys())
        
        # 为每个模态收集所有图像和索引范围
        modality_data = {}
        for modality in modalities:
            all_images = []
            image_ranges = []
            modality_available = []
            start_idx = 0
            
            for item in batch:
                images = item['modality_images'][modality]
                num_images = images.size(0)
                all_images.append(images)
                image_ranges.append((start_idx, start_idx + num_images))
                modality_available.append(item['modality_available'][modality])
                start_idx += num_images
            
            # 将所有图像堆叠成一个张量
            all_images = torch.cat(all_images, dim=0)
            
            modality_data[modality] = {
                'images': all_images,
                'image_ranges': image_ranges,
                'available': torch.tensor(modality_available, dtype=torch.bool)
            }
        
        return {
            'ids': ids,
            'patient_ids': patient_ids,
            'modality_data': modality_data,
            'labels': labels,
            'modalities': modalities
        }

def get_dataloaders(data_dir, multimodal_index_path, modalities=None, batch_size=8, 
                    num_workers=4, pin_memory=True, max_images_per_modality=5,
                    modality_completion=True, return_datasets=False):
    """
    获取多模态数据加载器
    
    Args:
        data_dir: 数据根目录
        multimodal_index_path: 多模态数据集索引文件路径
        modalities: 要使用的模态列表，默认全部使用
        batch_size: 批次大小
        num_workers: 数据加载工作线程数
        pin_memory: 是否将数据固定在内存中（提高GPU训练速度）
        max_images_per_modality: 每个模态最多使用的图像数量
        modality_completion: 是否进行模态补全
        return_datasets: 是否同时返回数据集对象
    
    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    # 分割文件路径
    train_split_path = "/root/autodl-tmp/eye_data_index/splits/train_set.json"
    val_split_path = "/root/autodl-tmp/eye_data_index/splits/val_set.json"
    test_split_path = "/root/autodl-tmp/eye_data_index/splits/test_set.json"
    
    # 创建训练集
    train_dataset = MultiModalEyeDataset(
        data_dir=data_dir,
        multimodal_index_path=multimodal_index_path,
        mode='train',
        modalities=modalities,
        split_path=train_split_path,
        max_images_per_modality=max_images_per_modality,
        modality_completion=modality_completion
    )
    
    # 创建验证集
    val_dataset = MultiModalEyeDataset(
        data_dir=data_dir,
        multimodal_index_path=multimodal_index_path,
        mode='val',
        modalities=modalities,
        split_path=val_split_path,
        max_images_per_modality=max_images_per_modality,
        modality_completion=modality_completion
    )
    
    # 创建测试集
    test_dataset = MultiModalEyeDataset(
        data_dir=data_dir,
        multimodal_index_path=multimodal_index_path,
        mode='test',
        modalities=modalities,
        split_path=test_split_path,
        max_images_per_modality=max_images_per_modality,
        modality_completion=modality_completion
    )
    
    # 创建排序函数，根据样本ID排序
    def collate_fn(batch):
        return MultiModalCollator()(batch)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    if return_datasets:
        return train_loader, val_loader, test_loader, train_dataset
    else:
        return train_loader, val_loader, test_loader