import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModalityFeatureExtractor(nn.Module):
    """单模态特征提取器"""
    
    def __init__(self, modality_name: str = 'generic', feature_dim: int = 512, dropout: float = 0.3):
        """
        初始化单模态特征提取器
        
        Args:
            modality_name (str): 模态名称，用于日志记录
            feature_dim (int): 特征维度
            dropout (float): Dropout比率
        """
        super(ModalityFeatureExtractor, self).__init__()
        self.modality_name = modality_name
        
        # 使用预训练的ResNet18作为特征提取器
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 去掉最后的全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 添加全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 添加特征处理层
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        batch_size = x.size(0)
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(batch_size, -1)
        features = self.fc(features)
        return features

class ModalityAttention(nn.Module):
    """模态间注意力机制"""
    
    def __init__(self, feature_dim: int, num_modalities: int):
        """
        初始化模态间注意力机制
        
        Args:
            feature_dim (int): 特征维度
            num_modalities (int): 模态数量
        """
        super(ModalityAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # 查询、键、值矩阵
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # 输出投影
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        
        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))
    
    def forward(self, features: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features (torch.Tensor): 模态特征 [batch_size, num_modalities, feature_dim]
            modality_mask (torch.Tensor): 模态掩码 [batch_size, num_modalities]
            
        Returns:
            torch.Tensor: 融合后的特征 [batch_size, feature_dim]
        """
        batch_size = features.shape[0]
        
        # 将缩放因子移至设备
        self.scale = self.scale.to(features.device)
        
        # 计算注意力分数
        Q = self.query(features)  # [batch_size, num_modalities, feature_dim]
        K = self.key(features)    # [batch_size, num_modalities, feature_dim]
        V = self.value(features)  # [batch_size, num_modalities, feature_dim]
        
        # 计算注意力权重
        # [batch_size, num_modalities, num_modalities]
        attention = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        
        # 应用模态掩码
        # 创建注意力掩码，将缺失模态的注意力分数设为非常小的负数
        modality_mask_2d = modality_mask.unsqueeze(1) * modality_mask.unsqueeze(2)  # [batch_size, num_modalities, num_modalities]
        attention = attention.masked_fill(modality_mask_2d == 0, -1e10)
        
        # 应用softmax获取注意力权重
        attention = F.softmax(attention, dim=-1)  # [batch_size, num_modalities, num_modalities]
        
        # 应用注意力权重
        # [batch_size, num_modalities, feature_dim]
        x = torch.matmul(attention, V)
        
        # 输出投影
        x = self.fc_out(x)  # [batch_size, num_modalities, feature_dim]
        
        # 使用模态掩码进行加权平均
        # [batch_size, num_modalities, 1]
        modality_weights = modality_mask.float().unsqueeze(-1)
        
        # 如果样本的所有模态都缺失，将权重设为均匀分布
        zero_modality_samples = (modality_weights.sum(1) == 0)
        if zero_modality_samples.any():
            # 对于没有可用模态的样本，使用均匀权重
            uniform_weights = torch.ones_like(modality_weights) / self.num_modalities
            modality_weights = torch.where(
                zero_modality_samples.unsqueeze(1).repeat(1, self.num_modalities, 1),
                uniform_weights,
                modality_weights
            )
        
        # 归一化权重
        modality_weights = modality_weights / (modality_weights.sum(1, keepdim=True) + 1e-8)
        
        # 应用权重并汇总特征
        x = (x * modality_weights).sum(1)  # [batch_size, feature_dim]
        
        return x

class MultiModalFusionModel(nn.Module):
    """多模态融合模型"""
    
    def __init__(self, modalities: List[str] = None, feature_dim: int = 512, 
                 num_classes: int = 3, dropout: float = 0.3, 
                 fusion_method: str = 'attention'):
        """
        初始化多模态融合模型
        
        Args:
            modalities (List[str]): 模态列表，默认为 ['IVCM', 'OCT', '前节照']
            feature_dim (int): 特征维度
            num_classes (int): 类别数量
            dropout (float): Dropout比率
            fusion_method (str): 融合方法，'attention'、'concat' 或 'average'
        """
        super(MultiModalFusionModel, self).__init__()
        
        if modalities is None:
            self.modalities = ['IVCM', 'OCT', '前节照']
        else:
            self.modalities = modalities
        
        self.num_modalities = len(self.modalities)
        self.feature_dim = feature_dim
        self.fusion_method = fusion_method
        
        # 创建每个模态的特征提取器
        self.feature_extractors = nn.ModuleDict()
        for modality in self.modalities:
            self.feature_extractors[modality] = ModalityFeatureExtractor(
                modality_name=modality,
                feature_dim=feature_dim,
                dropout=dropout
            )
        
        # 根据融合方法创建融合层
        if fusion_method == 'attention':
            self.fusion = ModalityAttention(feature_dim, self.num_modalities)
            fusion_output_dim = feature_dim
        elif fusion_method == 'concat':
            # 融合后，特征维度为所有模态特征维度之和
            fusion_output_dim = feature_dim * self.num_modalities
        elif fusion_method == 'average':
            # 简单平均，输出维度不变
            fusion_output_dim = feature_dim
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}, 请使用 'attention', 'concat' 或 'average'")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, fusion_output_dim // 2),
            nn.BatchNorm1d(fusion_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_dim // 2, num_classes)
        )
    
    def extract_modality_features(self, modality_data: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从多模态数据中提取特征
        
        Args:
            modality_data (Dict[str, Dict[str, torch.Tensor]]): 模态数据，包含每个模态的图像和可用性
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 模态特征和模态掩码
        """
        batch_size = len(next(iter(modality_data.values()))['image_ranges'])
        device = next(iter(self.feature_extractors.values())).fc[0].weight.device
        
        # 初始化模态特征和掩码
        modality_features = torch.zeros(batch_size, self.num_modalities, self.feature_dim, device=device)
        modality_mask = torch.zeros(batch_size, self.num_modalities, dtype=torch.bool, device=device)
        
        # 对每个模态进行特征提取
        for idx, modality in enumerate(self.modalities):
            if modality not in modality_data:
                # 该模态在整个批次中都不可用
                continue
            
            # 获取模态数据
            images = modality_data[modality]['images']
            image_ranges = modality_data[modality]['image_ranges']
            available = modality_data[modality]['available'].to(device)
            
            # 如果该模态有可用数据
            if images.size(0) > 0:
                # 将图像移至设备
                images = images.to(device)
                
                # 提取所有图像的特征
                all_image_features = self.feature_extractors[modality](images)
                
                # 对每个样本的图像特征进行汇总
                for i, (start_idx, end_idx) in enumerate(image_ranges):
                    if available[i]:
                        # 获取当前样本的图像特征
                        sample_features = all_image_features[start_idx:end_idx]
                        
                        # 对该样本的所有图像特征取平均
                        avg_features = torch.mean(sample_features, dim=0)
                        
                        # 存储特征
                        modality_features[i, idx] = avg_features
                        
                        # 标记该样本的该模态可用
                        modality_mask[i, idx] = True
        
        return modality_features, modality_mask
    
    def fuse_features(self, modality_features: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        """
        融合多模态特征
        
        Args:
            modality_features (torch.Tensor): 模态特征 [batch_size, num_modalities, feature_dim]
            modality_mask (torch.Tensor): 模态掩码 [batch_size, num_modalities]
            
        Returns:
            torch.Tensor: 融合后的特征 [batch_size, fusion_output_dim]
        """
        if self.fusion_method == 'attention':
            # 使用注意力机制融合特征
            return self.fusion(modality_features, modality_mask)
        
        elif self.fusion_method == 'concat':
            # 拼接特征
            # 对于不可用的模态，使用零向量
            # 返回形状为 [batch_size, num_modalities * feature_dim]
            return modality_features.view(modality_features.size(0), -1)
        
        elif self.fusion_method == 'average':
            # 对可用模态的特征取平均
            # 创建掩码以排除不可用模态 [batch_size, num_modalities, 1]
            mask = modality_mask.float().unsqueeze(-1)
            
            # 对于没有可用模态的样本，使用均匀权重
            zero_modality_samples = (mask.sum(1) == 0)
            if zero_modality_samples.any():
                uniform_weights = torch.ones_like(mask) / self.num_modalities
                mask = torch.where(
                    zero_modality_samples.unsqueeze(1).repeat(1, self.num_modalities, 1),
                    uniform_weights,
                    mask
                )
            
            # 归一化权重
            mask = mask / (mask.sum(1, keepdim=True) + 1e-8)
            
            # 加权平均
            return (modality_features * mask).sum(1)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch (Dict[str, torch.Tensor]): 批次数据
            
        Returns:
            Dict[str, torch.Tensor]: 包含logits和特征的字典
        """
        # 从批次中提取模态数据
        modality_data = batch['modality_data']
        
        # 提取每个模态的特征
        modality_features, modality_mask = self.extract_modality_features(modality_data)
        
        # 融合多模态特征
        fused_features = self.fuse_features(modality_features, modality_mask)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'features': fused_features,
            'modality_features': modality_features,
            'modality_mask': modality_mask
        }
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        预测类别
        
        Args:
            batch (Dict[str, torch.Tensor]): 批次数据
            
        Returns:
            torch.Tensor: 预测的类别索引 [batch_size]
        """
        output = self.forward(batch)
        return torch.argmax(output['logits'], dim=1)

class MultiModalEnsemblePredictor:
    """多模态集成预测器"""
    
    def __init__(self, model: nn.Module, method: str = 'mean', device: torch.device = None):
        """
        初始化多模态集成预测器
        
        Args:
            model (nn.Module): 用于预测的模型
            method (str): 集成方法，'mean' 表示平均概率，'vote' 表示投票
            device (torch.device): 计算设备
        """
        self.model = model
        self.method = method
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        将批次数据移至设备
        
        Args:
            batch (Dict[str, torch.Tensor]): 批次数据
            
        Returns:
            Dict[str, torch.Tensor]: 移至设备的批次数据
        """
        # 深拷贝批次，避免修改原始数据
        device_batch = {}
        
        # 处理模态数据
        if 'modality_data' in batch:
            device_batch['modality_data'] = {}
            for modality, data in batch['modality_data'].items():
                device_batch['modality_data'][modality] = {
                    'images': data['images'].to(self.device),
                    'image_ranges': data['image_ranges'],
                    'available': data['available'].to(self.device)
                }
        
        # 处理标签
        if 'labels' in batch:
            device_batch['labels'] = batch['labels'].to(self.device)
        
        # 处理模态列表
        if 'modalities' in batch:
            device_batch['modalities'] = batch['modalities']
        
        # 处理ID和患者ID
        if 'ids' in batch:
            device_batch['ids'] = batch['ids']
        
        if 'patient_ids' in batch:
            device_batch['patient_ids'] = batch['patient_ids']
        
        return device_batch
    
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预测批次数据
        
        Args:
            batch (Dict[str, torch.Tensor]): 批次数据
            
        Returns:
            Dict[str, torch.Tensor]: 包含预测结果的字典
        """
        # 将批次数据移至设备
        device_batch = self.move_batch_to_device(batch)
        
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(device_batch)
            logits = outputs['logits']
            
            # 对logits取softmax得到概率
            probs = F.softmax(logits, dim=1)
            
            # 预测类别
            preds = torch.argmax(logits, dim=1)
            
            return {
                'preds': preds.cpu(),
                'probs': probs.cpu(),
                'logits': logits.cpu(),
                'features': outputs['features'].cpu() if 'features' in outputs else None,
                'modality_features': outputs['modality_features'].cpu() if 'modality_features' in outputs else None,
                'modality_mask': outputs['modality_mask'].cpu() if 'modality_mask' in outputs else None
            }

def create_multimodal_model(modalities: List[str] = None, feature_dim: int = 512, 
                           num_classes: int = 3, dropout: float = 0.3, 
                           fusion_method: str = 'attention'):
    """
    创建多模态融合模型
    
    Args:
        modalities (List[str]): 模态列表
        feature_dim (int): 特征维度
        num_classes (int): 类别数量
        dropout (float): Dropout比率
        fusion_method (str): 融合方法
        
    Returns:
        MultiModalFusionModel: 创建的模型实例
    """
    return MultiModalFusionModel(
        modalities=modalities,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout=dropout,
        fusion_method=fusion_method
    )