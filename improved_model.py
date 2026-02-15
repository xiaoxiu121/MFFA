import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedModalityFeatureExtractor(nn.Module):
    """增强版单模态特征提取器，使用更强大的骨干网络和多尺度特征"""
    
    def __init__(self, modality_name: str = 'generic', feature_dim: int = 512, 
                 backbone: str = 'resnet50', dropout: float = 0.5):
        """
        初始化增强版单模态特征提取器
        
        Args:
            modality_name (str): 模态名称，用于日志记录
            feature_dim (int): 特征维度
            backbone (str): 特征提取骨干网络类型
            dropout (float): Dropout比率
        """
        super(ImprovedModalityFeatureExtractor, self).__init__()
        self.modality_name = modality_name
        
        # 根据指定的骨干网络类型选择预训练模型
        if backbone == 'resnet18':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            base_features = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            base_features = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            base_features = 2048
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            base_features = 1280
        else:
            # 默认使用ResNet50
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            base_features = 2048
            
        # 去掉最后的全连接层
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # 添加空间注意力模块
        self.spatial_attention = SpatialAttention()
        
        # 添加全局平均池化和最大池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 添加特征处理层
        self.fc = nn.Sequential(
            nn.Linear(base_features * 2, feature_dim),  # 平均池化和最大池化特征拼接
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        batch_size = x.size(0)
        
        # 提取基础特征
        features = self.backbone(x)
        
        # 应用空间注意力
        features = self.spatial_attention(features)
        
        # 全局平均池化和最大池化
        avg_features = self.avgpool(features).view(batch_size, -1)
        max_features = self.maxpool(features).view(batch_size, -1)
        
        # 拼接两种池化特征
        combined_features = torch.cat([avg_features, max_features], dim=1)
        
        # 处理特征
        features = self.fc(combined_features)
        
        return features

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 计算通道维度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接平均值和最大值
        attention = torch.cat([avg_out, max_out], dim=1)
        
        # 应用卷积和sigmoid激活
        attention = self.sigmoid(self.conv(attention))
        
        # 应用注意力
        return x * attention

class ImprovedModalityAttention(nn.Module):
    """增强版模态间注意力机制"""
    
    def __init__(self, feature_dim: int, num_modalities: int, num_heads: int = 4):
        """
        初始化增强版模态间注意力机制
        
        Args:
            feature_dim (int): 特征维度
            num_modalities (int): 模态数量
            num_heads (int): 多头注意力的头数
        """
        super(ImprovedModalityAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 层标准化
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        
        # 模态重要性预测层
        self.modality_importance = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
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
        
        # 创建注意力掩码，将不可用模态标记为-inf
        attn_mask = (modality_mask == 0).float() * -1e9
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_modalities, 1)
        
        # 应用多头自注意力
        attn_features, _ = self.self_attention(
            query=features,
            key=features,
            value=features,
            key_padding_mask=(modality_mask == 0)  # 掩码中的0表示不可用模态
        )
        
        # 残差连接和层标准化
        features = self.layer_norm1(features + attn_features)
        
        # 应用前馈网络
        ff_out = self.feed_forward(features)
        features = self.layer_norm2(features + ff_out)
        
        # 预测模态重要性，并使用sigmoid确保取值在0-1之间
        # [batch_size, num_modalities, 1]
        importance = torch.sigmoid(self.modality_importance(features))
        
        # 应用掩码，将不可用模态的重要性设为0
        modality_mask_expanded = modality_mask.float().unsqueeze(-1)  # [batch_size, num_modalities, 1]
        importance = importance * modality_mask_expanded
        
        # 如果样本的所有模态都缺失，将重要性设为均匀分布
        zero_modality_samples = (modality_mask_expanded.sum(1) == 0)
        if zero_modality_samples.any():
            uniform_importance = torch.ones_like(importance) / self.num_modalities
            importance = torch.where(
                zero_modality_samples.unsqueeze(1).repeat(1, self.num_modalities, 1),
                uniform_importance,
                importance
            )
        
        # 归一化重要性权重
        importance = importance / (importance.sum(1, keepdim=True) + 1e-8)
        
        # 应用重要性权重进行加权汇总
        # [batch_size, feature_dim]
        fused_features = (features * importance).sum(1)
        
        return fused_features

class ImprovedMultiModalFusionModel(nn.Module):
    """增强版多模态融合模型"""
    
    def __init__(self, modalities: List[str] = None, feature_dim: int = 512, 
                 num_classes: int = 3, dropout: float = 0.5, backbone: str = 'resnet50',
                 fusion_method: str = 'attention', use_focal_loss: bool = True,
                 class_weights: List[float] = None):
        """
        初始化增强版多模态融合模型
        
        Args:
            modalities (List[str]): 模态列表，默认为 ['IVCM', 'OCT', '前节照']
            feature_dim (int): 特征维度
            num_classes (int): 类别数量
            dropout (float): Dropout比率
            backbone (str): 特征提取骨干网络类型
            fusion_method (str): 融合方法，'attention'、'concat' 或 'average'
            use_focal_loss (bool): 是否使用Focal Loss
            class_weights (List[float]): 类别权重，用于加权损失函数
        """
        super(ImprovedMultiModalFusionModel, self).__init__()
        
        if modalities is None:
            self.modalities = ['IVCM', 'OCT', '前节照']
        else:
            self.modalities = modalities
        
        self.num_modalities = len(self.modalities)
        self.feature_dim = feature_dim
        self.fusion_method = fusion_method
        self.use_focal_loss = use_focal_loss
        self.class_weights = class_weights
        
        # 创建每个模态的特征提取器
        self.feature_extractors = nn.ModuleDict()
        for modality in self.modalities:
            self.feature_extractors[modality] = ImprovedModalityFeatureExtractor(
                modality_name=modality,
                feature_dim=feature_dim,
                backbone=backbone,
                dropout=dropout
            )
        
        # 根据融合方法创建融合层
        if fusion_method == 'attention':
            self.fusion = ImprovedModalityAttention(feature_dim, self.num_modalities, num_heads=4)
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
            nn.Linear(fusion_output_dim, fusion_output_dim),
            nn.BatchNorm1d(fusion_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_dim, fusion_output_dim // 2),
            nn.BatchNorm1d(fusion_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_dim // 2, num_classes)
        )
        
        # 初始化Focal Loss的参数
        if use_focal_loss:
            self.focal_loss_gamma = 2.0
            self.focal_loss_alpha = torch.tensor([1.0, 1.0, 2.0])  # 增加other类别权重
    
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
                        
                        # 对该样本的所有图像特征取加权平均
                        # 为特征分配权重 - 可以考虑使用注意力机制
                        weights = F.softmax(torch.matmul(sample_features, sample_features.transpose(0, 1)), dim=1)
                        avg_features = torch.matmul(weights, sample_features).mean(0)
                        
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

    def calculate_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            logits (torch.Tensor): 模型输出的logits [batch_size, num_classes]
            targets (torch.Tensor): 目标标签 [batch_size]
            
        Returns:
            torch.Tensor: 计算得到的Focal Loss
        """
        # 设置设备
        device = logits.device
        alpha = self.focal_loss_alpha.to(device) if hasattr(self, 'focal_loss_alpha') else None
        
        # 计算softmax概率
        probs = F.softmax(logits, dim=1)
        
        # 获取目标类别的概率
        batch_size = logits.size(0)
        class_probs = probs[torch.arange(batch_size), targets]
        
        # Focal Loss公式：-alpha * (1-p)^gamma * log(p)
        focal_weight = (1 - class_probs).pow(self.focal_loss_gamma)
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 应用Focal Loss权重
        focal_loss = focal_weight * ce_loss
        
        # 如果提供了alpha，则应用类别权重
        if alpha is not None:
            alpha_weight = alpha[targets]
            focal_loss = alpha_weight * focal_loss
        
        # 计算批次平均损失
        return focal_loss.mean()

class ImprovedMultiModalEnsemblePredictor:
    """增强版多模态集成预测器"""
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None, 
                 device: torch.device = None, temperature: float = 1.0,
                 threshold: float = 0.6):
        """
        初始化增强版多模态集成预测器
        
        Args:
            models (List[nn.Module]): 用于集成的模型列表
            weights (List[float]): 每个模型的权重，如果为None则均等权重
            device (torch.device): 计算设备
            temperature (float): softmax温度参数，用于调整概率分布
            threshold (float): 分类阈值，低于此值的预测将被视为不确定
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.threshold = threshold
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # 将所有模型设置为评估模式并移至设备
        for model in self.models:
            model.to(self.device)
            model.eval()
    
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
        为批次数据进行集成预测
        
        Args:
            batch (Dict[str, torch.Tensor]): 批次数据
            
        Returns:
            Dict[str, torch.Tensor]: 包含预测结果的字典
        """
        # 将批次数据移至设备
        device_batch = self.move_batch_to_device(batch)
        
        all_logits = []
        all_probs = []
        
        with torch.no_grad():
            # 获取每个模型的输出
            for i, model in enumerate(self.models):
                outputs = model(device_batch)
                logits = outputs['logits']
                
                # 应用温度缩放
                scaled_logits = logits / self.temperature
                
                # 计算概率
                probs = F.softmax(scaled_logits, dim=1)
                
                # 应用模型权重
                weighted_probs = probs * self.weights[i]
                
                all_logits.append(logits)
                all_probs.append(weighted_probs)
            
            # 合并所有模型的预测
            ensemble_probs = sum(all_probs)
            
            # 获取最大概率值和对应的类别
            max_probs, preds = torch.max(ensemble_probs, dim=1)
            
            # 处理不确定的预测
            uncertain_mask = max_probs < self.threshold
            if uncertain_mask.any():
                # 对于不确定的预测，使用加权logits
                ensemble_logits = sum([all_logits[i] * self.weights[i] for i in range(len(self.models))])
                # 重新预测不确定样本的类别
                _, uncertain_preds = torch.max(ensemble_logits[uncertain_mask], dim=1)
                # 更新预测结果
                preds[uncertain_mask] = uncertain_preds
            
            return {
                'preds': preds.cpu(),
                'probs': ensemble_probs.cpu(),
                'logits': all_logits[0].cpu(),  # 返回第一个模型的logits作为代表
                'features': outputs['features'].cpu() if 'features' in outputs else None,
                'modality_features': outputs['modality_features'].cpu() if 'modality_features' in outputs else None,
                'modality_mask': outputs['modality_mask'].cpu() if 'modality_mask' in outputs else None,
                'uncertainty': (~uncertain_mask).float().cpu()  # 1表示确定，0表示不确定
            }

def create_improved_multimodal_model(modalities: List[str] = None, feature_dim: int = 512, 
                                    num_classes: int = 3, dropout: float = 0.5, 
                                    backbone: str = 'resnet50',
                                    fusion_method: str = 'attention',
                                    use_focal_loss: bool = True,
                                    class_weights: List[float] = None):
    """
    创建增强版多模态融合模型
    
    Args:
        modalities (List[str]): 模态列表
        feature_dim (int): 特征维度
        num_classes (int): 类别数量
        dropout (float): Dropout比率
        backbone (str): 特征提取骨干网络类型
        fusion_method (str): 融合方法
        use_focal_loss (bool): 是否使用Focal Loss
        class_weights (List[float]): 类别权重
        
    Returns:
        ImprovedMultiModalFusionModel: 创建的模型实例
    """
    return ImprovedMultiModalFusionModel(
        modalities=modalities,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout=dropout,
        backbone=backbone,
        fusion_method=fusion_method,
        use_focal_loss=use_focal_loss,
        class_weights=class_weights
    )

# Focal Loss实现
class FocalLoss(nn.Module):
    """
    Focal Loss，用于处理类别不平衡问题
    
    参考: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha (torch.Tensor, optional): 每个类别的权重，形状为 [num_classes]
            gamma (float): 聚焦参数，增加此值会增加对难分类样本的关注
            reduction (str): 'none', 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        计算Focal Loss
        
        Args:
            inputs (torch.Tensor): 模型输出的logits，形状为 [batch_size, num_classes]
            targets (torch.Tensor): 目标标签，形状为 [batch_size]
            
        Returns:
            torch.Tensor: 计算得到的loss
        """
        # 设置设备
        device = inputs.device
        
        # 计算softmax概率
        probs = F.softmax(inputs, dim=1)
        
        # 获取目标类别的概率
        batch_size = inputs.size(0)
        class_probs = probs[torch.arange(batch_size), targets]
        
        # 计算Focal Loss权重
        focal_weight = (1 - class_probs).pow(self.gamma)
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 应用Focal Loss权重
        focal_loss = focal_weight * ce_loss
        
        # 如果提供了alpha，则应用类别权重
        if self.alpha is not None:
            alpha = self.alpha.to(device)
            alpha_weight = alpha[targets]
            focal_loss = alpha_weight * focal_loss
        
        # 根据reduction方式返回结果
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"不支持的reduction方式: {self.reduction}") 