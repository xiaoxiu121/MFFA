import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import argparse
import logging
import random
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from collections import Counter

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入自定义模块
from dataset import get_dataloaders
from model import MultiModalFusionModel, MultiModalEnsemblePredictor, create_multimodal_model

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class FocalLoss(nn.Module):
    """Focal Loss实现，针对难分样本和类别不平衡问题"""
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
    def forward(self, inputs, targets):
        """
        计算Focal Loss
        Args:
            inputs: 模型输出的logits, shape [N, C]
            targets: 目标类别索引, shape [N]
        """
        # 计算标准交叉熵损失
        ce_loss = self.ce_loss(inputs, targets)
        
        # 计算softmax概率
        probs = torch.softmax(inputs, dim=1)
        
        # 获取目标类别的概率
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算focal loss
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # 根据reduction方式处理结果
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class OtherClassAugmentation:
    """针对Other类的特定增强策略"""
    def __init__(self, dataset, transform_strength=1.5):
        """
        初始化Other类增强器
        Args:
            dataset: 数据集对象
            transform_strength: 增强强度系数
        """
        self.dataset = dataset
        self.strength = transform_strength
        self.base_transform = dataset.transform
        
        # 创建更强的Other类数据增强
        from torchvision import transforms
        self.other_transform = {}
        for modality in self.base_transform.keys():
            self.other_transform[modality] = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.7),  # 增加水平翻转概率
                transforms.RandomVerticalFlip(p=0.3),    # 添加垂直翻转
                transforms.RandomRotation(20),           # 增加旋转角度
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # 增强颜色抖动
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 添加仿射变换
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def apply(self):
        """应用Other类增强策略"""
        # 保存原始transform
        original_transform = self.dataset.transform
        
        # 获取数据集中的Other类样本索引
        other_indices = [i for i, sample in enumerate(self.dataset.samples) 
                         if sample['disease_category'] == 'other']
        
        logger.info(f"应用增强数据增强到{len(other_indices)}个Other类样本")
        
        # 创建一个自定义__getitem__方法，为Other类应用特定的增强
        original_getitem = self.dataset.__getitem__
        
        def augmented_getitem(self, idx):
            sample = self.samples[idx]
            
            # 如果是Other类，使用增强的transform
            if sample['disease_category'] == 'other':
                # 临时替换transform
                temp = self.transform
                self.transform = self.other_transform
                
                # 获取样本
                result = original_getitem(idx)
                
                # 恢复原始transform
                self.transform = temp
                return result
            else:
                return original_getitem(idx)
        
        # 替换数据集的__getitem__方法
        import types
        self.dataset.__getitem__ = types.MethodType(augmented_getitem, self.dataset)
        
        return self.dataset

def calculate_class_weights(class_counts, beta=0.99, other_boost=2.0):
    """
    计算类别权重，适用于不平衡数据集
    使用有效样本数(ENS)方法计算权重，这比简单的逆频率效果更好
    
    Args:
        class_counts: 各类别样本数量列表[count_0, count_1, count_2]
        beta: 平滑参数(0.9-0.999)，越高对少数类的权重越大
        other_boost: Other类的额外权重倍数
    
    Returns:
        class_weights: 类别权重张量
    """
    logger.info(f"类别样本分布: FECD={class_counts[0]}, Normal={class_counts[1]}, Other={class_counts[2]}")
    
    # 计算有效样本数
    n_samples = sum(class_counts)
    effective_num = 1.0 - np.power(beta, class_counts)
    
    # 计算权重
    weights = (1.0 - beta) / effective_num
    
    # 归一化权重，使其总和等于类别数
    weights = weights / np.sum(weights) * len(class_counts)
    
    # 特别提高Other类(索引2)的权重
    weights[2] *= other_boost  # 额外增加Other类的权重
    
    logger.info(f"计算得到的原始权重: {weights}")
    
    # 将numpy数组转换为PyTorch张量
    return torch.FloatTensor(weights)

class CustomLRScheduler:
    """自定义学习率调度器，结合余弦退火和早期线性预热"""
    def __init__(self, optimizer, initial_lr, min_lr, warmup_epochs=2, total_epochs=30, patience=5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        # 用于ReduceLROnPlateau行为
        self.patience = patience
        self.best_metric = -float('inf')
        self.bad_epochs = 0
        self.factor = 0.5
        
        self.current_epoch = 0
    
    def step(self, epoch=None, metric=None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        
        # 1. 预热阶段 - 线性增加学习率
        if self.current_epoch < self.warmup_epochs:
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        
        # 2. 退火阶段 - 余弦退火，同时考虑性能停滞
        else:
            # 检查是否应该降低学习率
            if metric is not None:
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.bad_epochs = 0
                else:
                    self.bad_epochs += 1
                    if self.bad_epochs >= self.patience:
                        # 降低学习率
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
                        logger.info(f"学习率降低到: {self.optimizer.param_groups[0]['lr']:.6f}")
                        self.bad_epochs = 0
            
            # 余弦退火的计算
            if self.current_epoch < self.total_epochs:
                progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
                
                # 确保不低于最小学习率
                lr = max(lr, self.min_lr)
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    def state_dict(self):
        """返回调度器状态"""
        return {
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'bad_epochs': self.bad_epochs
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.current_epoch = state_dict['current_epoch']
        self.best_metric = state_dict['best_metric']
        self.bad_epochs = state_dict['bad_epochs']

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="训练", ascii=True)
    for batch in progress_bar:
        # 从批次中提取标签
        labels = batch['labels'].to(device)
        
        # 使用混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # 前向传播
                outputs = model(batch)
                logits = outputs['logits']
                
                # 计算损失
                loss = criterion(logits, labels)
            
            # 反向传播和优化（使用梯度缩放器）
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播（无混合精度）
            outputs = model(batch)
            logits = outputs['logits']
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 累积统计信息
        epoch_loss += loss.item()
        
        # 计算预测结果
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix(loss=loss.item())
    
    # 计算整体指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 计算每个类别的详细指标
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    
    # 构建类别指标字典
    class_metrics = {
        'FECD': {
            'precision': class_precision[0],
            'recall': class_recall[0],
            'f1': class_f1[0],
            'support': int(class_support[0])
        },
        'Normal': {
            'precision': class_precision[1],
            'recall': class_recall[1],
            'f1': class_f1[1],
            'support': int(class_support[1])
        },
        'Other': {
            'precision': class_precision[2],
            'recall': class_recall[2],
            'f1': class_f1[2],
            'support': int(class_support[2])
        }
    }
    
    return {
        'loss': epoch_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels,
        'class_metrics': class_metrics
    }

def validate(model, dataloader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    all_modality_masks = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="验证", ascii=True)
        for batch in progress_bar:
            # 从批次中提取标签
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(batch)
            logits = outputs['logits']
            
            # 提取模态掩码
            if 'modality_mask' in outputs:
                all_modality_masks.extend(outputs['modality_mask'].cpu().numpy())
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 累积统计信息
            val_loss += loss.item()
            
            # 计算预测结果
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())
    
    # 计算整体指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 计算每个类别的详细指标
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    
    # 构建类别指标字典
    class_metrics = {
        'FECD': {
            'precision': class_precision[0],
            'recall': class_recall[0],
            'f1': class_f1[0],
            'support': int(class_support[0])
        },
        'Normal': {
            'precision': class_precision[1],
            'recall': class_recall[1],
            'f1': class_f1[1],
            'support': int(class_support[1])
        },
        'Other': {
            'precision': class_precision[2],
            'recall': class_recall[2],
            'f1': class_f1[2],
            'support': int(class_support[2])
        }
    }
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    return {
        'loss': val_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels,
        'class_metrics': class_metrics,
        'confusion_matrix': cm,
        'modality_masks': all_modality_masks
    }

def test(model, dataloader, device, output_dir=None):
    """在测试集上评估模型并保存结果"""
    predictor = MultiModalEnsemblePredictor(model, method='mean', device=device)
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []
    all_ids = []
    
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="测试", ascii=True)
        for batch in progress_bar:
            # 收集样本IDs和标签
            ids = batch.get('ids', ["unknown"] * len(batch['labels']))
            patient_ids = batch.get('patient_ids', ["unknown"] * len(batch['labels']))
            labels = batch['labels'].numpy()
            
            # 预测
            results = predictor.predict_batch(batch)
            preds = results['preds'].numpy()
            probs = results['probs'].numpy()
            
            # 累积结果
            all_ids.extend(ids)
            all_patient_ids.extend(patient_ids)
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)
    
    # 计算整体指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 计算每个类别的详细指标
    class_names = ['FECD', 'Normal', 'Other']
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    
    # 构建类别指标字典
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': float(class_precision[i]),
            'recall': float(class_recall[i]),
            'f1': float(class_f1[i]),
            'support': int(class_support[i])
        }
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    # 如果指定了输出目录，则保存测试结果
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'id': all_ids,
            'patient_id': all_patient_ids,
            'true_label': all_labels,
            'predicted_label': all_preds
        })
        
        # 添加类别概率列
        all_probs = np.array(all_probs)
        for i, class_name in enumerate(class_names):
            results_df[f'{class_name}_prob'] = all_probs[:, i]
        
        # 添加类别名称
        results_df['true_label_name'] = results_df['true_label'].map({i: name for i, name in enumerate(class_names)})
        results_df['predicted_label_name'] = results_df['predicted_label'].map({i: name for i, name in enumerate(class_names)})
        
        # 保存CSV
        results_df.to_csv(output_dir / 'test_predictions.csv', index=False)
        
        # 保存指标
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist()
        }
        
        with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('测试集混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(output_dir / 'test_confusion_matrix.png')
        plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics,
        'confusion_matrix': cm
    }

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_metric = checkpoint.get('best_metric', 0.0)
    
    return model, optimizer, scheduler, epoch, best_metric

def create_visualizations(train_metrics, val_metrics, output_dir):
    """创建训练过程可视化图表"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    
    plt.figure(figsize=(20, 15))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        plt.plot(train_metrics[metric], label=f'训练集 {metric}')
        plt.plot(val_metrics[metric], label=f'验证集 {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} 随时间变化')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png')
    plt.close()

def main(args):
    """主函数"""
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 预先定义modalities_str以便在整个函数中使用
    modalities_str = '_'.join(args.modalities.split(','))
    logger.info(f"使用模态: {modalities_str}")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        data_dir=args.data_dir,
        multimodal_index_path=args.multimodal_index_path,
        modalities=args.modalities.split(','),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images_per_modality=args.max_images_per_modality,
        modality_completion=args.modality_completion,
        return_datasets=True  # 返回数据集对象以便应用自定义增强
    )
    
    # 应用Other类的特定增强 - 简化操作以避免卡住
    logger.info("应用Other类的特定增强...")
    other_augmentor = OtherClassAugmentation(train_dataset)
    train_dataset = other_augmentor.apply()
    
    # 计算类别权重来平衡数据集 - 用更高效的方法统计类别分布
    logger.info("计算类别权重...")
    class_counts = [0, 0, 0]  # [FECD, Normal, Other]
    
    # 使用样本的disease_category直接计数，避免遍历整个数据集
    for sample in train_dataset.samples:
        category = sample['disease_category'].lower()
        if category == 'fecd':
            class_counts[0] += 1
        elif category == 'normal':
            class_counts[1] += 1
        elif category == 'other':
            class_counts[2] += 1
    
    class_weights = calculate_class_weights(class_counts, beta=0.99, other_boost=2.0)
    class_weights = class_weights.to(device)
    logger.info(f"最终类别权重: {class_weights}")
    
    # 创建多模态融合模型
    logger.info("创建多模态融合模型...")
    model = create_multimodal_model(
        modalities=args.modalities.split(','),
        fusion_method=args.fusion_method,
        feature_dim=args.feature_dim,
        num_classes=3  # FECD, Normal, Other
    )
    
    model = model.to(device)
    
    # 使用具有权重衰减的AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 使用自定义学习率调度器
    scheduler = CustomLRScheduler(
        optimizer, 
        initial_lr=args.lr,
        min_lr=args.lr * 0.01,
        warmup_epochs=2,
        total_epochs=args.epochs,
        patience=3
    )
    
    # 使用Focal Loss而不是标准交叉熵损失
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    logger.info(f"使用Focal Loss，gamma=2.0，类别权重={class_weights}")
    
    # 使用混合精度训练以提高速度
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # 初始化训练变量
    start_epoch = 0
    best_f1 = 0.0
    best_other_recall = 0.0
    patience_counter = 0
    patience = 8  # 早停等待轮数
    
    # 记录训练过程中的指标
    train_metrics_history = {metric: [] for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']}
    val_metrics_history = {metric: [] for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']}
    
    # 如果指定了恢复训练
    if args.resume and args.model_path:
        logger.info(f"从检查点恢复训练: {args.model_path}")
        model, optimizer, scheduler, start_epoch, best_f1 = load_checkpoint(
            model, optimizer, scheduler, args.model_path
        )
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"当前学习率: {current_lr:.6f}")
        
        # 训练一个epoch
        logger.info("开始训练...")
        train_result = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # 在验证集上评估
        logger.info("开始验证...")
        val_result = validate(model, val_loader, criterion, device)
        
        # 更新学习率调度器
        scheduler.step(metric=val_result['f1'])
        
        # 记录指标
        for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            train_metrics_history[metric].append(train_result[metric])
            val_metrics_history[metric].append(val_result[metric])
        
        # 打印当前epoch的训练和验证结果
        logger.info(f"训练 Loss: {train_result['loss']:.4f}, 准确率: {train_result['accuracy']:.4f}, F1: {train_result['f1']:.4f}")
        logger.info(f"验证 Loss: {val_result['loss']:.4f}, 准确率: {val_result['accuracy']:.4f}, F1: {val_result['f1']:.4f}")
        
        # 打印每个类别的指标
        logger.info("训练集各类别指标:")
        for class_name, metrics in train_result['class_metrics'].items():
            logger.info(f"  {class_name} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 样本数: {metrics['support']}")
        
        logger.info("验证集各类别指标:")
        for class_name, metrics in val_result['class_metrics'].items():
            logger.info(f"  {class_name} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 样本数: {metrics['support']}")
        
        # 跟踪Other类的召回率
        other_recall = val_result['class_metrics']['Other']['recall']
        other_f1 = val_result['class_metrics']['Other']['f1']
        
        # 保存本次epoch的检查点 (每5个epoch保存一次，节省空间)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}_{modalities_str}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, val_result['f1'], checkpoint_path)
            logger.info(f"保存检查点: {checkpoint_path}")
        
        # 如果验证F1分数提高或Other类指标显著提高，保存最佳模型
        current_f1 = val_result['f1']
        save_new_best = False
        
        # 判断条件1: 整体F1提高
        if current_f1 > best_f1:
            save_new_best = True
            reason = f"整体F1提高: {current_f1:.4f} > {best_f1:.4f}"
        
        # 判断条件2: 整体F1稍微下降但Other类指标大幅提高
        elif current_f1 >= best_f1 * 0.97 and other_recall > best_other_recall * 1.05:
            save_new_best = True
            reason = f"Other召回率大幅提高: {other_recall:.4f} > {best_other_recall*1.05:.4f}，整体F1: {current_f1:.4f}"
        
        # 判断条件3: Other F1分数显著提高
        elif other_f1 > 0.85 and current_f1 >= best_f1 * 0.95:
            save_new_best = True
            reason = f"Other F1分数高: {other_f1:.4f}，整体F1: {current_f1:.4f}"
        
        if save_new_best:
            best_f1 = max(current_f1, best_f1)
            best_other_recall = max(other_recall, best_other_recall)
            
            logger.info(f"保存新的最佳模型: {reason}")
            best_model_path = output_dir / f"{modalities_str}_best_model.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_model_path)
            
            # 重置早停计数器
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"模型性能未提高，早停计数: {patience_counter}/{patience}")
            
        # 早停逻辑 - 考虑Other类指标
        if patience_counter >= patience and other_recall >= 0.85:
            logger.info(f"验证集F1分数连续{patience}个epoch没有提高，且Other类召回率已达标，提前停止训练")
            break
        elif patience_counter >= patience + 4:  # 额外宽限期
            logger.info(f"验证集F1分数连续{patience+4}个epoch没有提高，强制提前停止训练")
            break
        
        # 保存当前训练进度的可视化图表
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            create_visualizations(train_metrics_history, val_metrics_history, output_dir)
    
    # 创建训练过程可视化
    create_visualizations(train_metrics_history, val_metrics_history, output_dir)
    
    # 测试最佳模型
    logger.info("测试最佳模型...")
    best_model_path = output_dir / f"{modalities_str}_best_model.pt"
    model, optimizer, scheduler, _, _ = load_checkpoint(model, optimizer, scheduler, best_model_path)
    
    test_metrics = test(model, test_loader, device, output_dir)
    
    # 打印测试结果
    logger.info(f"测试准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"测试F1分数: {test_metrics['f1']:.4f}")
    
    logger.info("测试集各类别指标:")
    for class_name, metrics in test_metrics['class_metrics'].items():
        logger.info(f"  {class_name} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 样本数: {metrics['support']}")
    
    # 保存训练配置
    config = vars(args)
    with open(output_dir / 'training_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    logger.info(f"训练完成，最佳模型和结果保存到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='眼科多模态疾病分类模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据根目录')
    parser.add_argument('--multimodal_index_path', type=str, required=True, help='多模态数据集索引文件路径')
    parser.add_argument('--modalities', type=str, default='IVCM,OCT,前节照', help='要使用的模态列表，用逗号分隔')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作线程数')
    parser.add_argument('--max_images_per_modality', type=int, default=5, help='每个模态使用的最大图像数量')
    parser.add_argument('--modality_completion', action='store_true', help='是否进行模态补全')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=512, help='特征维度')
    parser.add_argument('--fusion_method', type=str, default='attention', choices=['attention', 'concat', 'average'], help='模态融合方法')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练周期数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    parser.add_argument('--model_path', type=str, default=None, help='要加载的模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    
    args = parser.parse_args()
    
    main(args) 