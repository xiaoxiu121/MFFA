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
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入自定义模块
from dataset import get_dataloaders
from improved_model import create_improved_multimodal_model, FocalLoss, ImprovedMultiModalEnsemblePredictor

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
    torch.backends.cudnn.benchmark = False  # 提高稳定性

def create_class_weight_sampler(dataset, num_classes=3):
    """创建类别加权采样器，提高对小样本类别的采样率"""
    # 统计各类样本数量
    class_counts = [0] * num_classes
    # 直接从 dataset.samples 和 dataset.class_mapping 获取标签信息
    for sample_metadata in dataset.samples:
        label_str = sample_metadata['disease_category']
        label_idx = dataset.class_mapping[label_str]
        class_counts[label_idx] += 1
    
    # 计算权重
    weights = [1.0 / (count if count > 0 else 1) for count in class_counts]
    logger.info(f"类别样本数: {class_counts}")
    logger.info(f"类别权重: {weights}")
    
    # 创建每个样本的权重
    sample_weights = [0] * len(dataset.samples)
    # 直接从 dataset.samples 和 dataset.class_mapping 获取标签信息
    for i, sample_metadata in enumerate(dataset.samples):
        label_str = sample_metadata['disease_category']
        label_idx = dataset.class_mapping[label_str]
        sample_weights[i] = weights[label_idx]
    
    # 创建采样器
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, class_weights=None):
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
                if hasattr(model, 'use_focal_loss') and model.use_focal_loss:
                    loss = model.calculate_focal_loss(logits, labels)
                else:
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
            if hasattr(model, 'use_focal_loss') and model.use_focal_loss:
                loss = model.calculate_focal_loss(logits, labels)
            else:
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

def validate(model, dataloader, criterion, device, threshold=0.5):
    """在验证集上评估模型"""
    model.eval()
    val_loss = 0
    all_preds = []
    all_probs = []
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
            
            # 计算概率
            probs = torch.softmax(logits, dim=1)
            
            # 提取模态掩码
            if 'modality_mask' in outputs:
                all_modality_masks.extend(outputs['modality_mask'].cpu().numpy())
            
            # 计算损失
            if hasattr(model, 'use_focal_loss') and model.use_focal_loss:
                loss = model.calculate_focal_loss(logits, labels)
            else:
                loss = criterion(logits, labels)
            
            # 累积统计信息
            val_loss += loss.item()
            
            # 增加对Other类的阈值要求，提高精确率
            # 如果Other类(idx=2)的概率不够高，就改为预测第二高的类别
            max_probs, preds = torch.max(probs, dim=1)
            other_mask = (preds == 2) & (probs[:, 2] < threshold)
            if other_mask.any():
                # 创建一个排除Other类的probs副本
                probs_no_other = probs.clone()
                probs_no_other[:, 2] = -float('inf')
                # 重新对这些样本进行预测
                _, new_preds = torch.max(probs_no_other[other_mask], dim=1)
                preds[other_mask] = new_preds
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
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
        'probs': all_probs,
        'labels': all_labels,
        'class_metrics': class_metrics,
        'confusion_matrix': cm,
        'modality_masks': all_modality_masks
    }

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path, other_precision=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
        'other_precision': other_precision,
        'modalities': model.modalities
    }
    torch.save(checkpoint, save_path)
    logger.info(f"检查点已保存到 {save_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        logger.info(f"检查点 {checkpoint_path} 不存在，从头开始训练")
        return 0, 0, 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    other_precision = checkpoint.get('other_precision', 0)
    
    logger.info(f"从检查点 {checkpoint_path} 加载，上次最佳指标: {best_metric:.4f}, "
               f"other类精确率: {other_precision:.4f}, 当前周期: {epoch}")
    return epoch, best_metric, other_precision

def create_visualizations(train_metrics, val_metrics, output_dir):
    """创建训练和验证曲线图"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_metrics) + 1), [m[metric] for m in train_metrics], label='训练')
        plt.plot(range(1, len(val_metrics) + 1), [m[metric] for m in val_metrics], label='验证')
        plt.xlabel('周期')
        plt.ylabel(metric)
        plt.title(f'训练和验证 {metric} 曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{metric}_curve.png'))
        plt.close()
    
    # 创建各类别指标曲线
    class_names = ['FECD', 'Normal', 'Other']
    metrics = ['precision', 'recall', 'f1']
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        for class_name in class_names:
            # 训练集数据
            train_values = [m['class_metrics'][class_name][metric] for m in train_metrics]
            plt.plot(range(1, len(train_metrics) + 1), train_values, 
                    label=f'{class_name} 训练', linestyle='-')
            
            # 验证集数据
            val_values = [m['class_metrics'][class_name][metric] for m in val_metrics]
            plt.plot(range(1, len(val_metrics) + 1), val_values, 
                    label=f'{class_name} 验证', linestyle='--')
        
        plt.xlabel('周期')
        plt.ylabel(metric)
        plt.title(f'各类别 {metric} 曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'class_{metric}_curve.png'))
        plt.close()

def find_best_threshold(val_results):
    """
    使用验证集结果找到使other类别precision最高的阈值
    """
    probs = np.array(val_results['probs'])
    labels = np.array(val_results['labels'])
    
    best_threshold = 0.5
    best_precision = 0
    best_overall_f1 = 0
    
    for threshold in np.arange(0.5, 0.95, 0.05):
        # 创建预测结果的副本
        preds = np.argmax(probs, axis=1)
        
        # 修改other类的预测
        other_mask = (preds == 2) & (probs[:, 2] < threshold)
        if other_mask.any():
            # 创建一个排除other类的probs副本
            probs_no_other = probs.copy()
            probs_no_other[:, 2] = -float('inf')
            # 重新对这些样本进行预测
            new_preds = np.argmax(probs_no_other[other_mask], axis=1)
            preds[other_mask] = new_preds
        
        # 计算指标
        _, _, _, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # 计算other类的precision
        class_precision, _, class_f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        other_precision = class_precision[2]
        overall_f1 = np.mean(class_f1)  # 使用所有类别的平均F1
        
        logger.info(f"阈值 {threshold:.2f}: other精确率 = {other_precision:.4f}, 总体F1 = {overall_f1:.4f}")
        
        # 更新最佳阈值，平衡other精确率和总体F1
        if other_precision >= 0.9 and overall_f1 > best_overall_f1:
            best_threshold = threshold
            best_precision = other_precision
            best_overall_f1 = overall_f1
    
    logger.info(f"最佳阈值: {best_threshold:.2f}, other精确率: {best_precision:.4f}, 总体F1: {best_overall_f1:.4f}")
    
    return best_threshold

def main(args):
    """主函数"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志文件
    log_file = output_dir / f"{args.modalities_str}_training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 记录参数
    logger.info(f"训练参数: {args}")
    
    # 配置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 获取数据加载器和数据集
    _train_loader, _val_loader, _test_loader, train_dataset = get_dataloaders(
        data_dir=args.data_dir,
        multimodal_index_path=args.multimodal_index_path,
        modalities=args.modalities,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images_per_modality=args.max_images_per_modality,
        modality_completion=args.modality_completion,
        return_datasets=True
    )
    dataloaders = [_train_loader, _val_loader, _test_loader]
    
    # 为训练集创建类别加权采样器
    if args.use_weighted_sampler:
        logger.info("使用类别加权采样器重新创建训练数据加载器")
        sampler = create_class_weight_sampler(train_dataset, num_classes=args.num_classes)
        
        # 重新创建训练数据加载器，使用加权采样器
        dataloaders[0] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataloaders[0].collate_fn
        )
    
    # 创建模型
    model = create_improved_multimodal_model(
        modalities=args.modalities,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        backbone=args.backbone,
        fusion_method=args.fusion_method,
        use_focal_loss=args.use_focal_loss,
        class_weights=[1.0, 1.0, args.other_class_weight]  # 增加other类别权重
    )
    model.to(device)
    
    # 打印模型信息
    logger.info(f"模型结构:\n{model}")
    
    # 定义损失函数
    if args.use_focal_loss:
        # 使用模型内部的Focal Loss
        criterion = None
        logger.info(f"使用Focal Loss，other类别权重: {args.other_class_weight}")
    else:
        # 使用交叉熵损失，为类别分配权重
        weights = torch.tensor([1.0, 1.0, args.other_class_weight]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info(f"使用加权交叉熵损失，other类别权重: {args.other_class_weight}")
    
    # 定义优化器
    # 使用AdamW优化器，更好的权重衰减处理
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 定义学习率调度器
    # 使用余弦退火调度器，有助于跳出局部最优解
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.cosine_annealing_t0, T_mult=args.cosine_annealing_tmult, eta_min=args.min_lr
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # 记录训练和验证指标
    train_metrics = []
    val_metrics = []
    
    # 加载检查点（如果存在）
    checkpoint_path = output_dir / f"{args.modalities_str}_best_model.pt"
    start_epoch, best_metric, best_other_precision = load_checkpoint(model, optimizer, scheduler, checkpoint_path) if args.resume else (0, 0, 0)
    
    # 训练循环
    logger.info("开始训练...")
    best_threshold = 0.5  # 默认阈值
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        logger.info(f"周期 {epoch}/{args.epochs}")
        
        # 训练一个周期
        logger.info("训练阶段")
        train_result = train_epoch(model, dataloaders[0], criterion, optimizer, device, scaler)
        train_metrics.append(train_result)
        
        # 验证
        logger.info("验证阶段")
        val_result = validate(model, dataloaders[1], criterion, device, threshold=best_threshold)
        val_metrics.append(val_result)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"当前学习率: {current_lr:.6f}")
        
        # 打印和记录指标
        logger.info(f"训练 Loss: {train_result['loss']:.4f}, 准确率: {train_result['accuracy']:.4f}, F1: {train_result['f1']:.4f}")
        logger.info(f"验证 Loss: {val_result['loss']:.4f}, 准确率: {val_result['accuracy']:.4f}, F1: {val_result['f1']:.4f}")
        
        # 打印每个类别的详细指标
        logger.info("训练集各类别指标:")
        for class_name, metrics in train_result['class_metrics'].items():
            logger.info(f"  {class_name} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 样本数: {metrics['support']}")
        
        logger.info("验证集各类别指标:")
        for class_name, metrics in val_result['class_metrics'].items():
            logger.info(f"  {class_name} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 样本数: {metrics['support']}")
        
        # 检查other类别精确率是否已达到目标
        other_precision = val_result['class_metrics']['Other']['precision']
        
        # 每5个周期寻找最优阈值
        if epoch % 5 == 0 or (epoch == args.epochs):
            logger.info("计算最优阈值...")
            best_threshold = find_best_threshold(val_result)
        
        # 保存模型标准：同时满足F1提高和other类精确率达到要求
        other_precision_reached = other_precision >= args.other_precision_target
        f1_improved = val_result['f1'] > best_metric
        
        if other_precision_reached:
            logger.info(f"Other类精确率达到目标: {other_precision:.4f} >= {args.other_precision_target:.4f}")
            if f1_improved:
                logger.info(f"F1提高 {best_metric:.4f} -> {val_result['f1']:.4f}，保存模型")
                best_metric = val_result['f1']
                best_other_precision = other_precision
                save_checkpoint(model, optimizer, scheduler, epoch, best_metric, checkpoint_path, other_precision)
        elif f1_improved and other_precision > best_other_precision:
            logger.info(f"F1提高 {best_metric:.4f} -> {val_result['f1']:.4f}, Other类精确率提高 {best_other_precision:.4f} -> {other_precision:.4f}，保存模型")
            best_metric = val_result['f1']
            best_other_precision = other_precision
            save_checkpoint(model, optimizer, scheduler, epoch, best_metric, checkpoint_path, other_precision)
        
        # 保存验证结果和混淆矩阵
        val_output_dir = output_dir / "val_results"
        val_output_dir.mkdir(exist_ok=True)
        
        # 保存类别指标
        with open(val_output_dir / f"val_class_metrics_epoch{epoch}.json", 'w') as f:
            json.dump(val_result['class_metrics'], f, indent=2)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(val_result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['FECD', 'Normal', 'Other'],
                   yticklabels=['FECD', 'Normal', 'Other'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'验证集混淆矩阵 (周期 {epoch})')
        plt.tight_layout()
        plt.savefig(val_output_dir / f"val_confusion_matrix_epoch{epoch}.png")
        plt.close()
    
    # 创建训练曲线图
    create_visualizations(train_metrics, val_metrics, output_dir)
    
    # 使用最终找到的最佳阈值保存到配置文件
    threshold_config = {
        'best_threshold': best_threshold,
        'other_precision_target': args.other_precision_target,
        'final_other_precision': best_other_precision,
        'final_f1': best_metric
    }
    
    with open(output_dir / "threshold_config.json", 'w') as f:
        json.dump(threshold_config, f, indent=2)
    
    logger.info(f"模型训练完成。最佳F1: {best_metric:.4f}, 最佳Other类精确率: {best_other_precision:.4f}, 最佳阈值: {best_threshold:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='眼科疾病多模态分类模型训练（改进版）')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp', help='数据根目录')
    parser.add_argument('--multimodal_index_path', type=str, default='/root/autodl-tmp/eye_data_index/multimodal_dataset_index.json', help='多模态数据集索引路径')
    parser.add_argument('--modalities', type=str, default='IVCM,OCT,前节照', help='使用的模态列表，以逗号分隔')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作线程数')
    parser.add_argument('--max_images_per_modality', type=int, default=5, help='每个模态最多使用的图像数量')
    parser.add_argument('--modality_completion', action='store_true', help='是否进行模态补全')
    
    # 模型相关参数
    parser.add_argument('--feature_dim', type=int, default=512, help='特征维度')
    parser.add_argument('--num_classes', type=int, default=3, help='类别数量')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'], help='特征提取骨干网络')
    parser.add_argument('--fusion_method', type=str, choices=['attention', 'concat', 'average'], default='attention', help='特征融合方法')
    parser.add_argument('--use_focal_loss', action='store_true', help='是否使用Focal Loss')
    parser.add_argument('--other_class_weight', type=float, default=2.0, help='Other类别的加权系数')
    parser.add_argument('--other_precision_target', type=float, default=0.9, help='Other类别的目标精确率')
    
    # 训练相关参数
    parser.add_argument('--lr', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减系数')
    parser.add_argument('--epochs', type=int, default=50, help='训练周期数')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    parser.add_argument('--use_weighted_sampler', action='store_true', help='是否使用加权采样器')
    parser.add_argument('--cosine_annealing_t0', type=int, default=10, help='余弦退火调度器的初始周期数')
    parser.add_argument('--cosine_annealing_tmult', type=int, default=2, help='余弦退火调度器的周期增加倍数')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/eye_multimodal_ensemble/improved_train_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 将逗号分隔的模态字符串转换为列表
    if isinstance(args.modalities, str):
        args.modalities = [m.strip() for m in args.modalities.split(',')]
    
    # 生成模态字符串用于文件名
    args.modalities_str = '_'.join(args.modalities)
    
    # 开始训练
    start_time = time.time()
    main(args)
    end_time = time.time()
    
    # 记录总训练时间
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒") 