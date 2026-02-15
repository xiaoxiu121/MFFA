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
    
    # 计算指标
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
            'precision': float(class_precision[0]),
            'recall': float(class_recall[0]),
            'f1': float(class_f1[0]),
            'support': int(class_support[0])
        },
        'Normal': {
            'precision': float(class_precision[1]),
            'recall': float(class_recall[1]),
            'f1': float(class_f1[1]),
            'support': int(class_support[1])
        },
        'Other': {
            'precision': float(class_precision[2]),
            'recall': float(class_recall[2]),
            'f1': float(class_f1[2]),
            'support': int(class_support[2])
        }
    }
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    # 创建结果字典
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    # 逐样本结果
    sample_results = []
    class_mapping = {0: 'FECD', 1: 'normal', 2: 'other'}
    
    for i in range(len(all_ids)):
        sample = {
            'id': all_ids[i],
            'patient_id': all_patient_ids[i],
            'true_label': int(all_labels[i]),
            'true_class': class_mapping[int(all_labels[i])],
            'pred_label': int(all_preds[i]),
            'pred_class': class_mapping[int(all_preds[i])],
            'correct': bool(all_labels[i] == all_preds[i]),
            'probabilities': {
                'FECD': float(all_probs[i][0]),
                'normal': float(all_probs[i][1]),
                'other': float(all_probs[i][2])
            }
        }
        sample_results.append(sample)
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存整体指标
        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存逐样本结果
        with open(output_dir / 'sample_predictions.json', 'w') as f:
            json.dump(sample_results, f, indent=2)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['FECD', 'Normal', 'Other'],
                   yticklabels=['FECD', 'Normal', 'Other'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('测试集混淆矩阵')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
    
    return results, sample_results

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric
    }
    torch.save(checkpoint, save_path)
    logger.info(f"检查点已保存到 {save_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        logger.info(f"检查点 {checkpoint_path} 不存在，从头开始训练")
        return 0, 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    
    logger.info(f"从检查点 {checkpoint_path} 加载，上次最佳指标: {best_metric:.4f}, 当前周期: {epoch}")
    return epoch, best_metric

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
    
    # 获取数据加载器
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        multimodal_index_path=args.multimodal_index_path,
        modalities=args.modalities,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images_per_modality=args.max_images_per_modality,
        modality_completion=args.modality_completion
    )
    
    # 创建模型
    model = create_multimodal_model(
        modalities=args.modalities,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        fusion_method=args.fusion_method
    )
    model.to(device)
    
    # 打印模型信息
    logger.info(f"模型结构:\n{model}")
    
    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # 记录训练和验证指标
    train_metrics = []
    val_metrics = []
    
    # 加载检查点（如果存在）
    checkpoint_path = output_dir / f"{args.modalities_str}_best_model.pt"
    start_epoch, best_f1 = load_checkpoint(model, optimizer, scheduler, checkpoint_path) if args.resume else (0, 0)

    # 训练循环
    logger.info("开始训练...")
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        logger.info(f"周期 {epoch}/{args.epochs}")
        
        # 训练一个周期
        logger.info("训练阶段")
        train_result = train_epoch(model, dataloaders[0], criterion, optimizer, device, scaler)
        train_metrics.append(train_result)
        
        # 验证
        logger.info("验证阶段")
        val_result = validate(model, dataloaders[1], criterion, device)
        val_metrics.append(val_result)
        
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
        
        # 更新学习率调度器
        scheduler.step(val_result['f1'])
        
        # 如果当前模型是最佳模型，保存检查点
        if val_result['f1'] > best_f1:
            logger.info(f"F1提高 {best_f1:.4f} -> {val_result['f1']:.4f}，保存模型")
            best_f1 = val_result['f1']
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, checkpoint_path)

            # 保存验证结果
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
    
    # 在测试集上评估最佳模型
    logger.info("在测试集上评估最佳模型")
    
    # 加载最佳模型
    best_model = create_multimodal_model(
        modalities=args.modalities,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        fusion_method=args.fusion_method
    )
    
    checkpoint = torch.load(checkpoint_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)
    
    # 测试
    test_output_dir = output_dir / "test_results"
    test_metrics, sample_predictions = test(best_model, dataloaders[2], device, test_output_dir)
    
    # 打印测试结果
    logger.info(f"测试准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"测试 F1: {test_metrics['f1']:.4f}")
    logger.info(f"测试精确率: {test_metrics['precision']:.4f}")
    logger.info(f"测试召回率: {test_metrics['recall']:.4f}")
    
    # 打印每个类别的指标
    logger.info("测试集各类别指标:")
    for class_name, metrics in test_metrics['class_metrics'].items():
        logger.info(f"  {class_name} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 样本数: {metrics['support']}")
    
    logger.info(f"测试结果已保存到 {test_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='眼科疾病多模态分类模型训练')
    
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
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--fusion_method', type=str, choices=['attention', 'concat', 'average'], default='attention', help='特征融合方法')
    
    # 训练相关参数
    parser.add_argument('--lr', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减系数')
    parser.add_argument('--epochs', type=int, default=50, help='训练周期数')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/eye_multimodal_ensemble/train_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 将逗号分隔的模态字符串转换为列表
    if isinstance(args.modalities, str):
        args.modalities = [m.strip() for m in args.modalities.split(',')]
    
    # 生成模态字符串用于文件名
    args.modalities_str = '_'.join(args.modalities)
    
    main(args)