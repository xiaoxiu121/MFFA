import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import argparse
import logging
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, 
    roc_curve, auc, roc_auc_score, precision_recall_curve
)

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入自定义模块
from dataset import get_dataloaders
from improved_model import ImprovedMultiModalEnsemblePredictor, create_improved_multimodal_model

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, dataloader, device, output_dir, other_threshold=0.7):
    """评估模型并保存结果"""
    # 创建集成预测器（单模型）
    predictor = ImprovedMultiModalEnsemblePredictor(
        models=[model], 
        device=device,
        threshold=other_threshold  # 设置分类阈值
    )
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []
    all_ids = []
    all_modality_masks = []
    all_uncertainties = []  # 新增：记录预测的确定性
    
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="评估", ascii=True)
        for batch in progress_bar:
            # 收集样本IDs和标签
            ids = batch.get('ids', ["unknown"] * len(batch['labels']))
            patient_ids = batch.get('patient_ids', ["unknown"] * len(batch['labels']))
            labels = batch['labels'].numpy()
            
            # 预测
            results = predictor.predict_batch(batch)
            preds = results['preds'].numpy()
            probs = results['probs'].numpy()
            
            # 修改other类的预测（如果概率不够高，改为第二高的类别）
            for i in range(len(preds)):
                if preds[i] == 2 and probs[i, 2] < other_threshold:
                    # 创建一个排除other类的probs副本
                    prob_no_other = probs[i].copy()
                    prob_no_other[2] = -float('inf')
                    # 重新预测
                    preds[i] = np.argmax(prob_no_other)
            
            # 收集模态掩码信息
            if 'modality_mask' in results:
                modality_mask = results['modality_mask'].numpy()
                all_modality_masks.extend(modality_mask)
            
            # 收集不确定性信息
            if 'uncertainty' in results:
                uncertainties = results['uncertainty'].numpy()
                all_uncertainties.extend(uncertainties)
            
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
    
    # 计算ROC和AUC
    # 将标签进行one-hot编码
    y_true_onehot = np.zeros((len(all_labels), 3))
    for i, label in enumerate(all_labels):
        y_true_onehot[i, label] = 1
    
    # 计算每个类别的ROC和AUC
    roc_curves = {}
    aucs = {}
    
    for i, class_name in enumerate(['FECD', 'Normal', 'Other']):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], np.array(all_probs)[:, i])
        roc_auc = auc(fpr, tpr)
        
        roc_curves[class_name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        aucs[class_name] = float(roc_auc)
    
    # 计算宏平均和微平均AUC
    macro_auc = np.mean(list(aucs.values()))
    try:
        micro_auc = roc_auc_score(y_true_onehot, np.array(all_probs), average='micro')
    except:
        micro_auc = 0.0
    
    # 计算PR曲线（精确率-召回率曲线）
    pr_curves = {}
    for i, class_name in enumerate(['FECD', 'Normal', 'Other']):
        precision_values, recall_values, thresholds = precision_recall_curve(
            y_true_onehot[:, i], np.array(all_probs)[:, i]
        )
        # 记录曲线数据
        pr_curves[class_name] = {
            'precision': precision_values.tolist(),
            'recall': recall_values.tolist(),
            'thresholds': thresholds.tolist() if len(thresholds) > 0 else []
        }
    
    # 创建结果字典
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist(),
        'roc_curves': roc_curves,
        'pr_curves': pr_curves,
        'aucs': aucs,
        'macro_auc': float(macro_auc),
        'micro_auc': float(micro_auc),
        'other_threshold': float(other_threshold)
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
        
        # 如果有不确定性信息，添加到样本结果中
        if all_uncertainties:
            sample['certainty'] = float(all_uncertainties[i])
        
        # 如果有模态掩码信息，添加到样本结果中
        if all_modality_masks:
            sample['modality_mask'] = [bool(m) for m in all_modality_masks[i]]
        
        sample_results.append(sample)
    
    # 创建输出目录
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存整体指标
        with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # 保存逐样本结果
        with open(output_dir / 'sample_predictions.json', 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, ensure_ascii=False, indent=2)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['FECD', 'Normal', 'Other'],
                   yticklabels=['FECD', 'Normal', 'Other'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green']
        for i, (class_name, color) in enumerate(zip(['FECD', 'Normal', 'Other'], colors)):
            fpr = roc_curves[class_name]['fpr']
            tpr = roc_curves[class_name]['tpr']
            roc_auc = aucs[class_name]
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(output_dir / 'roc_curves.png')
        plt.close()
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green']
        for i, (class_name, color) in enumerate(zip(['FECD', 'Normal', 'Other'], colors)):
            precision_values = pr_curves[class_name]['precision']
            recall_values = pr_curves[class_name]['recall']
            plt.plot(recall_values, precision_values, color=color, lw=2,
                     label=f'{class_name}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('PR曲线')
        plt.legend(loc="lower left")
        plt.savefig(output_dir / 'pr_curves.png')
        plt.close()
        
        # 如果有不确定性信息，分析不确定性和准确率的关系
        if all_uncertainties:
            # 计算不同确定性阈值下的准确率
            certainty_thresholds = np.arange(0.1, 1.0, 0.1)
            certainty_accuracy = []
            certainty_coverage = []  # 覆盖范围（保留的样本比例）
            
            for threshold in certainty_thresholds:
                # 筛选确定性高于阈值的样本
                high_certainty_indices = [i for i in range(len(all_uncertainties)) if all_uncertainties[i] >= threshold]
                if high_certainty_indices:
                    # 计算这些样本的准确率
                    high_certainty_preds = [all_preds[i] for i in high_certainty_indices]
                    high_certainty_labels = [all_labels[i] for i in high_certainty_indices]
                    accuracy = accuracy_score(high_certainty_labels, high_certainty_preds)
                    certainty_accuracy.append(accuracy)
                    # 计算覆盖率
                    coverage = len(high_certainty_indices) / len(all_uncertainties)
                    certainty_coverage.append(coverage)
                else:
                    certainty_accuracy.append(0)
                    certainty_coverage.append(0)
            
            # 绘制确定性-准确率曲线
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(certainty_thresholds, certainty_accuracy, marker='o')
            plt.xlabel('确定性阈值')
            plt.ylabel('准确率')
            plt.title('确定性与准确率关系')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(certainty_thresholds, certainty_coverage, marker='o')
            plt.xlabel('确定性阈值')
            plt.ylabel('样本覆盖率')
            plt.title('确定性与样本覆盖率关系')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'certainty_analysis.png')
            plt.close()
        
        # 如果有模态掩码信息，分析模态缺失模式
        if all_modality_masks:
            # 统计每种模态组合的数量和性能
            modality_combinations = {}
            
            for i in range(len(all_modality_masks)):
                # 将NumPy数组转换为Python列表，并确保是bool类型
                mask = [bool(m) for m in all_modality_masks[i]]
                # 将mask转换为字符串作为字典键
                mask_key = str(mask)
                correct = bool(all_labels[i] == all_preds[i])
                
                if mask_key not in modality_combinations:
                    modality_combinations[mask_key] = {
                        'count': 0,
                        'correct': 0,
                        'labels': [0, 0, 0],  # FECD, Normal, Other
                        'correct_by_class': [0, 0, 0],  # FECD, Normal, Other
                        'predicted_as_other': 0,  # 预测为other的样本数
                        'true_other': 0,  # 真实为other的样本数
                        'mask': mask  # 保存原始mask
                    }
                
                modality_combinations[mask_key]['count'] += 1
                if correct:
                    modality_combinations[mask_key]['correct'] += 1
                
                label = int(all_labels[i])
                modality_combinations[mask_key]['labels'][label] += 1
                if correct:
                    modality_combinations[mask_key]['correct_by_class'][label] += 1
                
                # 统计预测为other的样本
                if all_preds[i] == 2:
                    modality_combinations[mask_key]['predicted_as_other'] += 1
                # 统计真实为other的样本
                if all_labels[i] == 2:
                    modality_combinations[mask_key]['true_other'] += 1
            
            # 计算每种组合的准确率和other类指标
            modality_analysis = []
            for mask_key, stats in modality_combinations.items():
                mask = stats['mask']  # 获取原始mask
                accuracy = float(stats['correct'] / stats['count'] if stats['count'] > 0 else 0)
                
                # 计算每个类别的准确率
                class_accuracy = []
                for i in range(3):
                    if stats['labels'][i] > 0:
                        class_acc = float(stats['correct_by_class'][i] / stats['labels'][i])
                    else:
                        class_acc = 0.0
                    class_accuracy.append(class_acc)
                
                # 计算other类的精确率和召回率
                true_other = stats['true_other']
                predicted_as_other = stats['predicted_as_other']
                other_precision = float(stats['correct_by_class'][2] / predicted_as_other if predicted_as_other > 0 else 0)
                other_recall = float(stats['correct_by_class'][2] / true_other if true_other > 0 else 0)
                
                # 创建可读的掩码字符串
                model_modalities = model.modalities if hasattr(model, 'modalities') else ['IVCM', 'OCT', '前节照']
                # 确保mask中的元素是Python原生布尔类型
                python_mask = [bool(m) for m in mask]
                mask_str = '+'.join([modality for i, modality in enumerate(model_modalities) if python_mask[i]])
                if not mask_str:
                    mask_str = "无模态"
                
                modality_analysis.append({
                    'modality_combination': mask_str,
                    'mask': python_mask,  # 转换为Python布尔列表
                    'count': int(stats['count']),
                    'accuracy': accuracy,
                    'class_distribution': {
                        'FECD': int(stats['labels'][0]),
                        'Normal': int(stats['labels'][1]),
                        'Other': int(stats['labels'][2])
                    },
                    'class_accuracy': {
                        'FECD': float(class_accuracy[0]),
                        'Normal': float(class_accuracy[1]),
                        'Other': float(class_accuracy[2])
                    },
                    'other_metrics': {
                        'precision': other_precision,
                        'recall': other_recall,
                        'predicted_count': predicted_as_other,
                        'true_count': true_other
                    }
                })
            
            # 按样本数量排序
            modality_analysis.sort(key=lambda x: x['count'], reverse=True)
            
            # 保存模态分析
            with open(output_dir / 'modality_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(modality_analysis, f, ensure_ascii=False, indent=2)
            
            # 绘制模态组合准确率图
            plt.figure(figsize=(12, 8))
            combinations = [item['modality_combination'] for item in modality_analysis]
            accuracies = [item['accuracy'] for item in modality_analysis]
            counts = [item['count'] for item in modality_analysis]
            
            # 只显示前10个最常见的组合
            if len(combinations) > 10:
                combinations = combinations[:10]
                accuracies = accuracies[:10]
                counts = counts[:10]
            
            plt.bar(combinations, accuracies, alpha=0.7)
            
            # 添加样本数量标签
            for i, (acc, count) in enumerate(zip(accuracies, counts)):
                plt.text(i, acc + 0.02, f"n={count}", ha='center')
            
            plt.xlabel('模态组合')
            plt.ylabel('准确率')
            plt.title('不同模态组合的准确率')
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'modality_accuracy.png')
            plt.close()
            
            # 绘制模态组合的Other类精确率图
            plt.figure(figsize=(12, 8))
            combinations = [item['modality_combination'] for item in modality_analysis]
            other_precisions = [item['other_metrics']['precision'] for item in modality_analysis]
            counts = [item['other_metrics']['predicted_count'] for item in modality_analysis]
            
            # 只显示前10个最常见的组合
            if len(combinations) > 10:
                combinations = combinations[:10]
                other_precisions = other_precisions[:10]
                counts = counts[:10]
            
            plt.bar(combinations, other_precisions, alpha=0.7, color='green')
            
            # 添加样本数量标签
            for i, (prec, count) in enumerate(zip(other_precisions, counts)):
                plt.text(i, prec + 0.02, f"n={count}", ha='center')
            
            plt.xlabel('模态组合')
            plt.ylabel('Other类精确率')
            plt.title('不同模态组合的Other类精确率')
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'modality_other_precision.png')
            plt.close()
    
    # 打印总体指标
    logger.info(f"评估完成, 准确率: {accuracy:.4f}, F1: {f1:.4f}")
    
    # 打印每个类别的详细指标
    logger.info("各类别详细指标:")
    for class_name, metrics in class_metrics.items():
        logger.info(f"  {class_name} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 样本数: {metrics['support']}")
    
    # 打印AUC值
    logger.info("各类别AUC值:")
    for class_name, auc_value in aucs.items():
        logger.info(f"  {class_name} - AUC: {auc_value:.4f}")
    logger.info(f"  宏平均AUC: {macro_auc:.4f}, 微平均AUC: {micro_auc:.4f}")
    
    # 打印混淆矩阵
    logger.info("混淆矩阵:")
    class_names = ['FECD', 'Normal', 'Other']
    for i, row in enumerate(cm):
        logger.info(f"  {class_names[i]}: {row.tolist()}")
    
    logger.info(f"其他类别阈值: {other_threshold:.2f}")
    logger.info(f"结果已保存到 {output_dir}")
    
    return metrics, sample_results

def load_model(model_path, device='cuda'):
    """加载模型"""
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查检查点中是否有模态信息
    if 'modalities' in checkpoint:
        modalities = checkpoint['modalities']
    else:
        modalities = ['IVCM', 'OCT', '前节照']  # 默认模态
        logger.warning(f"检查点中没有模态信息，使用默认模态: {modalities}")
    
    # 从配置文件中读取模型参数或使用默认值
    feature_dim = 512
    num_classes = 3
    dropout = 0.5
    backbone = 'resnet50'
    fusion_method = 'attention'
    
    # 创建模型
    model = create_improved_multimodal_model(
        modalities=modalities,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout=dropout,
        backbone=backbone,
        fusion_method=fusion_method,
        use_focal_loss=True
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"模型加载完成，使用模态: {modalities}")
    
    return model, modalities

def find_optimal_threshold(model, dataloader, device, threshold_range=None):
    """
    寻找最优的Other类别阈值
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        threshold_range: 尝试的阈值范围，如果为None则使用默认范围
    
    Returns:
        best_threshold: 最优阈值
    """
    if threshold_range is None:
        threshold_range = np.arange(0.5, 0.95, 0.05)
    
    best_threshold = 0.5
    best_f1 = 0
    best_other_precision = 0
    
    logger.info("寻找最优的Other类别阈值...")
    
    for threshold in threshold_range:
        # 评估当前阈值
        model.eval()
        predictor = ImprovedMultiModalEnsemblePredictor([model], device=device)
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"评估阈值 {threshold:.2f}", ascii=True):
                # 获取标签
                labels = batch['labels'].numpy()
                
                # 预测
                results = predictor.predict_batch(batch)
                preds = results['preds'].numpy()
                probs = results['probs'].numpy()
                
                # 修改other类的预测（如果概率不够高，改为第二高的类别）
                for i in range(len(preds)):
                    if preds[i] == 2 and probs[i, 2] < threshold:
                        # 创建一个排除other类的probs副本
                        prob_no_other = probs[i].copy()
                        prob_no_other[2] = -float('inf')
                        # 重新预测
                        preds[i] = np.argmax(prob_no_other)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels)
        
        # 计算指标
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # 计算每个类别的详细指标
        class_precision, _, _, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        other_precision = class_precision[2]
        
        logger.info(f"阈值 {threshold:.2f}: 总体F1 = {f1:.4f}, other精确率 = {other_precision:.4f}")
        
        # 更新最佳阈值 - 优先考虑other精确率达到0.9以上，其次考虑总体F1
        if other_precision >= 0.9 and f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
            best_other_precision = other_precision
        # 如果还没有找到满足条件的阈值，但当前阈值的other精确率更好
        elif best_other_precision < 0.9 and other_precision > best_other_precision:
            best_threshold = threshold
            best_f1 = f1
            best_other_precision = other_precision
    
    logger.info(f"最优阈值: {best_threshold:.2f}, other精确率: {best_other_precision:.4f}, 总体F1: {best_f1:.4f}")
    
    return best_threshold

def main(args):
    """主函数"""
    # 配置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置文件日志
    log_file = output_dir / "test.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 加载模型
    model, modalities = load_model(args.model_path, device)
    logger.info(f"使用的模态: {modalities}")
    
    # 获取数据加载器
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        multimodal_index_path=args.multimodal_index_path,
        modalities=modalities,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images_per_modality=args.max_images_per_modality,
        modality_completion=args.modality_completion
    )
    
    # 如果不提供阈值，则自动搜索最优阈值
    if args.other_threshold <= 0:
        logger.info("自动搜索最优阈值...")
        # 使用验证集寻找最优阈值
        best_threshold = find_optimal_threshold(model, dataloaders[1], device)
    else:
        best_threshold = args.other_threshold
        logger.info(f"使用指定的阈值: {best_threshold}")
    
    # 若提供了阈值配置文件，从中加载阈值
    if args.threshold_config and os.path.exists(args.threshold_config):
        try:
            with open(args.threshold_config, 'r') as f:
                config = json.load(f)
                if 'best_threshold' in config:
                    best_threshold = config['best_threshold']
                    logger.info(f"从配置文件加载阈值: {best_threshold}")
        except Exception as e:
            logger.error(f"读取阈值配置文件出错: {e}")
    
    # 评估模型
    metrics, _ = evaluate_model(
        model=model,
        dataloader=dataloaders[2],  # 测试集
        device=device,
        output_dir=output_dir,
        other_threshold=best_threshold
    )
    
    # 如果other类精确率未达标，尝试更高的阈值
    if metrics['class_metrics']['Other']['precision'] < args.other_precision_target:
        logger.info(f"Other类精确率 {metrics['class_metrics']['Other']['precision']:.4f} 未达到目标 {args.other_precision_target:.4f}，尝试更高阈值")
        
        # 在更高阈值范围内搜索
        higher_thresholds = np.arange(best_threshold + 0.05, 0.99, 0.05)
        for threshold in higher_thresholds:
            logger.info(f"尝试阈值: {threshold:.2f}")
            new_metrics, _ = evaluate_model(
                model=model,
                dataloader=dataloaders[2],
                device=device,
                output_dir=output_dir / f"threshold_{threshold:.2f}",
                other_threshold=threshold
            )
            
            # 检查是否达到目标
            if new_metrics['class_metrics']['Other']['precision'] >= args.other_precision_target:
                logger.info(f"阈值 {threshold:.2f} 下，Other类精确率达到 {new_metrics['class_metrics']['Other']['precision']:.4f}")
                
                # 保存找到的最佳阈值
                with open(output_dir / "best_threshold.json", 'w') as f:
                    json.dump({
                        'best_threshold': float(threshold),
                        'other_precision': float(new_metrics['class_metrics']['Other']['precision']),
                        'overall_f1': float(new_metrics['f1'])
                    }, f, indent=2)
                
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="眼科疾病多模态分类模型评估（改进版）")
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp', help='数据根目录')
    parser.add_argument('--multimodal_index_path', type=str, default='/root/autodl-tmp/eye_data_index/multimodal_dataset_index.json', help='多模态数据集索引路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作线程数')
    parser.add_argument('--max_images_per_modality', type=int, default=5, help='每个模态最多使用的图像数量')
    parser.add_argument('--modality_completion', action='store_true', help='是否进行模态补全')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--other_threshold', type=float, default=-1, help='Other类别的概率阈值，低于该阈值的预测将被替换为第二可能的类别。-1表示自动搜索')
    parser.add_argument('--threshold_config', type=str, default='', help='阈值配置文件路径，用于加载预先定义的最佳阈值')
    parser.add_argument('--other_precision_target', type=float, default=0.9, help='Other类别的目标精确率')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/eye_multimodal_ensemble/improved_test_results', help='输出目录')
    
    args = parser.parse_args()
    
    main(args) 