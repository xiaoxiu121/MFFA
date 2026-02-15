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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, roc_auc_score

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入自定义模块
from dataset import get_dataloaders
from model import MultiModalEnsemblePredictor, create_multimodal_model

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, dataloader, device, output_dir):
    """评估模型并保存结果"""
    predictor = MultiModalEnsemblePredictor(model, method='mean', device=device)
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []
    all_ids = []
    all_modality_masks = []
    
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
            
            # 收集模态掩码信息
            if 'modality_mask' in results:
                modality_mask = results['modality_mask'].numpy()
                all_modality_masks.extend(modality_mask)
            
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
    
    # 创建结果字典
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'class_metrics': class_metrics,
        'confusion_matrix': cm.tolist(),
        'roc_curves': roc_curves,
        'aucs': aucs,
        'macro_auc': float(macro_auc),
        'micro_auc': float(micro_auc)
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
                        'mask': mask  # 保存原始mask
                    }
                
                modality_combinations[mask_key]['count'] += 1
                if correct:
                    modality_combinations[mask_key]['correct'] += 1
                
                label = int(all_labels[i])
                modality_combinations[mask_key]['labels'][label] += 1
                if correct:
                    modality_combinations[mask_key]['correct_by_class'][label] += 1
            
            # 计算每种组合的准确率
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
    
    # 如果有模态分析信息，打印模态组合性能
    if all_modality_masks and len(modality_analysis) > 0:
        logger.info("模态组合性能 (前5个):")
        for i, combo in enumerate(modality_analysis[:5]):
            logger.info(f"  {combo['modality_combination']} - 准确率: {combo['accuracy']:.4f}, 样本数: {combo['count']}")
    
    logger.info(f"结果已保存到 {output_dir}")
    
    return metrics, sample_results

def load_model(model_path, modalities=None, feature_dim=512, num_classes=3, dropout=0.3, fusion_method='attention', device='cuda'):
    """加载模型"""
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查检查点中是否有模态信息
    if 'modalities' in checkpoint and modalities is None:
        modalities = checkpoint['modalities']
    
    # 创建模型
    model = create_multimodal_model(
        modalities=modalities,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout=dropout,
        fusion_method=fusion_method
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, modalities

def main(args):
    """主函数"""
    # 配置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 解析模态列表
    modalities = args.modalities.split(',') if args.modalities else None
    
    # 加载模型
    model, modalities = load_model(
        model_path=args.model_path, 
        modalities=modalities,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        fusion_method=args.fusion_method,
        device=device
    )
    
    logger.info(f"使用的模态: {modalities}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置文件日志
    log_file = output_dir / f"{'-'.join(modalities)}_test.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
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
    
    # 评估模型
    evaluate_model(
        model=model,
        dataloader=dataloaders[2],
        device=device,
        output_dir=output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="眼科疾病多模态分类模型评估")
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp', help='数据根目录')
    parser.add_argument('--multimodal_index_path', type=str, default='/root/autodl-tmp/eye_data_index/multimodal_dataset_index.json', help='多模态数据集索引路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作线程数')
    parser.add_argument('--max_images_per_modality', type=int, default=5, help='每个模态最多使用的图像数量')
    parser.add_argument('--modality_completion', action='store_true', help='是否进行模态补全')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--modalities', type=str, default=None, help='使用的模态列表，用逗号分隔')
    parser.add_argument('--feature_dim', type=int, default=512, help='特征维度')
    parser.add_argument('--num_classes', type=int, default=3, help='类别数量')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--fusion_method', type=str, choices=['attention', 'concat', 'average'], default='attention', help='特征融合方法')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/eye_multimodal_ensemble/test_results', help='输出目录')
    
    args = parser.parse_args()
    
    main(args)