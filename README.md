# MFFA

## 环境要求

- Python 3.7+
- PyTorch 1.10+
- CUDA 11.0+ (推荐)
- scikit-learn, pandas, matplotlib, seaborn 等库

## 目录结构

```
mian/
├── dataset.py         # 数据集和数据加载器
├── model.py           # model definition
├── train.py           # train
├── evaluate.py        # evaluate
├── run.sh             # 运行脚本
├── logs/              # 日志文件
├── train_results/     # 训练结果
└── test_results/      # 测试结果
```

## 使用方法

### 训练模型

Train the model

```bash
./run.sh --mode train --modalities "IVCM,OCT,前节照" --fusion_method attention
```

### 评估模型

使用以下命令评估训练好的模型：

```bash
./run.sh --mode evaluate --model_path "/path/to/model.pt" --modalities "IVCM,OCT,前节照"
```

### 参数说明

运行脚本支持以下参数：

- `--mode`：运行模式，可选 train（训练）、evaluate（评估）、finetune（微调）
- `--modalities`：使用的模态列表，用逗号分隔，例如 "IVCM,OCT,前节照"
- `--batch_size`：批次大小，默认为8
- `--max_images`：每个模态最多使用的图像数量，默认为5
- `--feature_dim`：特征维度，默认为512
- `--fusion_method`：融合方法，可选 attention、concat、average，默认为attention
- `--epochs`：训练周期数，默认为50
- `--lr`：学习率，默认为0.0001
- `--weight_decay`：权重衰减，默认为0.01
- `--no_mixed_precision`：不使用混合精度训练（默认使用）
- `--no_modality_completion`：不使用模态补全（默认使用）
- `--model_path`：模型路径，用于评估或微调
- `--device`：使用的设备，可选 cuda、cpu，默认为cuda


