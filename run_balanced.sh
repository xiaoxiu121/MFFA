#!/bin/bash

# 设置默认值
MODE="train"
MODALITIES="IVCM,OCT,前节照"
BATCH_SIZE=8
NUM_WORKERS=4
MAX_IMAGES=5
FEATURE_DIM=512
FUSION_METHOD="attention"
EPOCHS=50
LR=0.0001
WEIGHT_DECAY=0.01
MIXED_PRECISION=true
MODALITY_COMPLETION=true
MODEL_PATH=""
DEVICE="cuda"
SEED=42

# 帮助信息
print_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help                  显示帮助信息"
    echo "  -m, --mode MODE             运行模式: train, evaluate, finetune (默认: train)"
    echo "  --modalities MODALITIES     要使用的模态列表，用逗号分隔 (默认: IVCM,OCT,前节照)"
    echo "  --batch_size SIZE           批次大小 (默认: 8)"
    echo "  --num_workers NUM           数据加载工作线程数 (默认: 4)"
    echo "  --max_images NUM            每个模态最多使用的图像数量 (默认: 5)"
    echo "  --feature_dim DIM           特征维度 (默认: 512)"
    echo "  --fusion_method METHOD      融合方法: attention, concat, average (默认: attention)"
    echo "  --epochs NUM                训练周期数 (默认: 50)"
    echo "  --lr RATE                   学习率 (默认: 0.0001)"
    echo "  --weight_decay RATE         权重衰减 (默认: 0.01)"
    echo "  --no_mixed_precision        不使用混合精度训练"
    echo "  --no_modality_completion    不使用模态补全"
    echo "  --model_path PATH           模型路径，用于评估或微调"
    echo "  --device DEVICE             设备: cuda, cpu (默认: cuda)"
    echo "  --seed SEED                 随机种子 (默认: 42)"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            print_help
            exit 0
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --modalities)
            MODALITIES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --max_images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --feature_dim)
            FEATURE_DIM="$2"
            shift 2
            ;;
        --fusion_method)
            FUSION_METHOD="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --no_mixed_precision)
            MIXED_PRECISION=false
            shift
            ;;
        --no_modality_completion)
            MODALITY_COMPLETION=false
            shift
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "未知选项: $1"
            print_help
            exit 1
            ;;
    esac
done

# 日志目录
LOG_DIR="/root/autodl-tmp/eye_multimodal_ensemble/logs_balanced"
mkdir -p $LOG_DIR

# 模态字符串，用于文件名
MODALITIES_STR=$(echo $MODALITIES | tr ',' '_')

# 当前日期时间
DATETIME=$(date +"%Y%m%d_%H%M%S")

# 处理模式
case $MODE in
    train)
        echo "开始训练强化Other类的平衡模型..."
        
        # 设置混合精度和模态补全标志
        MIXED_PRECISION_FLAG=""
        if [ "$MIXED_PRECISION" = true ]; then
            MIXED_PRECISION_FLAG="--mixed_precision"
        fi
        
        MODALITY_COMPLETION_FLAG=""
        if [ "$MODALITY_COMPLETION" = true ]; then
            MODALITY_COMPLETION_FLAG="--modality_completion"
        fi
        
        # 训练命令
        TRAIN_CMD="python /root/autodl-tmp/eye_multimodal_ensemble/train_balanced.py \
            --data_dir /root/autodl-tmp \
            --multimodal_index_path /root/autodl-tmp/eye_data_index/multimodal_dataset_index.json \
            --modalities $MODALITIES \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --max_images_per_modality $MAX_IMAGES \
            --feature_dim $FEATURE_DIM \
            --fusion_method $FUSION_METHOD \
            --epochs $EPOCHS \
            --lr $LR \
            --weight_decay $WEIGHT_DECAY \
            --device $DEVICE \
            --seed $SEED \
            $MIXED_PRECISION_FLAG \
            $MODALITY_COMPLETION_FLAG \
            --output_dir /root/autodl-tmp/eye_multimodal_ensemble/balanced_results/${MODALITIES_STR}_${DATETIME}"
        
        echo "运行命令: $TRAIN_CMD"
        eval $TRAIN_CMD 2>&1 | tee $LOG_DIR/${MODALITIES_STR}_training_balanced_${DATETIME}.log
        ;;
        
    evaluate)
        if [ -z "$MODEL_PATH" ]; then
            echo "错误: 评估模式需要提供 --model_path"
            exit 1
        fi
        
        echo "开始评估模型: $MODEL_PATH"
        
        # 设置模态补全标志
        MODALITY_COMPLETION_FLAG=""
        if [ "$MODALITY_COMPLETION" = true ]; then
            MODALITY_COMPLETION_FLAG="--modality_completion"
        fi
        
        # 评估命令
        EVAL_CMD="python /root/autodl-tmp/eye_multimodal_ensemble/evaluate.py \
            --data_dir /root/autodl-tmp \
            --multimodal_index_path /root/autodl-tmp/eye_data_index/multimodal_dataset_index.json \
            --modalities $MODALITIES \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --max_images_per_modality $MAX_IMAGES \
            --feature_dim $FEATURE_DIM \
            --fusion_method $FUSION_METHOD \
            --device $DEVICE \
            $MODALITY_COMPLETION_FLAG \
            --model_path $MODEL_PATH \
            --output_dir /root/autodl-tmp/eye_multimodal_ensemble/balanced_test_results/${MODALITIES_STR}_${DATETIME}"
        
        echo "运行命令: $EVAL_CMD"
        eval $EVAL_CMD 2>&1 | tee $LOG_DIR/${MODALITIES_STR}_test_balanced_${DATETIME}.log
        ;;
        
    finetune)
        if [ -z "$MODEL_PATH" ]; then
            echo "错误: 微调模式需要提供 --model_path"
            exit 1
        fi
        
        echo "开始微调模型: $MODEL_PATH"
        
        # 设置混合精度和模态补全标志
        MIXED_PRECISION_FLAG=""
        if [ "$MIXED_PRECISION" = true ]; then
            MIXED_PRECISION_FLAG="--mixed_precision"
        fi
        
        MODALITY_COMPLETION_FLAG=""
        if [ "$MODALITY_COMPLETION" = true ]; then
            MODALITY_COMPLETION_FLAG="--modality_completion"
        fi
        
        # 微调命令
        FINETUNE_CMD="python /root/autodl-tmp/eye_multimodal_ensemble/train_balanced.py \
            --data_dir /root/autodl-tmp \
            --multimodal_index_path /root/autodl-tmp/eye_data_index/multimodal_dataset_index.json \
            --modalities $MODALITIES \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --max_images_per_modality $MAX_IMAGES \
            --feature_dim $FEATURE_DIM \
            --fusion_method $FUSION_METHOD \
            --epochs $EPOCHS \
            --lr $LR \
            --weight_decay $WEIGHT_DECAY \
            --device $DEVICE \
            --seed $SEED \
            $MIXED_PRECISION_FLAG \
            $MODALITY_COMPLETION_FLAG \
            --resume \
            --model_path $MODEL_PATH \
            --output_dir /root/autodl-tmp/eye_multimodal_ensemble/balanced_results/${MODALITIES_STR}_finetune_${DATETIME}"
        
        echo "运行命令: $FINETUNE_CMD"
        eval $FINETUNE_CMD 2>&1 | tee $LOG_DIR/${MODALITIES_STR}_finetune_balanced_${DATETIME}.log
        ;;
        
    *)
        echo "未知模式: $MODE"
        print_help
        exit 1
        ;;
esac

echo "完成!" 