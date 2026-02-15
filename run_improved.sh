#!/bin/bash

# 设置默认值
MODE="train"
MODALITIES="IVCM,OCT,前节照"
BATCH_SIZE=8
NUM_WORKERS=4
MAX_IMAGES=5
FEATURE_DIM=512
BACKBONE="resnet50"
FUSION_METHOD="attention"
EPOCHS=50
LR=0.0001
MIN_LR=0.000001
WEIGHT_DECAY=0.01
OTHER_CLASS_WEIGHT=2.0
OTHER_PRECISION_TARGET=0.9
MIXED_PRECISION=true
MODALITY_COMPLETION=true
USE_FOCAL_LOSS=true
USE_WEIGHTED_SAMPLER=true
MODEL_PATH=""
DEVICE="cuda"
OUTPUT_DIR="/root/autodl-tmp/eye_multimodal_ensemble/improved_results"
COSINE_ANNEALING_T0=10
COSINE_ANNEALING_TMULT=2
OTHER_THRESHOLD=-1
THRESHOLD_CONFIG=""

# 帮助信息
print_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help                  显示帮助信息"
    echo "  -m, --mode MODE             运行模式: train, evaluate (默认: train)"
    echo "  --modalities MODALITIES     要使用的模态列表，用逗号分隔 (默认: IVCM,OCT,前节照)"
    echo "  --batch_size SIZE           批次大小 (默认: 8)"
    echo "  --num_workers NUM           数据加载工作线程数 (默认: 4)"
    echo "  --max_images NUM            每个模态最多使用的图像数量 (默认: 5)"
    echo "  --feature_dim DIM           特征维度 (默认: 512)"
    echo "  --backbone BACKBONE         特征提取骨干网络: resnet18, resnet34, resnet50, efficientnet_b0 (默认: resnet50)"
    echo "  --fusion_method METHOD      融合方法: attention, concat, average (默认: attention)"
    echo "  --epochs NUM                训练周期数 (默认: 50)"
    echo "  --lr RATE                   学习率 (默认: 0.0001)"
    echo "  --min_lr RATE               最小学习率 (默认: 0.000001)"
    echo "  --weight_decay RATE         权重衰减 (默认: 0.01)"
    echo "  --other_class_weight WEIGHT Other类别的加权系数 (默认: 2.0)"
    echo "  --other_precision_target PREC Other类别的目标精确率 (默认: 0.9)"
    echo "  --other_threshold THRESH    Other类别的阈值，用于评估模式，-1表示自动搜索 (默认: -1)"
    echo "  --threshold_config PATH     阈值配置文件路径，用于加载预先定义的阈值"
    echo "  --no_mixed_precision        不使用混合精度训练"
    echo "  --no_modality_completion    不使用模态补全"
    echo "  --no_focal_loss             不使用Focal Loss"
    echo "  --no_weighted_sampler       不使用加权采样器"
    echo "  --resume                    从检查点恢复训练"
    echo "  --model_path PATH           模型路径，用于评估或恢复训练"
    echo "  --device DEVICE             设备: cuda, cpu (默认: cuda)"
    echo "  --output_dir DIR            输出目录 (默认: /root/autodl-tmp/eye_multimodal_ensemble/improved_results)"
    echo "  --cosine_annealing_t0 T0    余弦退火调度器的初始周期数 (默认: 10)"
    echo "  --cosine_annealing_tmult TM 余弦退火调度器的周期增加倍数 (默认: 2)"
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
        --backbone)
            BACKBONE="$2"
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
        --min_lr)
            MIN_LR="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --other_class_weight)
            OTHER_CLASS_WEIGHT="$2"
            shift 2
            ;;
        --other_precision_target)
            OTHER_PRECISION_TARGET="$2"
            shift 2
            ;;
        --other_threshold)
            OTHER_THRESHOLD="$2"
            shift 2
            ;;
        --threshold_config)
            THRESHOLD_CONFIG="$2"
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
        --no_focal_loss)
            USE_FOCAL_LOSS=false
            shift
            ;;
        --no_weighted_sampler)
            USE_WEIGHTED_SAMPLER=false
            shift
            ;;
        --resume)
            RESUME=true
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cosine_annealing_t0)
            COSINE_ANNEALING_T0="$2"
            shift 2
            ;;
        --cosine_annealing_tmult)
            COSINE_ANNEALING_TMULT="$2"
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
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p $LOG_DIR

# 模态字符串，用于文件名
MODALITIES_STR=$(echo $MODALITIES | tr ',' '_')

# 当前日期时间
DATETIME=$(date +"%Y%m%d_%H%M%S")

# 格式化输出目录
FORMATTED_OUTPUT_DIR="${OUTPUT_DIR}/${MODALITIES_STR}_${BACKBONE}_${FUSION_METHOD}_${DATETIME}"

# 处理模式
case $MODE in
    train)
        echo "开始训练改进的多模态模型..."
        
        # 设置参数标志
        MIXED_PRECISION_FLAG=""
        if [ "$MIXED_PRECISION" = true ]; then
            MIXED_PRECISION_FLAG="--mixed_precision"
        fi
        
        MODALITY_COMPLETION_FLAG=""
        if [ "$MODALITY_COMPLETION" = true ]; then
            MODALITY_COMPLETION_FLAG="--modality_completion"
        fi
        
        FOCAL_LOSS_FLAG=""
        if [ "$USE_FOCAL_LOSS" = true ]; then
            FOCAL_LOSS_FLAG="--use_focal_loss"
        fi
        
        WEIGHTED_SAMPLER_FLAG=""
        if [ "$USE_WEIGHTED_SAMPLER" = true ]; then
            WEIGHTED_SAMPLER_FLAG="--use_weighted_sampler"
        fi
        
        RESUME_FLAG=""
        if [ "$RESUME" = true ]; then
            RESUME_FLAG="--resume"
        fi
        
        MODEL_PATH_FLAG=""
        if [ ! -z "$MODEL_PATH" ]; then
            MODEL_PATH_FLAG="--model_path $MODEL_PATH"
        fi
        
        # 训练命令
        TRAIN_CMD="python /root/autodl-tmp/eye_multimodal_ensemble/improved_train.py \
            --data_dir /root/autodl-tmp \
            --multimodal_index_path /root/autodl-tmp/eye_data_index/multimodal_dataset_index.json \
            --modalities $MODALITIES \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --max_images_per_modality $MAX_IMAGES \
            --feature_dim $FEATURE_DIM \
            --backbone $BACKBONE \
            --fusion_method $FUSION_METHOD \
            --epochs $EPOCHS \
            --lr $LR \
            --min_lr $MIN_LR \
            --weight_decay $WEIGHT_DECAY \
            --other_class_weight $OTHER_CLASS_WEIGHT \
            --other_precision_target $OTHER_PRECISION_TARGET \
            --device $DEVICE \
            --cosine_annealing_t0 $COSINE_ANNEALING_T0 \
            --cosine_annealing_tmult $COSINE_ANNEALING_TMULT \
            $MIXED_PRECISION_FLAG \
            $MODALITY_COMPLETION_FLAG \
            $FOCAL_LOSS_FLAG \
            $WEIGHTED_SAMPLER_FLAG \
            $RESUME_FLAG \
            $MODEL_PATH_FLAG \
            --output_dir $FORMATTED_OUTPUT_DIR"
        
        echo "运行命令: $TRAIN_CMD"
        eval $TRAIN_CMD 2>&1 | tee $LOG_DIR/${MODALITIES_STR}_${BACKBONE}_training_${DATETIME}.log
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
        
        # 设置阈值配置文件标志
        THRESHOLD_CONFIG_FLAG=""
        if [ ! -z "$THRESHOLD_CONFIG" ]; then
            THRESHOLD_CONFIG_FLAG="--threshold_config $THRESHOLD_CONFIG"
        fi
        
        # 评估命令
        EVAL_CMD="python /root/autodl-tmp/eye_multimodal_ensemble/improved_evaluate.py \
            --data_dir /root/autodl-tmp \
            --multimodal_index_path /root/autodl-tmp/eye_data_index/multimodal_dataset_index.json \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --max_images_per_modality $MAX_IMAGES \
            --device $DEVICE \
            $MODALITY_COMPLETION_FLAG \
            --model_path $MODEL_PATH \
            --other_threshold $OTHER_THRESHOLD \
            --other_precision_target $OTHER_PRECISION_TARGET \
            $THRESHOLD_CONFIG_FLAG \
            --output_dir $FORMATTED_OUTPUT_DIR"
        
        echo "运行命令: $EVAL_CMD"
        eval $EVAL_CMD 2>&1 | tee $LOG_DIR/${MODALITIES_STR}_test_${DATETIME}.log
        ;;
        
    *)
        echo "未知模式: $MODE"
        print_help
        exit 1
        ;;
esac

echo "完成!" 