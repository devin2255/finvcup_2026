#!/bin/bash

# Phase 1 + Phase 2 优化版训练脚本
# 使用方法: bash scripts/run_train_optimized.sh

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_8g_optimized.yaml"
NUM_GPUS=1  # 8GB显存单卡训练

echo "=========================================="
echo "Phase 1 + Phase 2 优化训练"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS"
echo "=========================================="

if [ $NUM_GPUS -eq 1 ]; then
    # 单卡训练
    python -m src.train --config $CONFIG_FILE
else
    # 多卡训练
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        -m src.train \
        --config $CONFIG_FILE
fi

echo "=========================================="
echo "训练完成！"
echo "模型保存在: outputs/lmf_8g_optimized/checkpoints/"
echo "日志保存在: outputs/lmf_8g_optimized/logs/"
echo "=========================================="
