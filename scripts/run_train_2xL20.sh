#!/bin/bash
set -e

# ==========================================================
# 2 x L20 (48GB) 双卡 DDP 训练
# 使用方法: bash scripts/run_train_2xL20.sh
# ==========================================================

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_2xL20.yaml"
NUM_GPUS=2

# 两张卡都可见（如服务器上卡更多，按需改成 "0,1"）
export CUDA_VISIBLE_DEVICES=0,1
# 国内拉取 HF 权重走镜像（whisper-large-v3 / Qwen3-0.6B）
export HF_ENDPOINT=https://hf-mirror.com
# 降低显存碎片导致的 OOM 概率
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 避免 tokenizers 在多 worker 下 fork 警告
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "2xL20 双卡 DDP 训练"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    -m src.train \
    --config $CONFIG_FILE

echo "=========================================="
echo "训练完成！"
echo "模型保存在: outputs/lmf_2xL20/checkpoints/"
echo "日志保存在: outputs/lmf_2xL20/logs/"
echo "=========================================="
