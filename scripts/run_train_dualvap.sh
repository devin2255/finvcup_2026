#!/bin/bash
set -e

# ==========================================================
# Stage 1（方案2）：干净版 dualch + vapwin 组合训练（lmf_dualvap）
#   stereo 双声道活动分支 + VAP 窗口特征(18d, window=20)，不含 21d BC 特征。
#   前置：.cache/vap_ch_kyoto 训练缓存（与 lmf_vapfeat/lmf_vapwin 共用，无需重算）。
#   A/B 基线：lmf_dualch（看 vapwin 叠加净增益）与 lmf_vapwin（看 stereo 叠加净增益）。
# 使用方法:
#   bash scripts/run_train_dualvap.sh          # 默认单卡（5090）
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train_dualvap.sh
# ==========================================================

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_dualvap.yaml"
NUM_GPUS=${NUM_GPUS:-1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "Stage 1: dualch + vapwin 组合训练 (lmf_dualvap)"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29510 \
    -m src.train \
    --config "$CONFIG_FILE"

echo "=========================================="
echo "训练完成！输出: outputs/lmf_dualvap/"
echo "=========================================="
