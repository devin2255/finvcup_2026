#!/bin/bash
set -e

# ==========================================================
# 4×L20(48G) 四卡 DDP —— 调好 lr 调度 + 权重 EMA 版
#   等效批 512 不变；lr 1.2e-4 / warmup 0.03 / epochs 40 / wd 0.03 / use_ema
#   修复上一版"大批量 + 高 lr"导致的早退化。
# 使用方法:
#   bash scripts/run_train_ensemble_4xL20_tuned.sh
# ==========================================================

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_ensemble_4xL20_tuned.yaml"
NUM_GPUS=${NUM_GPUS:-4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "4xL20 四卡 DDP 集成训练（调优版 + EMA）"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29502 \
    -m src.train \
    --config "$CONFIG_FILE"

echo "=========================================="
echo "训练完成！"
echo "最优模型(EMA): outputs/lmf_ensemble_4xL20_tuned/checkpoints/best_lmf_ensemble.pt"
echo "集成成员(EMA): outputs/lmf_ensemble_4xL20_tuned/checkpoints/ensemble_ep*.pt"
echo "集成清单:      outputs/lmf_ensemble_4xL20_tuned/logs/ensemble_manifest.json"
echo "=========================================="
