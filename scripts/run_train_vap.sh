#!/bin/bash
set -e

# ==========================================================
# 4×L20(48G) 四卡 DDP —— VAP 辅助头实验（调优版 + 多任务 VAP）
#   等效批 512、lr 1.2e-4、EMA 同调优版；额外加 VAP 辅助损失(λ=0.3)。
#   与 run_train_ensemble_4xL20_tuned 做 A/B，对比 VAP 净增益。
# 使用方法:
#   bash scripts/run_train_vap.sh
# ==========================================================

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_vap.yaml"
NUM_GPUS=${NUM_GPUS:-4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "4xL20 四卡 DDP —— VAP 辅助头实验"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29503 \
    -m src.train \
    --config "$CONFIG_FILE"

echo "=========================================="
echo "训练完成！输出: outputs/lmf_vap/"
echo "=========================================="
