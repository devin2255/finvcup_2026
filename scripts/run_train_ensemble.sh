#!/bin/bash
set -e

# ==========================================================
# 集成训练：8g_optimized 调参 + batch_size=48 + ensemble_topk=5
# 保存“最优模型 + 4 个次优模型”（各自带最优阈值），用于集成推理。
# 使用方法:
#   bash scripts/run_train_ensemble.sh           # 单卡（默认）
#   NUM_GPUS=2 bash scripts/run_train_ensemble.sh  # 多卡 DDP
# ==========================================================

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_ensemble.yaml"
NUM_GPUS=${NUM_GPUS:-1}

# 卡可见性（多卡时按需改成 "0,1" 等）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# 模型已下载到本地（config 里 model_name 指向绝对路径），强制离线、不联网
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 降低显存碎片导致的 OOM 概率
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 避免 tokenizers 在多 worker 下 fork 警告
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "集成训练 (ensemble_topk=5)"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    python -m src.train --config "$CONFIG_FILE"
else
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        -m src.train \
        --config "$CONFIG_FILE"
fi

echo "=========================================="
echo "训练完成！"
echo "最优模型:   outputs/lmf_ensemble/checkpoints/best_lmf_ensemble.pt"
echo "集成成员:   outputs/lmf_ensemble/checkpoints/ensemble_ep*.pt"
echo "集成清单:   outputs/lmf_ensemble/logs/ensemble_manifest.json"
echo "=========================================="
