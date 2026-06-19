#!/bin/bash
set -e

# ==========================================================
# 4 x L20 (48G) 四卡 DDP 集成训练
#   等效批 = batch_size(每卡128) × 4卡 × grad_accum(1) = 512，lr=2.4e-4
#   保存“最优模型 + 4 个次优模型”（各自带最优阈值），用于集成推理。
# 使用方法:
#   bash scripts/run_train_ensemble_4xL20.sh
# ==========================================================

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_ensemble_4xL20.yaml"
NUM_GPUS=${NUM_GPUS:-4}

# 四张卡都可见（如卡的编号不同，按需改）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# 模型已下载到本地（config 里 model_name 指向绝对路径），强制离线、不联网
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 降低显存碎片导致的 OOM 概率
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 避免 tokenizers 在多 worker 下 fork 警告
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "4xL20 四卡 DDP 集成训练 (ensemble_topk=5)"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29501 \
    -m src.train \
    --config "$CONFIG_FILE"

echo "=========================================="
echo "训练完成！"
echo "最优模型:   outputs/lmf_ensemble_4xL20/checkpoints/best_lmf_ensemble.pt"
echo "集成成员:   outputs/lmf_ensemble_4xL20/checkpoints/ensemble_ep*.pt"
echo "集成清单:   outputs/lmf_ensemble_4xL20/logs/ensemble_manifest.json"
echo "=========================================="
