#!/bin/bash
set -e

# ==========================================================
# 4×L20(48G) 四卡 DDP —— VAP 特征晚融合(第5模态)实验
#   调优版超参 + 预训练 vap_mc_ch_kyoto 逐帧话轮先验(预计算缓存)拼进融合层。
#   前置：先跑 src.precompute_vap 生成 vap_feat.cache_dir 下的缓存。
#   与 run_train_ensemble_4xL20_tuned 做 A/B，看 VAP 特征净增益(尤其 T/turn-shift)。
# 使用方法:
#   bash scripts/run_train_vapfeat.sh
# ==========================================================

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_vapfeat.yaml"
NUM_GPUS=${NUM_GPUS:-4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "4xL20 四卡 DDP —— VAP 特征晚融合实验"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29503 \
    -m src.train \
    --config "$CONFIG_FILE"

echo "=========================================="
echo "训练完成！输出: outputs/lmf_vapfeat/"
echo "=========================================="
