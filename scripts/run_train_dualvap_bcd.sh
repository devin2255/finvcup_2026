#!/bin/bash
set -e

# ==========================================================
# Stage 2（方案3+4）：在 dualvap 基座上叠加
#   ③ BC 逐 chunk 密集监督（bc_dense 辅助头）
#   ④ 动态上下文增广扩到 (0,30]（min_context_chunks 125 -> 25）
# A/B 基线：lmf_dualvap（Stage 1）。
# 观察指标：bc_best_f1 / bcdense_best_f1（epoch 日志与 eval_epoch_*.json）。
# 使用方法:
#   bash scripts/run_train_dualvap_bcd.sh
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train_dualvap_bcd.sh
# ==========================================================

CONFIG_FILE="configs/whisper_qwen0_6b_lmf_dualvap_bcd.yaml"
NUM_GPUS=${NUM_GPUS:-4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "Stage 2: dualvap + BC密集监督 + 短上下文增广 (lmf_dualvap_bcd)"
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29511 \
    -m src.train \
    --config "$CONFIG_FILE"

echo "=========================================="
echo "训练完成！输出: outputs/lmf_dualvap_bcd/"
echo "=========================================="
