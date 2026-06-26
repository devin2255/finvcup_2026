#!/bin/bash
set -e

# ==========================================================
# Qwen2.5-Omni-3B 原生方案 —— 4×L20(48G) 四卡 DDP 训练。
#   模型：LoRA 微调 thinker + 5 路分类头（bf16 + 梯度检查点 + EMA）
#   入口：src/train_omni.py
# 使用方法:
#   bash scripts/run_train_omni_3b_4xL20.sh
# 冒烟（先小样本跑通管线，强烈建议第一次先跑这个）:
#   SMOKE=1 bash scripts/run_train_omni_3b_4xL20.sh
# ==========================================================

CONFIG_FILE="configs/qwen2_5_omni3b_4xL20.yaml"
NUM_GPUS=${NUM_GPUS:-4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[SMOKE] 小样本冒烟：64 train / 64 valid / 1 epoch / 5 steps"
  EXTRA+=(--max_train_samples 64 --max_valid_samples 64 --epochs 1 --max_steps_per_epoch 5)
fi

echo "=========================================="
echo "Qwen2.5-Omni-3B 四卡 DDP 训练"
echo "配置: $CONFIG_FILE   GPU: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=${MASTER_PORT:-29503} \
    -m src.train_omni \
    --config "$CONFIG_FILE" \
    "${EXTRA[@]}"

echo "=========================================="
echo "训练完成！产物目录: outputs/omni3b/"
echo "  best:     checkpoints/best_omni3b.pt"
echo "  ensemble: checkpoints/ensemble_ep*.pt + logs/ensemble_manifest.json"
echo "=========================================="
