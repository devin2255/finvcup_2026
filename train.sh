#!/usr/bin/env bash
# ============================================================
# 训练入口（产出提交所用 checkpoint 的完整配方）。
# 默认：4 卡 DDP 集成训练（与提交镜像里的 ensemble 成员一致）。
#
# 用法：
#   bash train.sh                       # 用默认 tuned 配置 + 4 卡
#   bash train.sh <config> <num_gpus>   # 自定义
#   NUM_GPUS=2 bash train.sh            # 用环境变量指定卡数
#
# 训练前请确认 config 里的数据/模型路径已指向你机器上的真实位置：
#   paths.train_audio_dir / train_text_dir / train_labels_dir
#   audio_encoder.model_name (whisper-large-v3) / text_encoder.model_name (Qwen3-0.6B)
# ============================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG_PATH="${1:-${CONFIG_PATH:-configs/whisper_qwen0_6b_lmf_ensemble_4xL20_tuned.yaml}}"
NUM_GPUS="${2:-${NUM_GPUS:-4}}"
CONFIG_PATH="${CONFIG_PATH//$'\r'/}"
NUM_GPUS="${NUM_GPUS//$'\r'/}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "config=${CONFIG_PATH}  num_gpus=${NUM_GPUS}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT:-29502}" \
  -m src.train \
  --config "${CONFIG_PATH}"

echo "训练完成。成员 checkpoint 与 ensemble_manifest.json 见 config 的 paths.checkpoints_dir / logs_dir。"
