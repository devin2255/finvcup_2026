#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_infer.sh <checkpoint_path>  <output_pred_csv> <test_root> [config_path]"
  exit 1
fi

export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/nlp/.cache/torch
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CKPT_PATH=$1
OUT_CSV=$2
TEST_ROOT=$3
CONFIG_PATH=${4:-configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml}
CKPT_PATH="${CKPT_PATH//$'\r'/}"
OUT_CSV="${OUT_CSV//$'\r'/}"
TEST_ROOT="${TEST_ROOT//$'\r'/}"
CONFIG_PATH="${CONFIG_PATH//$'\r'/}"

python3 -m src.infer_test --config "${CONFIG_PATH}" --checkpoint "${CKPT_PATH}"  --test_root "${TEST_ROOT}" --output_csv "${OUT_CSV}"
