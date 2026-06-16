#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# 集成推理：用 ensemble_manifest.json 里的多个模型逐标签多数投票
# （每个模型用自己的最优阈值）。
# 使用方法:
#   bash scripts/run_infer_ensemble.sh <test_root> <output_pred_csv> [config_path] [topk]
# ==========================================================

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_infer_ensemble.sh <test_root> <output_pred_csv> [config_path] [topk]"
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TEST_ROOT=$1
OUT_CSV=$2
CONFIG_PATH=${3:-configs/whisper_qwen0_6b_lmf_ensemble.yaml}
TOPK=${4:-}

# 兼容 Windows 换行符
TEST_ROOT="${TEST_ROOT//$'\r'/}"
OUT_CSV="${OUT_CSV//$'\r'/}"
CONFIG_PATH="${CONFIG_PATH//$'\r'/}"
TOPK="${TOPK//$'\r'/}"

EXTRA=()
if [[ -n "$TOPK" ]]; then
  EXTRA+=(--topk "$TOPK")
fi

python3 -m src.infer_ensemble \
  --config "${CONFIG_PATH}" \
  --test_root "${TEST_ROOT}" \
  --output_csv "${OUT_CSV}" \
  "${EXTRA[@]}"
