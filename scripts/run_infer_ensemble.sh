#!/usr/bin/env bash
set -euo pipefail

# 集成推理：用 ensemble_manifest.json 里的 top-N 成员（可选加上 best.pt），
# 在验证集上对集成平均概率重拟合 per-label 阈值后预测测试集。
#
# 用法:
#   bash scripts/run_infer_ensemble.sh <manifest_json> <best_pt> <test_root> <output_csv> [config_path]

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/run_infer_ensemble.sh <manifest_json> <best_pt> <test_root> <output_csv> [config_path]"
  exit 1
fi

MANIFEST=${1//$'\r'/}
BEST_PT=${2//$'\r'/}
TEST_ROOT=${3//$'\r'/}
OUT_CSV=${4//$'\r'/}
CONFIG_PATH=${5:-configs/whisper_qwen0_6b_lmf_2xL20.yaml}
CONFIG_PATH=${CONFIG_PATH//$'\r'/}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python3 -m src.infer_ensemble \
    --config "${CONFIG_PATH}" \
    --manifest "${MANIFEST}" \
    --include_best "${BEST_PT}" \
    --test_root "${TEST_ROOT}" \
    --output_csv "${OUT_CSV}" \
    --tune_thresholds_on_valid \
    --weighted
