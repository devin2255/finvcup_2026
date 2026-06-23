#!/usr/bin/env bash
# ============================================================
# 复赛推理入口（官方约定：/app/run.sh）
#   输入：只读挂载的私有测试集 /xydata（audio/ context/ text/）
#   输出：/app/submit/submit.csv（header: segment_id,c,na,i,bc,t）
#   单模型：models/whisper-large-v3 + models/Qwen3-0.6B + ckpt/ensemble_ep6.pt
#   推理时 VAP 特征喂零向量（见 configs/submit_single_vapfeat.yaml 说明）
# 用法（容器内）： bash run.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

# 全离线，禁用任何联网下载
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 测试集根目录默认 /xydata，可用环境变量覆盖（本地自测方便）
TEST_ROOT=${XYDATA_DIR:-/xydata}
OUT_DIR=/app/submit
mkdir -p "${OUT_DIR}"

python3 -m src.infer_test \
  --config configs/submit_single_vapfeat.yaml \
  --checkpoint ckpt/ensemble_ep6.pt \
  --threshold_file thresholds/best_thresholds.json \
  --test_root "${TEST_ROOT}" \
  --output_csv "${OUT_DIR}/submit.csv"

echo "[run.sh] done -> ${OUT_DIR}/submit.csv"
