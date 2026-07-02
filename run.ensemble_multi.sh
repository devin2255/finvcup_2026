#!/usr/bin/env bash
# ============================================================
# 跨模型软投票推理入口（vapwin ⊕ dualch / dualvap_bcd 等）：
#   阶段 A：/opt/maai-env 现场预计算 18 维 VAP 窗口特征（window=20，一份缓存
#           全部模型组共用；单帧模型在 forward 里自动取末帧）。
#   阶段 B：主环境按 SPEC（configs/submit_multi_spec.json）逐组逐成员推理，
#           组内概率均值 -> 组间加权均值 -> 阈值加权均值二值化
#           -> /app/submit/submit.csv
# 60 分钟硬限：成员总数 × 单模型耗时(~8-12min)，spec 里的 topk 控制预算。
# 用法（容器内）： bash run.ensemble_multi.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TEST_ROOT=${XYDATA_DIR:-/xydata}
OUT_DIR=/app/submit
VAP_CACHE=/app/.cache/vap_test
SPEC=${MULTI_SPEC:-configs/submit_multi_spec.json}
mkdir -p "${OUT_DIR}" "${VAP_CACHE}"

CPC_MODEL=/app/models/60k_epoch4-d0f474de.pt
VAP_LOCAL=/app/models/vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt
MAAI_DIR=/app/MaAI

t0=$(date +%s)

echo "[run] === Stage A: VAP precompute (workers=${VAP_WORKERS:-8}) ==="
/opt/maai-env/bin/python -m src.precompute_vap_test \
  --maai_dir "${MAAI_DIR}" \
  --lang ch_kyoto --mode vap_mc \
  --frame_rate 10 --context_sec 20 \
  --cpc_model "${CPC_MODEL}" \
  --vap_local_model "${VAP_LOCAL}" \
  --test_root "${TEST_ROOT}" \
  --out_dir "${VAP_CACHE}" \
  --sample_rate 16000 \
  --window "${VAP_WINDOW:-20}" \
  --workers "${VAP_WORKERS:-8}"

t1=$(date +%s)
echo "[run] stage A done in $((t1 - t0))s"

echo "[run] === Stage B: multi-model soft-vote inference (spec=${SPEC}) ==="
python3 -m src.infer_ensemble_multi \
  --spec "${SPEC}" \
  --test_root "${TEST_ROOT}" \
  --output_csv "${OUT_DIR}/submit.csv"

t2=$(date +%s)
echo "[run] stage B done in $((t2 - t1))s"
echo "[run] total $((t2 - t0))s -> ${OUT_DIR}/submit.csv"
