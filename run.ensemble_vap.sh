#!/usr/bin/env bash
# ============================================================
# v2 复赛推理入口：真 VAP + 5 模型 ensemble
#   阶段 A：用 /opt/maai-env（torch 2.6 + MaAI）对 /xydata/audio/*.wav
#           现场预计算 18 维 VAP 特征 -> /app/.cache/vap_test/<seg>.npy
#   阶段 B：主环境（torch 2.5.1 + 我们的模型）跑 5 个 checkpoint 投票
#           -> /app/submit/submit.csv （header: segment_id,c,na,i,bc,t）
# 推理超时硬上限 60 分钟；本脚本两段共耗时主要看测试集大小。
# 用法（容器内）： bash run.ensemble_vap.sh
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
mkdir -p "${OUT_DIR}" "${VAP_CACHE}"

CPC_MODEL=/app/models/60k_epoch4-d0f474de.pt
VAP_LOCAL=/app/models/vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt
MAAI_DIR=/app/MaAI

t0=$(date +%s)

# -------- 阶段 A：VAP 预计算（maai-env，多进程并行） --------
# 单进程基准 ~80 frame/s -> 1000 段 ~66 min 超时；4 进程 ~18-22 min；8 进程更快
# （VAP/CPC 是小模型，单进程 GPU 利用率低 -> 加进程能近线性提速）。
# 默认 8；评测机直接 bash run.sh（注入不了 env），靠此内置默认。
# 本地 smoke 可 docker run -e VAP_WORKERS=12 覆盖；若 OOM 降到 6/4。
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

# -------- 阶段 B：ensemble 推理（主环境） --------
# 总预算 60min 的大头在这里：每模型 ~8-12min，5 个串行 ~40-60min。
# ENSEMBLE_TOPK 设非空可只用前 K 个最强成员（manifest 已按指标降序）。
# 时间紧时设 3（约 30min）；想要单模型快测设 1。默认空=全部 5 个。
echo "[run] === Stage B: ensemble inference (topk=${ENSEMBLE_TOPK:-all}) ==="
python3 -m src.infer_ensemble \
  --config configs/submit_ensemble_vap.yaml \
  --test_root "${TEST_ROOT}" \
  --vap_feat_dir "${VAP_CACHE}" \
  --ensemble_mode "${ENSEMBLE_MODE:-soft}" \
  --threshold_mode "${ENSEMBLE_THRESHOLD_MODE:-mean}" \
  ${ENSEMBLE_TOPK:+--topk "${ENSEMBLE_TOPK}"} \
  --output_csv "${OUT_DIR}/submit.csv"

t2=$(date +%s)
echo "[run] stage B done in $((t2 - t1))s"
echo "[run] total $((t2 - t0))s -> ${OUT_DIR}/submit.csv"
