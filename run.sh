#!/usr/bin/env bash
# ============================================================
# 复赛镜像入口脚本（放在 /app 下）。
# 容器内直接执行：bash run.sh
# 读取 /xydata 的测试集，集成推理，把结果写到 /app/submit/submit.csv。
# ============================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 缓存路径（与官方 run.sh 对齐）+ 离线 + 单卡（评测 GPU 不联网，骨干权重已烤进镜像本地加载）
export HF_HOME="${HF_HOME:-/app/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/app/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/app/.cache/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TEST_ROOT="${TEST_ROOT:-/xydata}"
OUT_CSV="${OUT_CSV:-./submit/submit.csv}"
CONFIG_PATH="${CONFIG_PATH:-configs/docker_infer_ensemble.yaml}"

if [[ ! -d "${TEST_ROOT}" ]]; then
  echo "Error: 测试集目录不存在: ${TEST_ROOT}"
  echo "请把测试集以只读方式挂载到 /xydata，期望结构："
  echo "  /xydata/audio  /xydata/text  /xydata/context"
  exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"

# 集成方式：默认 soft（概率平均软投票，统一用 best pt 阈值）。
#   VOTE=hard 切回逐标签多数硬投票。
VOTE="${VOTE:-soft}"

EXTRA=(--vote "${VOTE}")
# 可选环境变量覆盖：
#   TOPK=3         只用 manifest 前 K 个成员
#   BATCH_SIZE=32  覆盖推理批大小（显存不足时调小）
#   MAX_SEGMENTS=20  冒烟测试只跑前 N 条
[[ -n "${TOPK:-}" ]]         && EXTRA+=(--topk "${TOPK}")
[[ -n "${BATCH_SIZE:-}" ]]   && EXTRA+=(--batch_size "${BATCH_SIZE}")
[[ -n "${MAX_SEGMENTS:-}" ]] && EXTRA+=(--max_segments "${MAX_SEGMENTS}")

echo "test_root=${TEST_ROOT}"
echo "output_csv=${OUT_CSV}"
echo "config=${CONFIG_PATH}"

python3 -m src.infer_ensemble \
  --config "${CONFIG_PATH}" \
  --test_root "${TEST_ROOT}" \
  --output_csv "${OUT_CSV}" \
  "${EXTRA[@]}"

echo "Done. Results: ${OUT_CSV}"
