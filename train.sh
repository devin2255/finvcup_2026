#!/usr/bin/env bash
# ============================================================
# 训练入口（复赛阶段二复现训练用；阶段一仅推理可忽略）
#   用法： bash train.sh [GPU数量, 默认1]
#   训练数据需挂载到 configs/submit_single_vapfeat.yaml 的 paths.train_*_dir
#   （默认 /app/data/train/{audio,text,labels}）。
#
# 注意：本单模型在训练时使用了 VAP 第5模态特征。若要完整复现该特征，
# 需先用 MaAI(vap_mc_ch_kyoto)+CPC 权重对训练集预计算 VAP 缓存：
#   python -m src.precompute_vap --config configs/submit_single_vapfeat.yaml \
#       --maai_dir ./MaAI --lang ch_kyoto --frame_rate 10 --context_sec 20 \
#       --device cuda --cpc_model /path/to/60k_epoch4-d0f474de.pt \
#       --out_dir /app/.cache/vap_ch_kyoto
# 未提供缓存时训练仍可运行（vap_feat 退化为零向量），但与提交权重不完全一致。
# 详见 README.md。
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

NPROC=${1:-1}
NPROC="${NPROC//$'\r'/}"

torchrun --nproc_per_node="${NPROC}" -m src.train --config configs/submit_single_vapfeat.yaml
