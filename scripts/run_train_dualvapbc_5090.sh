#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/whisper_qwen0_6b_lmf_dualvapbc_5090.yaml}
ROOT=${FINVCUP_ROOT:-/root/autodl-tmp/finvcup_2026}

CPC_MODEL=${CPC_MODEL:-${ROOT}/models/60k_epoch4-d0f474de.pt}
VAP_LOCAL=${VAP_LOCAL:-${ROOT}/models/vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt}
BC_LOCAL=${BC_LOCAL:-${ROOT}/models/vap-bc_state_dict_ch_10hz_20000msec.pt}
VAP_CACHE=${VAP_CACHE:-${ROOT}/.cache/vap_bc_ch_21d}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${ROOT}"

echo "[dualvapbc] config=${CONFIG}"
echo "[dualvapbc] precompute cache=${VAP_CACHE}"
python -m src.precompute_vap \
  --config "${CONFIG}" \
  --maai_dir ./MaAI \
  --lang ch_kyoto --mode vap_mc \
  --frame_rate 10 --context_sec 20 \
  --device cuda \
  --cpc_model "${CPC_MODEL}" \
  --local_model "${VAP_LOCAL}" \
  --bc_enabled --bc_lang ch --bc_mode bc \
  --bc_local_model "${BC_LOCAL}" \
  --bc_tail_sec 2.0 \
  --out_dir "${VAP_CACHE}"

echo "[dualvapbc] train"
python -m src.train --config "${CONFIG}"
