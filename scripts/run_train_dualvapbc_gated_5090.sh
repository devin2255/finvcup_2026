#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/whisper_qwen0_6b_lmf_dualvapbc_gated_5090.yaml}
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/finvcup_2026}
CACHE_DIR=${VAP_BC_CACHE:-${PROJECT_ROOT}/.cache/vap_bc_ch_21d}
VAP_OVERWRITE=${VAP_OVERWRITE:-0}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CACHE_DIR

cd "${PROJECT_ROOT}"

mkdir -p "${CACHE_DIR}"

echo "[dualvapbc-gated] config=${CONFIG}"
echo "[dualvapbc-gated] cache=${CACHE_DIR} overwrite=${VAP_OVERWRITE}"

PRECOMPUTE_ARGS=(
  -m src.precompute_vap
  --config "${CONFIG}"
  --maai_dir "${PROJECT_ROOT}/MaAI"
  --lang ch_kyoto
  --mode vap_mc
  --frame_rate 10
  --context_sec 20
  --cpc_model "${PROJECT_ROOT}/models/60k_epoch4-d0f474de.pt"
  --vap_local_model "${PROJECT_ROOT}/models/vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt"
  --train_audio_dir "${PROJECT_ROOT}/train/audio"
  --out_dir "${CACHE_DIR}"
  --sample_rate 16000
  --bc_enabled
  --bc_lang ch
  --bc_mode bc
  --bc_local_model "${PROJECT_ROOT}/models/bc_state_dict_ch_10hz_20000msec.pt"
  --bc_tail_sec 2.0
)

if [[ "${VAP_OVERWRITE}" == "1" ]]; then
  PRECOMPUTE_ARGS+=(--overwrite)
fi

echo "[dualvapbc-gated] precompute/check cache"
/opt/maai-env/bin/python "${PRECOMPUTE_ARGS[@]}"

python - <<'PY'
import os
from pathlib import Path
import numpy as np

cache = Path(os.environ["CACHE_DIR"])
sample = next(cache.glob("*.npy"), None)
if sample is None:
    raise SystemExit(f"[dualvapbc-gated] no cache files under {cache}")
arr = np.load(sample)
if arr.ndim != 2 or arr.shape[1] != 21:
    raise SystemExit(f"[dualvapbc-gated] bad cache shape {arr.shape} in {sample}")
print(f"[dualvapbc-gated] cache shape OK {arr.shape} {sample}")
PY

echo "[dualvapbc-gated] train"
python -m src.train --config "${CONFIG}"
