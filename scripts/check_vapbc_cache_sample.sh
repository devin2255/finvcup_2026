#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/whisper_qwen0_6b_lmf_dualvapbc_5090.yaml}
ROOT=${FINVCUP_ROOT:-/root/autodl-tmp/finvcup_2026}

CPC_MODEL=${CPC_MODEL:-${ROOT}/models/60k_epoch4-d0f474de.pt}
VAP_LOCAL=${VAP_LOCAL:-${ROOT}/models/vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt}
BC_LOCAL=${BC_LOCAL:-${ROOT}/models/vap-bc_state_dict_ch_10hz_20000msec.pt}
VAP_CACHE=${VAP_CACHE:-${ROOT}/.cache/vap_bc_ch_21d}
SAMPLE_N=${SAMPLE_N:-5}
RTOL=${RTOL:-1e-5}
ATOL=${ATOL:-1e-6}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${ROOT}"
mkdir -p "${ROOT}/.cache"
TMP_CACHE=${TMP_CACHE:-$(mktemp -d "${ROOT}/.cache/vap_bc_ch_21d_check_${SAMPLE_N}_XXXXXX")}

echo "[vapbc-check] config=${CONFIG}"
echo "[vapbc-check] existing cache=${VAP_CACHE}"
echo "[vapbc-check] sample cache=${TMP_CACHE}"
echo "[vapbc-check] sample_n=${SAMPLE_N} rtol=${RTOL} atol=${ATOL}"

if [ ! -d "${VAP_CACHE}" ]; then
  echo "[vapbc-check] missing existing cache: ${VAP_CACHE}" >&2
  exit 2
fi

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
  --max_convs "${SAMPLE_N}" \
  --out_dir "${TMP_CACHE}" \
  --overwrite

CACHE_DIR="${VAP_CACHE}" SAMPLE_DIR="${TMP_CACHE}" RTOL="${RTOL}" ATOL="${ATOL}" python - <<'PY'
import glob
import os
import sys

import numpy as np

cache_dir = os.environ["CACHE_DIR"]
sample_dir = os.environ["SAMPLE_DIR"]
rtol = float(os.environ["RTOL"])
atol = float(os.environ["ATOL"])

sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.npy")))
if not sample_files:
    print(f"[vapbc-check] no sample files generated under {sample_dir}", file=sys.stderr)
    sys.exit(2)

ok = True
for sample_path in sample_files:
    name = os.path.basename(sample_path)
    cache_path = os.path.join(cache_dir, name)
    if not os.path.exists(cache_path):
        print(f"[vapbc-check] {name}: missing in existing cache")
        ok = False
        continue

    cached = np.load(cache_path)
    sampled = np.load(sample_path)
    if cached.shape != sampled.shape:
        print(f"[vapbc-check] {name}: shape mismatch cache={cached.shape} sample={sampled.shape}")
        ok = False
        continue

    exact = bool(np.array_equal(cached, sampled))
    close = bool(np.allclose(cached, sampled, rtol=rtol, atol=atol))
    max_abs_diff = float(np.max(np.abs(cached - sampled))) if cached.size else 0.0
    status = "OK" if close else "DIFF"
    print(
        f"[vapbc-check] {name}: shape={cached.shape} "
        f"exact={exact} max_abs_diff={max_abs_diff:.6g} {status}"
    )
    ok = ok and close

if ok:
    print("[vapbc-check] CACHE_SAMPLE_MATCH: existing cache can be reused")
    sys.exit(0)

print("[vapbc-check] CACHE_SAMPLE_MISMATCH: regenerate with VAP_OVERWRITE=1 or rebuild the cache")
sys.exit(2)
PY
