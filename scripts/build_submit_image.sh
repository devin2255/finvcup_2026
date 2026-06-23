#!/usr/bin/env bash
# ============================================================
# 在【训练服务器】上构建复赛提交镜像。
# 因为模型权重 / 成员 checkpoint / manifest 都在服务器上（不在 git 里），
# 本脚本会先把它们暂存到 docker_assets/，再 docker build。
#
# 用法（在仓库根目录执行）：
#   bash scripts/build_submit_image.sh
#
# 可用环境变量覆盖默认源路径：
#   RUN_DIR      训练输出目录（含 checkpoints/ 与 logs/ensemble_manifest.json）
#   CKPT_DIR     成员 checkpoint 目录       （默认 $RUN_DIR/checkpoints）
#   MANIFEST     ensemble_manifest.json 路径（默认 $RUN_DIR/logs/ensemble_manifest.json）
#   WHISPER_DIR  whisper-large-v3 权重目录
#   QWEN_DIR     Qwen3-0.6B 权重目录
#   IMAGE_TAG    镜像 tag（默认 finvcup-infer:latest）
#   TRANSFORMERS_VERSION  默认自动探测当前 python 环境的 transformers 版本
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_DIR="${RUN_DIR:-outputs/lmf_ensemble_4xL20_tuned}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/checkpoints}"
MANIFEST="${MANIFEST:-${RUN_DIR}/logs/ensemble_manifest.json}"
WHISPER_DIR="${WHISPER_DIR:-/mnt/workspace/dorihue/modelscope/whisper-large-v3}"
QWEN_DIR="${QWEN_DIR:-/mnt/workspace/dorihue/modelscope/Qwen3-0.6B}"
IMAGE_TAG="${IMAGE_TAG:-finvcup-infer:latest}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime}"
# 自动探测当前（训练）环境的版本注入镜像，确保镜像与训练机一致
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-$(python3 -c 'import transformers;print(transformers.__version__)' 2>/dev/null || echo 4.57.1)}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-$(python3 -c 'import torchaudio;print(torchaudio.__version__)' 2>/dev/null || echo 2.11.0+cu128)}"

echo "==> 源路径"
echo "    MANIFEST    = ${MANIFEST}"
echo "    CKPT_DIR    = ${CKPT_DIR}"
echo "    WHISPER_DIR = ${WHISPER_DIR}"
echo "    QWEN_DIR    = ${QWEN_DIR}"
echo "    IMAGE_TAG   = ${IMAGE_TAG}"
echo "    BASE_IMAGE  = ${BASE_IMAGE}"
echo "    transformers== ${TRANSFORMERS_VERSION}   torchaudio== ${TORCHAUDIO_VERSION}"

# ---- 校验源文件存在 ----
for p in "${MANIFEST}" "${CKPT_DIR}" "${WHISPER_DIR}" "${QWEN_DIR}"; do
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: 找不到 ${p}"
    exit 1
  fi
done

# ---- 暂存到 docker_assets/ ----
ASSETS="${ROOT}/docker_assets"
echo "==> 清理并重建 ${ASSETS}"
rm -rf "${ASSETS}"
mkdir -p "${ASSETS}/models" "${ASSETS}/checkpoints" "${ASSETS}/logs"

cp "${MANIFEST}" "${ASSETS}/logs/ensemble_manifest.json"

echo "==> 拷贝 manifest 中的成员 checkpoint"
python3 - "${MANIFEST}" "${CKPT_DIR}" "${ASSETS}/checkpoints" <<'PY'
import json, os, shutil, sys
manifest, ckpt_dir, dst = sys.argv[1:4]
data = json.load(open(manifest, encoding="utf-8"))
members = data.get("members", [])
assert members, f"manifest 里没有成员: {manifest}"
for mb in members:
    name = mb["name"]
    src = os.path.join(ckpt_dir, name)
    assert os.path.exists(src), f"成员 checkpoint 缺失: {src}"
    shutil.copy2(src, os.path.join(dst, name))
    print(f"   staged {name}  (epoch={mb.get('epoch')}, metric={mb.get('metric')})")
print(f"   共 {len(members)} 个成员")
PY

echo "==> 瘦身拷贝模型权重（只拷推理必需文件，优先 safetensors；跳过 *.bin / original/ / fp32 副本）"
python3 - "${WHISPER_DIR}" "${ASSETS}/models/whisper-large-v3" "${QWEN_DIR}" "${ASSETS}/models/Qwen3-0.6B" <<'PY'
import glob, os, shutil, sys

# 只拷推理必需：单文件 fp16 权重 model.safetensors + 各种小配置/分词器。
# 明确跳过：fp32 副本(*.fp32*)、所有 *.bin、flax(*.msgpack)、tf(*.h5)、子目录、modelscope 杂项。
BIG = 50 * 1024 * 1024
SKIP_NAMES = {"README.md", "LICENSE", "configuration.json", ".msc", ".mv"}

def want_small(name):
    if name in SKIP_NAMES:
        return False
    if "fp32" in name:                                   # fp32 的 index 等
        return False
    low = name.lower()
    if low.endswith((".bin", ".msgpack", ".h5", ".safetensors", ".pt", ".onnx")):
        return False                                     # 权重单独处理，其余大格式不要
    return True

def stage(src, dst):
    os.makedirs(dst, exist_ok=True)
    # 1) 小配置/分词器文件
    for name in sorted(os.listdir(src)):
        p = os.path.join(src, name)
        if os.path.isfile(p) and os.path.getsize(p) < BIG and want_small(name):
            shutil.copy2(p, os.path.join(dst, name))
    # 2) 权重：优先单文件 model.safetensors；其次标准分片；最后非 fp32 的 pytorch_model.bin
    if os.path.isfile(os.path.join(src, "model.safetensors")):
        shutil.copy2(os.path.join(src, "model.safetensors"), os.path.join(dst, "model.safetensors"))
    elif os.path.isfile(os.path.join(src, "model.safetensors.index.json")):
        shutil.copy2(os.path.join(src, "model.safetensors.index.json"),
                     os.path.join(dst, "model.safetensors.index.json"))
        for p in glob.glob(os.path.join(src, "model-*-of-*.safetensors")):
            shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
    else:
        bins = [b for b in sorted(glob.glob(os.path.join(src, "pytorch_model*.bin")))
                if "fp32" not in os.path.basename(b)]
        assert bins, f"找不到可用权重(model.safetensors / 标准分片 / pytorch_model.bin): {src}"
        for b in bins:
            shutil.copy2(b, os.path.join(dst, os.path.basename(b)))
        idx = os.path.join(src, "pytorch_model.bin.index.json")
        if os.path.isfile(idx):
            shutil.copy2(idx, os.path.join(dst, "pytorch_model.bin.index.json"))
    total = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    print(f"   {os.path.basename(dst):20s} 大小={total/1e9:.2f} GB  文件: {sorted(os.listdir(dst))}")

stage(sys.argv[1], sys.argv[2])
stage(sys.argv[3], sys.argv[4])
PY

echo "==> 暂存内容大小"
du -sh "${ASSETS}"/* 2>/dev/null || true
ASSET_GB=$(du -s "${ASSETS}" | awk '{printf "%.1f", $1/1024/1024}')
echo "    docker_assets 合计 ${ASSET_GB} GB（基础镜像另算 ~9GB，目标镜像 ≤20GB）"

# ---- 构建 ----
# DSW 等受限容器里 BuildKit 的 mount 会被内核拒绝（operation not permitted），
# 关掉 BuildKit 用老构建器即可绕过。
export DOCKER_BUILDKIT=0
echo "==> docker build (DOCKER_BUILDKIT=0)"
docker build \
  -f Dockerfile.infer \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION}" \
  --build-arg TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION}" \
  -t "${IMAGE_TAG}" \
  .

echo ""
echo "==> 构建完成: ${IMAGE_TAG}"
docker images "${IMAGE_TAG}"
echo ""
echo "下一步本地自测（确认能产出 submit.csv）："
echo "  docker run --rm -it --gpus all \\"
echo "    -v \$(pwd)/test_data:/xydata:ro \\"
echo "    -v \$(pwd)/_submit_out:/app/submit \\"
echo "    ${IMAGE_TAG} bash -lc 'bash run.sh && head /app/submit/submit.csv'"
