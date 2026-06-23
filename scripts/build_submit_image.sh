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
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-$(python3 -c 'import transformers;print(transformers.__version__)' 2>/dev/null || echo 4.57.1)}"

echo "==> 源路径"
echo "    MANIFEST    = ${MANIFEST}"
echo "    CKPT_DIR    = ${CKPT_DIR}"
echo "    WHISPER_DIR = ${WHISPER_DIR}"
echo "    QWEN_DIR    = ${QWEN_DIR}"
echo "    IMAGE_TAG   = ${IMAGE_TAG}"
echo "    transformers== ${TRANSFORMERS_VERSION}"

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

echo "==> 拷贝模型权重（较大，请稍候）"
cp -r "${WHISPER_DIR}" "${ASSETS}/models/whisper-large-v3"
cp -r "${QWEN_DIR}"    "${ASSETS}/models/Qwen3-0.6B"

echo "==> 暂存内容大小"
du -sh "${ASSETS}"/* 2>/dev/null || true

# ---- 构建 ----
echo "==> docker build"
docker build \
  -f Dockerfile.infer \
  --build-arg TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION}" \
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
