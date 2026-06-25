# ============================================================
# 复赛提交镜像（单模型推理）
# 基础镜像：官方 baseline（torch2.5.1 / cuda12.4 / python3.10）
# 构建（在仓库根目录、先执行 scripts/stage_submission.ps1 准备 models/ ckpt/ thresholds/）：
#   docker build -t finvcup-infer:v1 .
# ============================================================
FROM finvcup-registry.cn-shanghai.cr.aliyuncs.com/finvcup/team35:finv_baseline

WORKDIR /app

# 全离线运行
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# 依赖：Qwen3-0.6B 需要 transformers>=4.51（基础镜像自带版本可能偏旧）；
# 官方打分文件 import pynvml + requests，必须装。注意：不重装 torch/torchaudio，
# 沿用基础镜像的 CUDA 版本。
RUN pip install --no-cache-dir --upgrade \
        "transformers>=4.51,<5" \
        "tokenizers>=0.21" \
        pynvml \
        requests \
        pyyaml \
        tqdm

# 代码与配置
COPY src/ /app/src/
COPY configs/submit_single_vapfeat.yaml /app/configs/submit_single_vapfeat.yaml
COPY run.sh /app/run.sh
COPY train.sh /app/train.sh
COPY SUBMIT_README.md /app/README.md

# 模型权重（已裁剪）、checkpoint、阈值（由 scripts/stage_submission.ps1 生成）
COPY models/ /app/models/
COPY ckpt/ /app/ckpt/
COPY thresholds/ /app/thresholds/

# 兼容 Windows CRLF，去回车并赋可执行权限
RUN sed -i 's/\r$//' /app/run.sh /app/train.sh \
    && chmod +x /app/run.sh /app/train.sh \
    && mkdir -p /app/submit

CMD ["bash", "run.sh"]
