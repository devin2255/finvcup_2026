# 第11届信也科技杯 · 对话轮次预测（Turn-Taking）复赛方案

给定过去 ≤30s 的双声道对话（音频 + ASR 文本 + 历史 chunk 标签），预测未来 2s（25×80ms chunk）
内是否出现 5 类语音事件 `C / NA / I / BC / T`（event-level 多标签，sigmoid + BCE）。
因果约束：只使用上下文，不读取未来音频或未来标签。

## 方案概述

**多模态融合模型**（`src/models/multimodal_baseline.py`）：

| 模态 | 编码器 |
|------|--------|
| 音频 | Whisper-large-v3 encoder（冻结，末 2 层微调）+ 注意力池化 |
| 文本 | Qwen3-0.6B（冻结）+ 尾部注意力池化 |
| 历史标签 | 标签序列 CNN（tail + full 双分支） |
| 手工特征 | 标签分布/转移/时间衰减/话轮间隔等统计量 |

→ **低秩张量融合 + 自适应门控**（`MultimodalFusion`）→ 5 路 sigmoid 头。

**训练**（`src/train.py`）：focal loss + capped per-label pos_weight + label smoothing，
权重 EMA，动态上下文增强（(0,30] 变长），4 卡 DDP。

**集成推理**（`src/infer_ensemble.py`）：训练时按 valid `best_f1` 保留 top-5 成员
（每个成员自带各标签最优阈值，写入 `ensemble_manifest.json`）；推理时逐成员用自己的阈值二值化，
再做**逐标签多数投票**得到最终 0/1。

---

## 环境要求

- Linux + NVIDIA GPU，CUDA 12.4（推理镜像基于 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`）
- Python 3.10，PyTorch 2.5.1 / torchaudio 2.5.1 / **transformers 4.57.x**（见「特殊注意事项」）

```bash
conda create -n finvcup python=3.10 -y && conda activate finvcup
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

预训练骨干（建议提前下到本地，离线加载）：`openai/whisper-large-v3`、`Qwen/Qwen3-0.6B`。
把它们的本地路径填到配置的 `audio_encoder.model_name` / `text_encoder.model_name`。

---

## 数据准备

**训练集**（整通对话，路径见配置 `paths.train_*`）：

| 路径 | 说明 |
|------|------|
| `train/audio/<conv_id>.wav` | 双声道整段音频 |
| `train/text/<conv_id>.json` | ASR 转写 |
| `train/labels/<conv_id>.npy` | 逐 chunk 标签，`0~4` = `C/T/BC/I/NA` |

**测试集 / 复赛私有集**（推理时以只读挂载到 `/xydata`）：

```
/xydata
├── audio/<segment_id>.wav     # 上下文音频（复赛为 (0,30] 变长）
├── text/<segment_id>.json     # ASR
└── context/<segment_id>.npy   # 上下文标签序列
```

---

## 1）训练 —— `train.sh`

```bash
bash train.sh                      # 默认：tuned 集成配置 + 4 卡 DDP
bash train.sh <config> <num_gpus>  # 自定义配置与卡数
NUM_GPUS=2 bash train.sh           # 也可用环境变量
```

- 默认配置：`configs/whisper_qwen0_6b_lmf_ensemble_4xL20_tuned.yaml`
- 产物：`<checkpoints_dir>/` 下的 top-5 成员 `ensemble_ep*.pt` 与 best，
  以及 `<logs_dir>/ensemble_manifest.json`（含各成员各标签最优阈值）。
- **运行前**请把配置里的 `paths.train_*` 与两个 `model_name` 改成你机器上的真实路径。

## 2）推理 —— `run.sh`（程序入口）

`run.sh` 是评测/审核的统一入口：读 `/xydata` 测试集 → 集成推理 → 把结果写到
**`/app/submit/submit.csv`**。

```bash
# 容器内（工作目录 /app）直接执行：
bash run.sh
```

等价命令（`run.sh` 内部即执行）：

```bash
python3 -m src.infer_ensemble \
  --config configs/docker_infer_ensemble.yaml \
  --test_root /xydata \
  --output_csv /app/submit/submit.csv
```

可选环境变量：`TEST_ROOT`、`OUT_CSV`、`CONFIG_PATH`、`TOPK`（只用前 K 个成员）、
`BATCH_SIZE`（显存不足调小）、`MAX_SEGMENTS`（冒烟测试）。

**输出格式**（英文逗号分隔，含表头）：

```
segment_id,c,na,i,bc,t
0000,1,1,0,0,1
...
```

## 3）Docker 镜像（复赛提交）

详见 [DOCKER_SUBMIT.md](DOCKER_SUBMIT.md)。在训练服务器上一键构建：

```bash
bash scripts/build_submit_image.sh        # 暂存模型/checkpoint/manifest → docker build
```

本地自测（确认能产出 submit.csv）：

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/test_data:/xydata:ro \
  -v $(pwd)/_submit_out:/app/submit \
  finvcup-infer:latest bash -lc 'bash run.sh && head /app/submit/submit.csv'
```

---

## ⚠️ 特殊注意事项（运行不成功通常出在这几条）

1. **模型结构字段必须与训练 checkpoint 一致**：成员 checkpoint 为「瘦身」格式（只存可训练参数），
   以 `strict=False` 叠加到预训练骨干上。若 `configs/docker_infer_ensemble.yaml` 的
   `audio_encoder / text_encoder / context_encoder / fusion / labels` 与训练那份不一致，
   会静默错配、跑分异常。本仓库的 docker 配置已对齐 `..._4xL20_tuned.yaml`。
2. **transformers 版本**：`WhisperAudioEncoder._forward_encoder_split` 镜像了 transformers
   **4.57.x** 的 WhisperEncoder 内部实现（`unfreeze_layers>0` 时推理会走这条分支）。
   `scripts/build_submit_image.sh` 会自动探测当前环境的 transformers 版本注入镜像——
   请在**训练用的同一个 conda 环境**里执行构建。
3. **离线加载**：评测 GPU 不联网。骨干权重已拷进镜像 `/app/models/`，并设
   `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`，推理不会联网下载。
4. **换行符**：`*.sh` 必须为 LF（已用 `.gitattributes` 固定），否则容器内 `bash run.sh` 会因 `\r` 报错。
5. **资源约束**：镜像 ≤ 20GB、模型参数 ≤ 8B（单成员 ≈1.2B，推理一次只载一个成员）、
   推理 ≤ 60 分钟（超时调小 `BATCH_SIZE` 或 `TOPK`）。

## 仓库结构

```text
train.sh                       # 训练入口
run.sh                         # 推理入口 → /app/submit/submit.csv
Dockerfile.infer               # 推理镜像
DOCKER_SUBMIT.md               # 镜像构建/提交手册
configs/                       # 训练配置 + docker_infer_ensemble.yaml
scripts/build_submit_image.sh  # 一键构建提交镜像
src/
  data/dataset.py              # 训练/测试 Dataset + collate
  models/multimodal_baseline.py
  train.py                     # 训练（DDP + EMA + 集成成员保存）
  infer_ensemble.py            # 集成推理（多数投票）→ submit.csv
  infer_test.py                # 单模型推理
  utils.py
test_data/                     # 10 条样例，仅本地自测（不进镜像）
```
