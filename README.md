# 第11届FinvCup 对话轮次预测（Turn-Taking）Baseline代码

Baseline代码实现「过去 30s 音频 + ASR 文本 + 历史标签序列 → 预测未来窗口内话权相关事件」的多模态模型。支持：

- **分类**：未来 2s（默认 25×80ms chunk）内是否出现 `C` / `NA`  / `T` / `BC` / `I`（与 `positive_ids` 一致）。
- **多标签**：对 `labels.multi_targets` 中每一类分别预测是否在未来窗口内出现（sigmoid + BCE）。
本 baseline 固化为 **event-level 多标签**：预测未来 2s（默认 25×80ms chunk）内 5 个标签是否出现（sigmoid + BCE）。

因果约束：仅使用配置中的 `context_chunks`（默认 375×80ms=30s）作为上下文，不读取未来音频或未来标签。


## 环境要求

- Linux + NVIDIA GPU（训练默认 4 卡 DDP，可改）
- Python 3.10 推荐
- PyTorch / torchaudio / transformers 需与 CUDA 版本匹配（见下）

---

## 安装说明

### 1. Conda 环境

```bash
conda create -n finvcup python=3.10 -y
conda activate finvcup
```

### 2. 安装 PyTorch（按你机器的 CUDA 版本选择）

请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装带 CUDA 的 `torch` / `torchaudio`，例如：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装其余依赖

```bash
cd /path/to/finvcup_11th_baseline
pip install -r requirements.txt
```

`requirements.txt` 含：`numpy`、`scipy`、`scikit-learn`、`pyyaml`、`tqdm`、`transformers<5`；具体 `torch`/`torchaudio`根据实际情况进行安装，避免与本地 CUDA 冲突。

### 4. 缓存与镜像（推荐）

根据实际情况酌情修改模型下载地址，可用镜像加速下载：

```bash
export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export HF_ENDPOINT=https://hf-mirror.com
```

### 5. 运行时代码路径
修改yaml配置文件中的路径为真实训练数据路径，在项目根目录执行，保证 `python -m src.train` 可解析包 `src`：

```bash
cd /path/to/finvcup_11th_baseline
```

---

## 数据准备

### 训练集（整通对话）

| 路径 | 说明 |
|------|------|
| `train/audio/<conv_id>.wav` | 双声道整段音频 |
| `train/text/<conv_id>.json` | 对应 ASR |
| `train/labels/<conv_id>.npy` | 逐 chunk 标签，`0~4` 表示 `C/T/BC/I/NA` |

### 测试集（30s 切片）

测试数据根目录在 **推理时通过参数显式传入**（即 `--test_root /path/to/test_data`）。常见子目录：

| 路径 | 说明 |
|------|------|
| `test/audio/<segment_id>.wav` | 30s 切片 |
| `test/text/<segment_id>.json` | ASR |
| `test/context/<segment_id>.npy` | 上下文标签序列 |
---

## 配置说明（YAML）

根据具体情况自行修改以下字段：

- `chunk_ms`、`context_chunks`、`target_chunks`、`stride`、`sample_rate`：时间窗与采样。
- `paths.*`：数据目录、`output_root`、`checkpoints_dir`、`logs_dir`、`cache_root`。（本 baseline 不在 config 中保存 `test_root`）
- `labels`：`C/T/BC/I/NA` 的 id、`positive_ids`（二分类正类）、`multi_targets`（多标签头输出顺序）。
- `split.valid_ratio`、`split.by_conversation`：验证比例与是否按会话划分。
- `audio_encoder.type`：`cnn` 或 `whisper`（Whisper 时需 `model_name`、`proj_dim`、`freeze` 等）。
- `text_encoder.model_name`、`max_length`、`freeze_backbone`。
- `train`：`multi_label`、`epochs`、`batch_size`、`save_metric`（如 `roc_auc`）、`early_stop_patience`、`--resume` 等。
- `env`: 具体缓存路径，可选
示例：

- baseline 配置：`configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml`

---

## 训练


## Baseline

### 1. 训练（train/valid）

```bash
bash scripts/run_train.sh configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml 4
```
根据具体环境，GPU数量，修改脚本参数，训练结束后在 `paths.checkpoints_dir` 下得到 `best.pt`（由 `train.save_metric` 选择）。

### 2. 测试集推理（仅输出 pred.csv）

```bash
bash scripts/run_infer.sh /path/to/best.pt /path/to/pred_test1.csv /path/to/test configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml
```

`pred_test1.csv` 列：`segment_id` + `labels.multi_targets`（小写）对应的 0/1 预测列。格式参考repo中 pred_test1.csv即可

/path/to/best.pt指训练好的模型checkpoint，/path/to/pred_test1.csv指测试集结果输出路径，/path/to/test指测试集数据路径


## TensorBoard

```bash
tensorboard --logdir /path/to/outputs/logs/tb --host 0.0.0.0 --port 6006
```

多标签训练时，常见标量包括：`valid/macro_f1`、`valid/{label}_f1` 等。

---

## 仓库结构

```text
configs/          # 实验配置
scripts/          # run_train.sh, run_infer.sh
src/
  data/dataset.py
  models/multimodal_baseline.py
  train.py        # 训练
  eval.py         # 离线评估
  infer_test.py   # 测试集推理：仅导出 pred.csv
  utils.py        # 指标、配置、分布式工具
train/            # 训练数据（需自行下载解压）
test/       # 测试数据（需自行下载解压）
pred_test1.csv     # 参考提交结果格式
```
