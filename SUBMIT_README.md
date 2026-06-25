# 第11届信也科技杯 复赛提交说明

多模态话权预测（Turn-Taking）：**Whisper-large-v3（音频）+ Qwen3-0.6B（文本）+ 上下文标签编码器 + VAP 第5模态**，经低秩双线性融合后多标签输出未来 2s 内 5 个事件（C/NA/I/BC/T）是否发生。

- 模型参数量：1.54B + 0.6B = **2.14B < 8B** ✓
- 推理全程离线（`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`），不连任何网络。

本仓库含两种提交镜像：

| 变体 | Dockerfile | 入口 | 特点 |
|---|---|---|---|
| **v1 单模型 + 零填充 VAP** | `Dockerfile` | `run.sh` | 单 checkpoint(ep6) + per-label 阈值；VAP 喂 0；最快、镜像最小 |
| **v2 ensemble + 真 VAP** | `Dockerfile.ensemble_vap` | `run.ensemble_vap.sh`（镜像内 = `run.sh`） | 5 个 checkpoint 多数投票；容器内现场预计算 VAP（双 Python 环境） |

## 目录结构（镜像内 /app）

```
/app
├── run.sh                              # 推理入口
├── train.sh                            # 训练入口（阶段二复现用）
├── README.md                           # 本文件
├── configs/submit_single_vapfeat.yaml  # 提交配置（路径已指向 /app 与 /xydata）
├── src/                                # 训练 + 推理代码
├── models/
│   ├── whisper-large-v3/               # 仅保留 model.safetensors + 分词/配置
│   └── Qwen3-0.6B/
├── ckpt/ensemble_ep6.pt                # 单模型 checkpoint（valid macro-F1 最优档）
├── thresholds/best_thresholds.json     # 每标签最优阈值
└── submit/submit.csv                   # 运行后产出
```

## 推理（run.sh）

私有测试集以**只读**方式挂载在 `/xydata`，结构为 `audio/ context/ text/`；
结果写入 `/app/submit/submit.csv`。

```bash
# 容器内直接执行
bash run.sh
```

等价命令：

```bash
python -m src.infer_test \
  --config configs/submit_single_vapfeat.yaml \
  --checkpoint ckpt/ensemble_ep6.pt \
  --threshold_file thresholds/best_thresholds.json \
  --test_root /xydata \
  --output_csv /app/submit/submit.csv
```

输出 CSV header：`segment_id,c,na,i,bc,t`（英文逗号分隔，列顺序与官方一致）。

## 本地自测

```bash
docker run --rm -it --gpus all \
  -v /path/to/test_data:/xydata:ro \
  -v /path/to/output_dir:/app/submit \
  finvcup-infer:v1 bash
# 容器内：
bash run.sh
# 检查 ./submit/submit.csv
```

## 训练（train.sh，复赛阶段二复现用）

1. 把整通对话训练集挂载到 `configs/submit_single_vapfeat.yaml` 的
   `paths.train_*_dir`（默认 `/app/data/train/{audio,text,labels}`）。
2. 运行：

   ```bash
   bash train.sh 4   # 参数为 GPU 数量，默认 1
   ```

### 关于 VAP 第5模态特征（重要）

本单模型训练时引入了 VAP（vap_mc_ch_kyoto）逐帧话轮先验作为第 5 模态特征。

- **推理阶段（run.sh）**：测试集不计算 VAP，模型对该模态喂零向量（架构对齐、不影响运行）。
- **完整复现训练**：需先用 MaAI + CPC 权重对训练集离线预计算 VAP 缓存：

  ```bash
  python -m src.precompute_vap \
      --config configs/submit_single_vapfeat.yaml \
      --maai_dir ./MaAI --lang ch_kyoto --frame_rate 10 --context_sec 20 \
      --device cuda --cpc_model /path/to/60k_epoch4-d0f474de.pt \
      --out_dir /app/.cache/vap_ch_kyoto
  ```

  未提供缓存时训练仍可运行（VAP 退化为零向量）。

## v2 镜像构建/运行（ensemble + 真 VAP）

构建前先把以下资产放好：

```powershell
# 1) 主权重 + 5 个 checkpoint + CPC 权重 + VAP 权重 全部 staging 到仓库根
powershell -File scripts\download_vap_weight.ps1          # 下 vap_mc_ch_kyoto 权重
powershell -File scripts\stage_submission_ensemble.ps1   # 把所有资产复制到 models\ ckpt\

# 2) 构建 v2 镜像
docker build -f Dockerfile.ensemble_vap -t finvcup-infer:v2-ens .

# 3) 本地冒烟
docker run --rm -it --gpus all `
  -v D:\path\to\test_data:/xydata:ro `
  -v D:\path\to\out:/app/submit `
  finvcup-infer:v2-ens bash
#   容器内: bash run.sh
```

v2 内部为两阶段：
- 阶段 A：`/opt/maai-env`（torch 2.6 + MaAI）对 `/xydata/audio/*.wav` 预计算 18 维 VAP 特征 → `/app/.cache/vap_test/<seg>.npy`
- 阶段 B：主环境 5 模型 ensemble 投票，每个模型用各自 best thresholds，写 `/app/submit/submit.csv`

## 注意事项

- 基础镜像 CUDA 12.4（torch 2.5.1 / python 3.10）；构建时仅升级 `transformers>=4.51`
  （Qwen3 需要）并安装 `pynvml/requests`，不重装主环境 torch。
- v2 多一个 `/opt/maai-env` venv 装 torch 2.6.0+cu124 + MaAI（`--no-deps`，
  绕开它 pyproject 里 `transformers==5.5.3` 这种存在性可疑的 pin）。
- 测试音频为 5–30s 不定长；`context` 上下文标签序列长度 (0,30] 动态，
  推理时统一归一化到 375 chunk（不足前补 NA=4）。
- 推理超时限制 60 分钟内。v2 经验：VAP 预计算耗时 ≈ 测试集音频总时长 / 实时倍率；
  5 模型每个推理 ≈ 8-12 分钟，时间紧时可在 `run.sh` 给 `--topk 3` 削成 top-3 投票。
