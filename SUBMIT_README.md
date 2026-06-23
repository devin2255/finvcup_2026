# 第11届信也科技杯 复赛提交说明（单模型）

多模态话权预测（Turn-Taking）单模型：**Whisper-large-v3（音频）+ Qwen3-0.6B（文本）+ 上下文标签编码器**，
经低秩双线性融合后多标签输出未来 2s 内 5 个事件（C/NA/I/BC/T）是否发生。

- 模型参数量：1.54B + 0.6B = **2.14B < 8B** ✓
- 推理全程离线（`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`），不连任何网络。

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

## 注意事项

- 基础镜像 CUDA 12.4（torch 2.5.1 / python 3.10）；构建时仅升级 `transformers>=4.51`
  （Qwen3 需要）并安装 `pynvml`，不重装 torch。
- 测试音频为 5–30s 不定长；`context` 上下文标签序列长度 (0,30] 动态，
  推理时统一归一化到 375 chunk（不足前补 NA=4）。
- 推理超时限制 60 分钟内。
