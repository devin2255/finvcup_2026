# Qwen2.5-Omni-3B 原生方案（分支 `feat/omni3b`）

与 whisper+qwen 的 LMF 方案**完全独立**：不再拆多模态分支 + 手工特征，而是把
「音频 + ASR 文本 + 历史 chunk 事件」铺进**一段多模态对话**，交给 Omni-3B 的
**thinker** 联合编码，取最后一层 hidden state 池化后接 5 路 sigmoid 分类头。
不做生成解码（结构化多标签 + 逐标签阈值寻优，生成式无法阈值校准且慢）。

## 文件

| 文件 | 作用 |
|------|------|
| `configs/qwen2_5_omni3b_4xL20.yaml` | Omni 训练配置（4×L20，bf16，LoRA，历史 RLE 文本化） |
| `src/data/omni_dataset.py` | 数据渲染 + `OmniCollate`（用 Omni processor 打包对话） |
| `src/models/omni_turntaking.py` | thinker(LoRA) + 池化 + 分类头；Omni API 细节集中在此 |
| `src/train_omni.py` | 训练入口（DDP + bf16 + EMA + ensemble + 阈值，复用 train.py 工具） |
| `scripts/run_train_omni_3b_4xL20.sh` | 4 卡启动脚本（支持 `SMOKE=1` 冒烟） |

样本/标签窗口、focal loss、EMA、阈值寻优、ensemble top-k 瘦身保存、macro_best_f1
指标口径都**与 whisper 方案一致**，便于直接对比验证分数。

## 依赖（服务器先装）

```bash
pip install "peft>=0.11" "accelerate>=0.30"
# transformers 需支持 Qwen2.5-Omni（>=4.52；仓库已是 4.57.x，OK）
```

## 跑之前先改两处

1. `configs/qwen2_5_omni3b_4xL20.yaml` 的 `omni.model_path`：
   你给的目录是 `/mnt/workspace/dorihue/modelscope/Qwen2.5-Omni-3`（疑似被截断），
   **核对真实文件夹名**（一般是 `Qwen2.5-Omni-3B`）后填对。
2. 确认 `paths.*` 训练数据路径与机器一致（沿用了 tuned 配置的路径）。

## 运行

```bash
# 1) 先冒烟跑通管线（强烈建议）：64 样本 / 1 epoch / 5 步，几分钟内验证不报错
SMOKE=1 bash scripts/run_train_omni_3b_4xL20.sh

# 2) 正式训练
bash scripts/run_train_omni_3b_4xL20.sh
```

产物：`outputs/omni3b/{checkpoints/best_omni3b.pt, checkpoints/ensemble_ep*.pt, logs/ensemble_manifest.json}`，
瘦身 checkpoint 只存 LoRA+头，与现有 ensemble/软投票基建兼容。

## 首次跑通时重点核对（不同 transformers 版本可能差异）

这些都在 `src/models/omni_turntaking.py` 里集中处理，并做了 try/except 兜底；
若报错，基本只需改这一个文件：

1. **类名**：`Qwen2_5OmniForConditionalGeneration` / `Qwen2_5OmniProcessor`。
2. **不加载 talker**：`enable_audio_output=False`（不支持会自动回退后 `del talker`）。
3. **processor 调用**：`processor(text=..., audio=..., sampling_rate=16000, padding=True)`——
   若该版本 kwarg 名不同（如 `audios=`），改 `OmniCollate.__call__`。
4. **thinker 前向**：`output_hidden_states=True` + `logits_to_keep=1`（省 lm_head 显存），
   取 `hidden_states[-1]`；字段名不符时改 `_thinker_last_hidden`。

## 显存 / 速度

- 3B + 音频 + 全层反传较重：默认 `batch_size=4/卡 × 4卡 × grad_accum=8 = 有效批 128`，
  bf16 + 梯度检查点。OOM 就调小 `batch_size` 或 `omni.max_text_tokens`，或增大 `grad_accum`。
- 评估只在 rank0 跑，`eval_valid_sample_count=6000`（3B 评估慢，按需调）。

## 待办（看到训练效果后再做）

- `src/infer_omni.py` + docker：训练 OK 后照 `infer_ensemble.py` 的软投票逻辑加一版
  Omni 推理（构建同结构模型 → strict=False 叠加瘦身权重 → 概率平均 + best pt 阈值）。
  当前先聚焦“训练看效果”。
