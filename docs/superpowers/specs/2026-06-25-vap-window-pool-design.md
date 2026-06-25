# VAP 窗口学习式池化(第 5 模态升级) — 设计文档

- 日期:2026-06-25
- 分支:`feat/vap-window-pool`
- 基线:`outputs/lmf_vapfeat`(config `configs/whisper_qwen0_6b_lmf_vapfeat.yaml`)

## 1. 背景与动机

赛题为多方言中文对话的轮次交互建模(Turn-Taking),评测为**窗口级 5 类事件 Macro-F1**(未来 2s 内 C/T/BC/I/NA 各自"是否出现"),见 `结果评估说明.txt`。

当前最好一轮(`outputs/lmf_vapfeat/logs/eval_epoch_6.json`)的各类 best-F1:

| 类别 | best-F1 |
|---|---|
| C 延续 | 0.968 |
| NA 静音 | 0.815 |
| T 话权转移 | 0.593 |
| I 打断 | 0.500 |
| BC 附和 | 0.254 |
| **Macro** | **0.626** |

Macro-F1 等权平均,分数几乎全部损失在 BC/I/T。

VAP(`vap_mc_ch_kyoto`)预训练话轮先验本应强预测 BC/turn-shift,但当前作为"第 5 模态"**只取了边界处单帧 18 维**(`src/data/dataset.py` 中 `arr[fr]`),丢弃了整段逐帧轨迹。一个 BC 触发往往是听者在窗口内的**短暂发声 spike**,单帧或均值都会漏掉。

## 2. 目标

把 VAP 先验从「边界单帧 `[18]`」升级为「边界前 N=20 帧轨迹 `[20,18]`」,在模型内用 **conv1d + 注意力池化**自学时序权重。预期收益集中在 BC/I/T;C/NA 应基本不变。

非目标(本次不做):双声道 Whisper(#1)、阈值独立验证集(#3)、外部数据预训练。这些另议。

## 3. 关键约束

- **训练缓存**(`src/precompute_vap.py`)是整段 `[F,18]`,训练侧取窗口零额外计算,**缓存不用重算**。
- **测试缓存**(`src/precompute_vap_test.py`)目前只存最后 1 帧 `(18,)`,需改为存最后 N 帧 `(N,18)`;它本就逐帧流式跑完整段,只多保留几帧,**计算量不变**,仍在 60 分钟硬限内。
- **复赛上下文动态** `(0,30]s`,短段可能不足 N 帧,需 pad。
- **环境**:仓库在 Windows(`D:\`),无 GPU/数据;训练在 L20 服务器(config 路径 `/mnt/workspace/dorihue/...`)。本机仅做单测/形状校验,真正 A/B 训练在服务器跑。

## 4. 详细设计

### 4.1 Dataset 取窗规则(训练/测试共享) — `src/data/dataset.py`

新增模块级函数:

```
_extract_vap_window(arr, fr, N) -> np.ndarray[N, 18]
```

- 取 `arr[max(0, fr-N+1) : fr+1]`(含边界帧 `fr`)。
- 不足 N 帧时**左侧复制最早一帧**补齐(replicate pad),保持时序方向(末帧=边界)。
- `arr is None` 或空 → 返回 `zeros[N,18]`。
- `fr` 入参前已 clamp 到 `[0, F-1]`。

`TurnTakingTrainDataset.__getitem__`:
- `fr = round(end_idx * chunk_ms * vap_frame_rate / 1000)`,clamp 到 `[0, F-1]`。
- `out["vap_feat"] = torch.from_numpy(_extract_vap_window(arr, fr, N))` → `[N,18]`。
- 无 `vap_feat_dir` → 喂 `zeros[N,18]`(保持架构对齐)。

`TurnTakingTestDataset.__getitem__`:
- cache 现为 `(N,18)`;读出后过 `_extract_vap_window`(以 `fr=len-1` 取末 N 帧,统一 pad 规则)。
- 形状鲁棒:旧 `(18,)` 视为单帧 → tile/pad 到 `[N,18]`;任意 `(M,18)` 取末 N、pad。
- 缺失/不存在 → `zeros[N,18]`。

窗口长 `N` 由 config `vap_feat.window`(默认 20)驱动,dataset 构造函数新增 `vap_window` 参数。

### 4.2 Collate — `src/data/dataset.py`

零改动:现有 `torch.stack([b["vap_feat"] ...])` 自动把每个 `[N,18]` 堆成 `[B,N,18]`。

### 4.3 Model — `src/models/multimodal_baseline.py` `use_vap_feat` 分支

- 旧:`vap_feat_proj = Linear(18 → h)`,输入 `[B,18]`。
- 新:`vap_feat_encoder`:
  - `Conv1d(18 → c1=64, k=3, pad=1) → GELU → Conv1d(64 → h, k=3, pad=1) → GELU`,输入 `[B,18,N]`(对 `[B,N,18]` 做 `transpose(1,2)`)。
  - 复用现有 `AttentionPooling(h)` over `[B,N,h]`(把 conv 输出 `transpose` 回 `[B,N,h]`)→ `[B,h]`。
  - `h = self.fusion.out_dim`(=320),使 `vap_feat_merge(Linear(h*2 → h))` 不变。
- forward:`vap_feat` 现为 `[B,N,18]`;`None` → `zeros(B, N, 18)`;`v = vap_feat_encoder(vap_feat)`;其余(`vap_feat_merge`)不变。
- config:`vap_feat.window: 20`、`vap_feat.feat_dim: 18`(每帧)、`vap_feat.conv_channels: [64]`(可选,默认 64)。

### 4.4 Test 预计算 — `src/precompute_vap_test.py`

- `vap_last_frame_feature(...)` → `vap_last_n_frames(maai, audio2, frame_samples, N)`:
  - 用 `collections.deque(maxlen=N)` 累积逐帧 `_flat_result`,跑完取 `list(deque)` → `[M,18]`(M≤N)。
  - 不足 N 帧 → 左 replicate-pad(空 → `zeros[N,18]`)。
  - 输出 `<seg>.npy` shape `(N,18)`,dtype float32。
- `precompute_vap.py`(训练缓存)**不动**(已是整段 `[F,18]`)。
- 新增 CLI `--window`(默认 20),与训练 config `vap_feat.window` 对齐。

### 4.5 Config — 新建 `configs/whisper_qwen0_6b_lmf_vapwin.yaml`

- 从 `whisper_qwen0_6b_lmf_vapfeat.yaml` 复制。
- `vap_feat.window: 20`、`vap_feat.conv_channels: [64]`。
- `output_root: .../outputs/lmf_vapwin`、`checkpoints_dir`、`logs_dir` 全部独立,保证干净 A/B。

## 5. 一致性保证(train == test)

- 取窗 + pad 规则集中在唯一函数 `_extract_vap_window`,两个 dataset 都调它。
- 池化是模型内同一模块 → train/test 必然一致。

## 6. A/B 评判(本地为准,无榜分)

- split 固定(`seed=42`,`by_conversation`),baseline=`lmf_vapfeat` vs 新版=`lmf_vapwin`,只差 VAP 分支。
- 主指标:**Δmacro_roc_auc**(阈值无关,绕开 30k 阈值虚高)+ 同口径 **Δmacro_best_f1**(两边都用 oracle 阈值)。
- 逐类盯 **bc / i / t** 的 `best_f1` 与 `roc_auc`;C/NA 应基本持平。
- 注:当前 `eval_epoch_*.json` 与 `valid_epoch_*.json` 逐字节相同(阈值在被评分的同一 30k 上调出),故绝对 best-F1 偏乐观;A/B 用相对 delta + AUC 判断。

## 7. 测试计划(本机可执行)

单元/形状测试(pytest):

1. `_extract_vap_window`:
   - 正常窗:`F=100, fr=50, N=20` → 返回 `arr[31:51]`,shape `(20,18)`。
   - 近起点 replicate-pad:`fr=5, N=20` → 左侧用 `arr[0]` 复制补 15 帧。
   - 空/None arr → `zeros(20,18)`。
   - **train 路径与 test 路径对同一帧序列返回相同 `[N,18]`**(一致性回归)。
2. 模型 vap 分支形状:构造 `[B=4, N=20, 18]` 过模型,断言输出维度与无-VAP 路径一致;`vap_feat=None` → 内部 `zeros` 不崩。
3. `vap_last_n_frames`:用合成 maai stub(可控逐帧 result)断言保留最后 N 帧、不足时 pad 正确、空输入→零。

服务器 A/B(用户执行):按 `vapwin` config 跑与基线同等步数(`max_steps_per_epoch=500`,~6–8 epoch),对比第 6 节指标。

## 8. 风险与边界

- 旧 test 缓存 `(18,)` 与新模型不兼容 → 必须重跑 `precompute_vap_test`(已在提交流程内,预算内)。dataset 形状鲁棒读可避免崩,但语义上需重算才有窗口信息。
- N=20 时训练 `fr≈300 ≫ 20`,几乎不触发 pad;pad 主要在复赛 `(0,30]s` 短段。
- 学习式池化新增参数极小(~两个 conv + 一个 attention query,远 <1% 总可训练参数),不显著改变显存/吞吐。
