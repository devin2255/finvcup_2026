# 双声道音频分支（Whisper-mono + 轻量 stereo CNN） — 设计文档

- 日期:2026-06-25
- 分支:`feat/dualch-audio`(从基线 `feat/vap-aux-head` 开,与 `feat/vap-window-pool` 独立)
- 基线:`outputs/lmf_vapfeat`(config `configs/whisper_qwen0_6b_lmf_vapfeat.yaml`)

## 1. 背景与动机

赛题 Macro-F1 几乎全部损失在 **BC(0.254)/I(0.500)**(见 `outputs/lmf_vapfeat/logs/eval_epoch_6.json`)。
根因:`src/models/multimodal_baseline.py:150` 的 `mono = wave.mean(dim=1)` 把双声道平均成单声道,
抹掉了说话人分离信息。BC(听者在说话者讲话时插入短反馈)、I(双声道重叠)、T(话权从 A 转到 B)
都依赖**逐声道**的语音活动,而 C/NA 不依赖 —— 这解释了为什么 C(0.968)/NA(0.815)很好、BC/I 很差。

## 2. 目标

Whisper 仍单声道(承担语义内容,主要利好 T),**新增**一个轻量 2-声道 mel-CNN 捕捉
**声道活动/重叠**(利好 BC/I),两者拼接成 audio 模态。计算几乎不增,保住 5 模型 ensemble 与
60 分钟推理预算(`SUBMIT_README.md`:每模型 8–12 min,Whisper-large-v3 编码器固定 30s 输入,
翻倍会击穿预算 —— 故选轻量分支而非真双-Whisper)。

非目标(本次不做):真双-Whisper(成本×2)、声道交换增广(标签相对"说话方 A"定义,交换会失效)、
改 collate padding。

## 3. 关键约束

- **推理 60 分钟硬限**;Whisper 编码器架构固定 30s(1500 帧位置编码,不能缩短省算)。
  stereo CNN 只跑末 `tail_sec`(默认 6s),相对 Whisper 30s 可忽略。
- **参数预算 < 8B**:stereo CNN ~+0.1M,无碍。
- **环境**:本机 Windows(py3.8 / torch1.7.1 / 无 GPU/数据),训练在 L20 服务器。
  本机仅能单测 torch-only 模块(见下"测试");重模块(transformers)只 `py_compile` + grep 校验。
- 已验证本机:`torchaudio.transforms.MelSpectrogram` 与 stereo `conv2d([B,2,mel,T])` 可跑。

## 4. 详细设计

### 4.1 `StereoActivityEncoder`（新,`src/audio_stereo.py`,torch-only,本地可单测）

```
StereoActivityEncoder(sample_rate, n_mels=64, conv_channels=[32,64,96],
                      tail_sec=6.0, dropout=0.1)
forward(wave[B, 2, T]) -> [B, out_dim]   # out_dim = conv_channels[-1]
```

- 取末 `tail_samples = int(tail_sec * sample_rate)`:`wave = wave[:, :, -tail_samples:]`
  (`T < tail_samples` 时取全长)。对齐 Whisper 分支"只看近端"。
- 逐声道 `torchaudio.transforms.MelSpectrogram`(lazy 构造,n_fft=1024/hop=320/win=1024)→
  `log(clamp(mel, 1e-4))`,stack 成 `[B, 2, n_mels, frames]`。
- conv2d 栈(结构类似现有 `AudioEncoder`,但**用 `GroupNorm` 替代 `BatchNorm2d`**,理由见 §8):
  Conv2d(2→c1)/GN/GELU → Conv2d(c1→c2,stride2)/GN/GELU → Conv2d(c2→c3,stride2)/GN/GELU →
  AdaptiveAvgPool2d((1,1)) → Flatten → Dropout → `[B, c3]`。
  GroupNorm 组数取 `min(8, C)` 且需整除 C(c1=32,c2=64,c3=96 均可被 8 整除)。
- 第一层 conv 输入通道=2 → 天然看到跨声道(重叠/差异)模式。
- mel 计算在 `torch.amp.autocast(enabled=False)` 内(与现有 AudioEncoder 一致,避免半精度 stft 问题)。

### 4.2 `DualChannelAudioEncoder`（新 wrapper,`multimodal_baseline.py`,非本地可测）

```
DualChannelAudioEncoder(whisper: WhisperAudioEncoder, stereo: StereoActivityEncoder)
forward(wave[B,2,T]) -> cat([whisper(wave), stereo(wave)], dim=-1)   # [B, 512+96]
out_dim = whisper.out_dim + stereo.out_dim
```

- 两个子编码器吃同一个 `[B,2,T]`(whisper 内部自己 mono 化,stereo 用双声道)。

### 4.3 Model `__init__`（`MultimodalTurnTakingModel`）

- audio_type == "whisper" 且 `audio_encoder.stereo_branch.enabled` 为真时:
  先建 `WhisperAudioEncoder`(同现状),再建 `StereoActivityEncoder`,用 `DualChannelAudioEncoder` 包起来。
- 否则保持现状(纯 WhisperAudioEncoder 或 CNN AudioEncoder)。
- `self.fusion` 用 `self.audio_encoder.out_dim` 构造 → **自动适配 608 维,fusion 4-模态结构不动**。

### 4.4 Config `configs/whisper_qwen0_6b_lmf_dualch.yaml`

- 复制 `whisper_qwen0_6b_lmf_vapfeat.yaml`(本分支基线)。
- `audio_encoder` 增加:
  ```yaml
  stereo_branch:
    enabled: true
    n_mels: 64
    conv_channels: [32, 64, 96]
    tail_sec: 6.0
    dropout: 0.1
  ```
- `paths.output_root/checkpoints_dir/logs_dir` → `outputs/lmf_dualch`。
- `train.best_checkpoint_name: best_lmf_dualch.pt`。
- 其余(含 `vap_feat` 单帧版)与 vapfeat 完全一致 → 与 `lmf_vapfeat` 只差 stereo 分支,干净 A/B。

## 5. 数据流

`wave[B,2,T]`(collate 后) → `DualChannelAudioEncoder`:
  - `WhisperAudioEncoder`:mono → 30s logmel(CPU 特征抽取)→ 冻结编码器(末2层可训)→ tail-attn-pool → proj512
  - `StereoActivityEncoder`:末6s 双声道 → 逐声道 logmel → stereo conv2d → 全局池化 → 96
  - concat → `[B,608]` → 进 `MultimodalFusion`(audio 模态),其余模态不变。

## 6. A/B 评判（本地为准,无榜分）

- split 固定(`seed=42`,`by_conversation`),baseline=`lmf_vapfeat` vs 新版=`lmf_dualch`,只差 stereo 分支。
- 主指标:**Δmacro_roc_auc**(阈值无关)+ 逐类 **bc/i/t** 的 `best_f1`/`roc_auc`;C/NA 应基本持平或略升。

## 7. 测试计划

本地单测(pytest,torch-only):

1. `StereoActivityEncoder` 形状:`[4,2,96000]` → `[4, 96]`(默认 conv_channels)。
2. tail 切片:`T > tail_samples`(如 10s @16k=160000)→ 内部只用末 96000;`T < tail_samples`(如 3s)→ 用全长;两种都返回 `[B, out_dim]`。可用一个可注入的 `_slice_tail` 纯函数单测切片逻辑(长/短/相等)。
3. eval 确定性:同输入两次前向 `allclose`。
4. 零输入有限:`zeros(2,2,96000)` → 输出 `isfinite` 全真,shape 正确。

非本地(parse-check + grep):`DualChannelAudioEncoder` wrapper、model `__init__` 分支、config。

服务器 smoke(用户执行):`python -m src.train --config configs/whisper_qwen0_6b_lmf_dualch.yaml`
跑与基线同步数(`max_steps_per_epoch=500`,~6–8 ep),对比第 6 节指标。

## 8. 风险与边界

- 训练 `dynamic_context` 短样本经 collate 右补零,stereo 的 tail 可能含零 —— 与现有 Whisper-tail
  既有行为一致,非新 bug;eval/test 均满长 30s,tail 是真实近端音频,评分场景正确。
- `StereoActivityEncoder` 用 `BatchNorm2d`:与现有 `AudioEncoder` 一致;EMA/瘦身 checkpoint 只存
  可训练参数 —— BN 的 running stats 是 buffer 不在可训练参数里。需确认:本分支若用 ensemble 瘦身
  保存(`_trainable_state_dict` 排除 buffer),推理重建时 BN running stats 会是初始值(eval 模式下
  影响输出)。**缓解**:stereo 分支用 `nn.GroupNorm` 或 `nn.LayerNorm` 替代 `BatchNorm2d`,
  避免 buffer 丢失问题(现有 CNN AudioEncoder 在 whisper 配置下未被使用,故其 BN 问题未暴露)。
  → **本设计采用 GroupNorm(num_groups=min(8,C), C)**,无 running-stats buffer,瘦身/推理一致。
- stereo CNN 增加的训练显存/算力极小;推理 +6s mel-CNN,远在 60min 预算内。
