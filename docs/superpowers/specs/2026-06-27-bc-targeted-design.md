# BC 专项提分(短窗注意力池化 + BC 损失解锁) — 设计文档

- 日期:2026-06-27
- 实现基线:`feat/dualch-audio`(stereo 分支)或组合分支 `feat/dualch-vapwin`;A/B 对照取所基于的那个(推荐先基于 `lmf_dualch`)
- 状态:**设计待审,未实现**

## 1. 背景与动机

三发提交(vapfeat 0.638 / vapwin 0.6409 / dualch 0.6396)的 **BC F1 都卡在 ≈0.25**,是 macro 的单点瓶颈(其余类 0.5~0.97)。BC = 听者在说话者讲话中插入的**短暂**反馈("嗯/对")。难点叠加:

1. **稀有**:BC 正例 ~3.65% → 监督信号少。
2. **瞬时**:听者声道里的短 spike,被 dualch `StereoActivityEncoder` 末端的 `AdaptiveAvgPool2d((1,1))`(mel+时间全局平均)**稀释**。这解释了 dualch 涨 T/I 却没动 BC。
3. **窗口标签粗**:"未来 2s 内是否出现 BC" 对短事件钝。

本设计针对 (2) 和 (1):① 把全局平均换成能保留瞬时尖峰的时间池化;② 解开 BC 的损失权重。(3) 留作后续(逐 chunk 密集监督,见 §8)。

## 2. 目标

- 抬高 BC best_f1(主),不显著拉低 I/T(dualch 已涨的部分)。
- 改动可配置、可本地单测、可干净 A/B。

非目标:逐 chunk 密集监督(④,另案)、BC 专用 VAP(③,另案)。

## 3. 详细设计

### 3.1 ① 短窗时间池化 — `src/audio_stereo.py` `StereoActivityEncoder`

当前:`conv(mel) -> [B,C,F',T'] -> AdaptiveAvgPool2d((1,1)) -> [B,C]`。

改为可配置 `time_pool`:
```
StereoActivityEncoder(..., time_pool: str = "attn_max")   # avg | max | attn | attn_max
forward:
  h = conv(mel)                 # [B, C, F', T']
  h = h.mean(dim=2)             # 频率轴平均 -> [B, C, T']（保留时间）
  h = h.transpose(1, 2)         # [B, T', C]
  pooled = _time_pool(h)        # 按 time_pool 聚合时间轴 -> [B, C] 或 [B, 2C]
  return drop(norm(pooled))
```
时间池化算子(对 `[B,T',C]`):
- `avg`:`h.mean(1)` → `[B,C]`(= 旧行为，向后兼容)。
- `max`:`h.max(1).values` → `[B,C]`(瞬时尖峰检测器)。
- `attn`:单查询注意力池化(复用 `vap_pool._AttnPool1d` 同款)→ `[B,C]`。
- `attn_max`(默认):`cat([attn(h), max(h)], -1)` → `[B,2C]`(注意力聚焦 + max 兜底瞬时)。

`out_dim`:`avg/max/attn` = `C`;`attn_max` = `2C`。`LayerNorm`/`Dropout` 维度随 `out_dim`。
**仍用 GroupNorm**(瘦身 checkpoint 不丢 running stats,见 dualch spec)。注意力 query 是可训练参数,进瘦身 checkpoint。

`DualChannelAudioEncoder.out_dim = whisper.out_dim + stereo.out_dim` 自动随 `time_pool` 变(`attn_max` 时 stereo=192)→ 融合层经 `audio_encoder.out_dim` 自动适配,无需改 fusion。

### 3.2 ② BC 损失解锁 — `src/train.py` pos_weight

当前(train.py main 内联):`pw = neg/pos`;`capped_per_label` 时 `pw = min(pw, cap)`,`cap` 对所有标签统一(配置 `pos_weight_cap: 8.0`)。

改:抽出**可单测的纯函数** `compute_pos_weight(y_mat, label_names, cap, per_label_cap)`:
- `per_label_cap`: dict,如 `{bc: 16}`,对指定标签用单独 cap,其余用全局 `cap`。
- 逻辑:`pw_i = min(neg_i/max(1,pos_i), per_label_cap.get(name_i, cap))`。
- train.py 调它替换内联计算。

配置新增:
```yaml
train:
  pos_weight_cap: 8.0
  pos_weight_cap_per_label: { bc: 16.0 }   # BC 原始 ~26，放到 16（其余仍 8）
```

### 3.3 Config

新建 `configs/whisper_qwen0_6b_lmf_bc.yaml`:
- 基于 `lmf_dualch`(隔离 B 的效果;若想叠在组合上则基于 `lmf_dualvap`)。
- `audio_encoder.stereo_branch.time_pool: attn_max`。
- `train.pos_weight_cap_per_label: { bc: 16.0 }`。
- 输出 `outputs/lmf_bc`,`best_checkpoint_name: best_lmf_bc.pt`。
- 与基线只差这两处 → 干净 A/B。

## 4. 数据流(只 stereo 分支变)

`wave[B,2,T]` → 末 tail_sec 双声道 logmel → conv → 频率平均 → **时间 attn⊕max 池化** → `[B,192]` → 与 whisper(512) 拼 → audio 模态。其余不变。

## 5. A/B 评判

`lmf_bc` vs `lmf_dualch`(或 vs `lmf_dualvap`),固定 seed-42 split。**主看 bc_best_f1 / bc_roc_auc**,兼看 i/t 不退、macro 不降。注意:本地是真榜弱代理(见 [[finvcup-score-levers]]),小涨需真榜确认。

## 6. 测试计划(本地,torch-only)

`tests/test_audio_stereo.py` 扩充:
1. `time_pool` 各模式输出维度:`avg/max/attn` → `[B,C]`;`attn_max` → `[B,2C]`;`out_dim` 属性一致。
2. **BC 瞬时敏感性(核心)**:构造两个输入,一个全程低能量、一个在末窗插入**单帧高能量 spike**;`max`/`attn_max` 的输出差异应**显著大于** `avg` 的输出差异(证明新池化保留瞬时尖峰)。
3. tail 切片 / 短输入 / 零输入有限 / eval 确定性(沿用现有)。

`tests/test_pos_weight.py`(新):
4. `compute_pos_weight`:合成 `y_mat`,验证 BC 用 per-label cap(16)、其余用全局 cap(8);pos=0 不除零;无 per_label 时退化为旧行为。

非本地:config + train.py 接线 parse-check。

## 7. 风险与边界

- `attn_max` 改变 stereo out_dim → 模型架构变 → 须重训(不可加载 dualch checkpoint)。预期。
- ② 单独常只换 P/R;① 是主力。BC cap 设太高伤 precision → 先 16,可调 12/20。
- max-pool 对噪声敏感:logmel 已 log 压缩 + GroupNorm,且 tail 只 6s,风险可控;若噪声放大可只用 `attn`。
- 参数增量极小(注意力 query + 维度翻倍的 LayerNorm/Linear 入口),远 <8B。

## 8. 后续(本设计不含)

- ④ 逐 chunk BC 密集监督(未来 25 chunk 逐帧 BC,25× 监督)—— 天花板更高、改动更大,作为"①②不够再上"的大招。
- ③ `vap_bc_ch`(BC 专用 VAP)特征注入。
