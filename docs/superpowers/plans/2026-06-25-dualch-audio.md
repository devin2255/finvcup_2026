# Dual-Channel Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight 2-channel mel-CNN (`StereoActivityEncoder`) alongside the mono Whisper encoder so the audio modality regains speaker-channel separation for BC/I/T, without doubling the Whisper forward.

**Architecture:** New torch-only module `src/audio_stereo.py` holds the locally-testable `StereoActivityEncoder` (per-channel log-mel over the last `tail_sec`, GroupNorm conv stack, global pool). A thin `DualChannelAudioEncoder` wrapper in `multimodal_baseline.py` concatenates `[whisper_emb ; stereo_emb]`; the model builds it only when `audio_encoder.stereo_branch.enabled`. Fusion auto-adapts via `audio_encoder.out_dim`, so its 4-modality structure is untouched.

**Tech Stack:** Python, PyTorch, torchaudio (MelSpectrogram), pytest. Local tests on CPU (torch 1.7.1); real A/B training on the L20 server (torch 2.x).

---

## Environment Reality (read first)

- Local: Windows, no GPU/data, Python 3.8, torch 1.7.1+cpu, transformers 4.5.1.
- `src.models.multimodal_baseline` **cannot be imported locally** (transformers 4.5.1 lacks `WhisperFeatureExtractor`). So `StereoActivityEncoder` lives in its own torch-only module `src/audio_stereo.py` (importable + unit-testable locally); the wrapper + model wiring are parse-checked with `python -m py_compile` and grep only.
- Verified locally: `torchaudio.transforms.MelSpectrogram`, stereo `conv2d`, `torch.cuda.amp.autocast(enabled=False)`, `nn.GroupNorm` all run on torch 1.7.1 CPU.
- Run tests: `python -m pytest tests/ -v` from repo root.
- Branch is `feat/dualch-audio` off baseline `feat/vap-aux-head` (mono Whisper, original single-frame VAP). Independent of `feat/vap-window-pool`.

## File Structure

- Create `src/audio_stereo.py` — torch-only: `_num_groups`, `_no_autocast`, `_slice_tail`, `StereoActivityEncoder`. One responsibility: stereo waveform → per-channel-activity embedding.
- Create `tests/test_audio_stereo.py`.
- Modify `src/models/multimodal_baseline.py` — import `StereoActivityEncoder`; add `DualChannelAudioEncoder`; wrap in `MultimodalTurnTakingModel.__init__` when `stereo_branch.enabled`.
- Create `configs/whisper_qwen0_6b_lmf_dualch.yaml`.

---

## Task 1: tail-slice helper (`_slice_tail`)

**Files:**
- Create: `src/audio_stereo.py`
- Test: `tests/test_audio_stereo.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio_stereo.py
import torch
from src.audio_stereo import _slice_tail


def test_slice_returns_last_tail_samples_when_longer():
    wave = torch.arange(2 * 2 * 100, dtype=torch.float32).reshape(2, 2, 100)
    out = _slice_tail(wave, 30)
    assert out.shape == (2, 2, 30)
    assert torch.equal(out, wave[..., -30:])


def test_slice_returns_full_when_shorter_or_equal():
    wave = torch.randn(2, 2, 20)
    assert torch.equal(_slice_tail(wave, 50), wave)   # shorter than tail
    assert torch.equal(_slice_tail(wave, 20), wave)   # equal to tail


def test_slice_nonpositive_or_none_returns_full():
    wave = torch.randn(1, 2, 10)
    assert torch.equal(_slice_tail(wave, 0), wave)
    assert torch.equal(_slice_tail(wave, None), wave)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_audio_stereo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.audio_stereo'`.

- [ ] **Step 3: Implement `src/audio_stereo.py` (helpers only for now)**

```python
"""轻量双声道活动编码器（torch-only，本地可单测，不引入 transformers）。

Whisper 分支负责语义（单声道）；本模块对**末 tail_sec 秒**的双声道音频做逐声道
log-mel + GroupNorm conv 栈，捕捉声道活动/重叠（利好 BC/I/T）。用 GroupNorm 而非
BatchNorm2d：ensemble 瘦身 checkpoint 只存可训练参数（丢 buffer），BN 的 running
stats 会在推理时丢失；GroupNorm 无 running 统计，训练/推理一致。
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """GroupNorm 组数：取 <=max_groups 且能整除 channels 的最大值。"""
    for g in (max_groups, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def _no_autocast():
    """跨 torch 版本的'关闭 autocast'上下文（保证 mel/stft 走 fp32）。"""
    try:
        return torch.cuda.amp.autocast(enabled=False)
    except Exception:
        return nullcontext()


def _slice_tail(wave: torch.Tensor, tail_samples) -> torch.Tensor:
    """取 [..., T] 沿时间轴的末 tail_samples；T<=tail 或 tail<=0/None 时返回全长。"""
    T = wave.shape[-1]
    if tail_samples is None or tail_samples <= 0 or T <= tail_samples:
        return wave
    return wave[..., -tail_samples:]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_audio_stereo.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/audio_stereo.py tests/test_audio_stereo.py
git commit -m "feat(audio): add _slice_tail helper for stereo branch"
```

---

## Task 2: `StereoActivityEncoder`

**Files:**
- Modify: `src/audio_stereo.py`
- Test: `tests/test_audio_stereo.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_audio_stereo.py`)**

```python
from src.audio_stereo import StereoActivityEncoder


def _enc(**kw):
    # small + cheap defaults for CPU tests
    return StereoActivityEncoder(sample_rate=16000, n_mels=32,
                                 conv_channels=(16, 32, 48), tail_sec=1.0, dropout=0.0, **kw)


def test_output_shape_is_batch_by_out_dim():
    enc = _enc().eval()
    out = enc(torch.randn(2, 2, 32000))   # 2s stereo
    assert out.shape == (2, 48)
    assert enc.out_dim == 48


def test_uses_only_tail_window():
    # tail_sec=1.0 -> last 16000 samples; changing the pre-tail region must not change output
    enc = _enc().eval()
    x = torch.randn(1, 2, 32000)
    x2 = x.clone()
    x2[..., :16000] = torch.randn(1, 2, 16000)   # mutate only the discarded prefix
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x2), atol=1e-5)


def test_short_input_shorter_than_tail_still_works():
    enc = _enc().eval()
    out = enc(torch.randn(2, 2, 8000))    # 0.5s < tail
    assert out.shape == (2, 48)


def test_zeros_input_is_finite():
    enc = _enc().eval()
    with torch.no_grad():
        out = enc(torch.zeros(2, 2, 32000))
    assert out.shape == (2, 48) and torch.isfinite(out).all()


def test_deterministic_in_eval():
    enc = _enc().eval()
    x = torch.randn(2, 2, 24000)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_audio_stereo.py -k "StereoActivity or out_dim or tail_window or short_input or zeros or deterministic" -v`
Expected: FAIL — `ImportError: cannot import name 'StereoActivityEncoder'`.

- [ ] **Step 3: Implement (append to `src/audio_stereo.py`)**

```python
class StereoActivityEncoder(nn.Module):
    """[B, 2, T] -> [B, out_dim]：末 tail_sec 秒逐声道 log-mel + GroupNorm conv 栈。"""

    def __init__(self, sample_rate: int, n_mels: int = 64,
                 conv_channels=(32, 64, 96), tail_sec: float = 6.0, dropout: float = 0.1,
                 n_fft: int = 1024, hop_length: int = 320, win_length: int = 1024):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.tail_samples = int(tail_sec * sample_rate)
        self._mel_cfg = dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self._mel_transform = None
        self.register_buffer("_log_clamp_min", torch.tensor(1e-4), persistent=False)

        c1, c2, c3 = conv_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(2, c1, 3, 1, 1), nn.GroupNorm(_num_groups(c1), c1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, 2, 1), nn.GroupNorm(_num_groups(c2), c2), nn.GELU(),
            nn.Conv2d(c2, c3, 3, 2, 1), nn.GroupNorm(_num_groups(c3), c3), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.out_dim = c3

    def _ensure_mel(self, device: torch.device):
        if self._mel_transform is None:
            import torchaudio
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=self.n_mels, **self._mel_cfg,
            )
        self._mel_transform = self._mel_transform.to(device)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        self._ensure_mel(wave.device)
        wave = _slice_tail(wave, self.tail_samples)
        if wave.shape[1] == 1:
            wave = wave.repeat(1, 2, 1)
        elif wave.shape[1] > 2:
            wave = wave[:, :2]
        mels = []
        for c in range(2):
            with _no_autocast():
                m = self._mel_transform(wave[:, c, :].float())
                m = torch.log(torch.clamp(m, min=float(self._log_clamp_min.item())))
            mels.append(m)
        mel = torch.stack(mels, dim=1)        # [B, 2, n_mels, frames]
        return self.encoder(mel)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_audio_stereo.py -v`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/audio_stereo.py tests/test_audio_stereo.py
git commit -m "feat(audio): add StereoActivityEncoder (tail log-mel + GroupNorm CNN)"
```

---

## Task 3: `DualChannelAudioEncoder` wrapper + model wiring

**Files:**
- Modify: `src/models/multimodal_baseline.py` (import; new class; `MultimodalTurnTakingModel.__init__` audio branch)

Not locally runnable (transformers import blocked). Verify by `python -m py_compile` + grep; `StereoActivityEncoder` behaviour is covered by Task 2.

- [ ] **Step 1: Add import near the top of `src/models/multimodal_baseline.py`**

After the line `from transformers import AutoModel, WhisperFeatureExtractor, WhisperModel` add:

```python

from src.audio_stereo import StereoActivityEncoder
```

- [ ] **Step 2: Add `DualChannelAudioEncoder` class**

Insert immediately before `class MultimodalTurnTakingModel(nn.Module):`:

```python
class DualChannelAudioEncoder(nn.Module):
    """Mono Whisper（语义内容）+ 轻量 stereo CNN（声道活动）拼接成 audio 模态。"""

    def __init__(self, whisper: WhisperAudioEncoder, stereo: StereoActivityEncoder):
        super().__init__()
        self.whisper = whisper
        self.stereo = stereo
        self.out_dim = whisper.out_dim + stereo.out_dim

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # 两个子编码器吃同一个 [B, 2, T]：whisper 内部自行 mono 化，stereo 用双声道。
        return torch.cat([self.whisper(wave), self.stereo(wave)], dim=-1)


```

- [ ] **Step 3: Replace the `audio_type == "whisper"` branch in `MultimodalTurnTakingModel.__init__`**

Replace:

```python
        if audio_type == "whisper":
            self.audio_encoder = WhisperAudioEncoder(
                model_name=cfg["audio_encoder"]["model_name"],
                sample_rate=cfg["sample_rate"],
                proj_dim=int(cfg["audio_encoder"]["proj_dim"]),
                freeze=bool(cfg["audio_encoder"].get("freeze", True)),
                tail_ratio=float(cfg["audio_encoder"].get("tail_ratio", 0.2)),
                unfreeze_layers=int(cfg["audio_encoder"].get("unfreeze_layers", 0)),
            )
        else:
```

With:

```python
        if audio_type == "whisper":
            whisper_enc = WhisperAudioEncoder(
                model_name=cfg["audio_encoder"]["model_name"],
                sample_rate=cfg["sample_rate"],
                proj_dim=int(cfg["audio_encoder"]["proj_dim"]),
                freeze=bool(cfg["audio_encoder"].get("freeze", True)),
                tail_ratio=float(cfg["audio_encoder"].get("tail_ratio", 0.2)),
                unfreeze_layers=int(cfg["audio_encoder"].get("unfreeze_layers", 0)),
            )
            sb_cfg = cfg["audio_encoder"].get("stereo_branch", {}) or {}
            if bool(sb_cfg.get("enabled", False)):
                stereo_enc = StereoActivityEncoder(
                    sample_rate=cfg["sample_rate"],
                    n_mels=int(sb_cfg.get("n_mels", 64)),
                    conv_channels=tuple(sb_cfg.get("conv_channels", [32, 64, 96])),
                    tail_sec=float(sb_cfg.get("tail_sec", 6.0)),
                    dropout=float(sb_cfg.get("dropout", 0.1)),
                )
                self.audio_encoder = DualChannelAudioEncoder(whisper_enc, stereo_enc)
            else:
                self.audio_encoder = whisper_enc
        else:
```

- [ ] **Step 4: Syntax check + confirm refs**

Run:
```bash
python -m py_compile src/models/multimodal_baseline.py && echo OK
grep -n "DualChannelAudioEncoder\|StereoActivityEncoder\|stereo_branch" src/models/multimodal_baseline.py
```
Expected: `OK`, and grep shows the import, the class def, the wiring, and the `stereo_branch` config read.

- [ ] **Step 5: Commit**

```bash
git commit src/models/multimodal_baseline.py -m "feat(audio): wire DualChannelAudioEncoder into model"
```

(Note: `.gitignore` has `models/` which also matches `src/models/`; the file is already tracked, so commit it by path — `git add src/models/...` prints an ignore hint and is a no-op.)

---

## Task 4: Config for clean A/B

**Files:**
- Create: `configs/whisper_qwen0_6b_lmf_dualch.yaml`

- [ ] **Step 1: Copy the baseline config**

```bash
cp configs/whisper_qwen0_6b_lmf_vapfeat.yaml configs/whisper_qwen0_6b_lmf_dualch.yaml
```

- [ ] **Step 2: Edit `configs/whisper_qwen0_6b_lmf_dualch.yaml` — add `stereo_branch` under `audio_encoder`**

Change the `audio_encoder:` block to:

```yaml
audio_encoder:
  type: whisper
  model_name: /mnt/workspace/dorihue/modelscope/whisper-large-v3
  proj_dim: 512
  freeze: true
  unfreeze_layers: 2
  tail_ratio: 0.25
  stereo_branch:
    enabled: true
    n_mels: 64
    conv_channels: [32, 64, 96]
    tail_sec: 6.0
    dropout: 0.1
```

- [ ] **Step 3: Edit the three output paths under `paths:` (lmf_vapfeat -> lmf_dualch)**

```yaml
  output_root: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_dualch
  checkpoints_dir: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_dualch/checkpoints
  logs_dir: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_dualch/logs
```

- [ ] **Step 4: Edit the best-checkpoint name under `train:`**

Change `best_checkpoint_name: best_lmf_vapfeat.pt` to:

```yaml
  best_checkpoint_name: best_lmf_dualch.pt
```

- [ ] **Step 5: Validate YAML and commit**

Run:
```bash
python -c "import yaml; d=yaml.safe_load(open('configs/whisper_qwen0_6b_lmf_dualch.yaml',encoding='utf-8')); sb=d['audio_encoder']['stereo_branch']; assert sb['enabled'] and sb['tail_sec']==6.0 and sb['conv_channels']==[32,64,96]; assert 'lmf_dualch' in d['paths']['output_root']; print('YAML OK')"
git add configs/whisper_qwen0_6b_lmf_dualch.yaml
git commit -m "feat(audio): add lmf_dualch config (stereo_branch, clean A/B dir)"
```
Expected: `YAML OK`.

---

## Task 5: Full local test sweep

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: PASS (the audio-stereo tests; 11 total in `tests/test_audio_stereo.py`).

- [ ] **Step 2: Final review**

Confirm `git log --oneline master..HEAD` shows the spec + per-task commits and `git status --porcelain --untracked-files=no` is empty.

---

## Server smoke (run by user, after merge)

1. Train: `python -m src.train --config configs/whisper_qwen0_6b_lmf_dualch.yaml` for the same budget as the baseline (`max_steps_per_epoch=500`, ~6–8 ep).
2. Compare `outputs/lmf_dualch/logs/eval_epoch_*.json` vs `outputs/lmf_vapfeat/logs`: focus on **Δmacro_roc_auc** and per-class `bc/i/t` `best_f1`/`roc_auc`. C/NA should stay flat or rise slightly.
3. Inference cost sanity: stereo CNN over 6s adds negligible time; the 5-model ensemble must still finish < 60 min (`SUBMIT_README.md`).
