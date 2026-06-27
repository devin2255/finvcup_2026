# BC-Targeted Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the stuck BC F1 (~0.25) by (①) replacing the stereo branch's global avg-pool with time attention+max pooling so BC's brief listener-channel spike survives, and (②) unlocking BC's per-label pos_weight cap.

**Architecture:** `StereoActivityEncoder` is refactored to separate the conv stack from a configurable time pooling (`avg`/`max`/`attn`/`attn_max`); a new numpy-only `src/pos_weight.py` holds a testable `compute_pos_weight` with per-label caps. Both new units are unit-testable in the local py3.8/torch-1.7.1 env; train.py/model/config are thin wiring.

**Tech Stack:** PyTorch, torchaudio, NumPy, pytest. Local tests on CPU (torch 1.7.1); real A/B training on the L20 server.

---

## Environment Reality (read first)

- Local: Windows, no GPU/data, py3.8, torch 1.7.1+cpu, transformers 4.5.1.
- Branch `feat/bc-targeted` is off `feat/dualch-audio` (has `src/audio_stereo.py` with the stereo branch; **does NOT have `src/vap_pool.py`** — define the attention pool inline in `audio_stereo.py`, do not import vap_pool).
- `src.models.multimodal_baseline` / `src.train` can't be imported locally (transformers / torch.amp). New logic goes in `src/audio_stereo.py` (torch-only) and `src/pos_weight.py` (numpy-only), both locally testable; wiring is `py_compile` + grep only.
- Backward compat: `time_pool` default = `"avg"` (preserves dualch behavior + existing tests). The `lmf_bc` config opts into `attn_max`.
- Run tests: `python -m pytest tests/ -v` from repo root.

## File Structure

- Modify `src/audio_stereo.py` — add `_AttnTimePool`; refactor `StereoActivityEncoder` to split conv from a configurable `time_pool`.
- Modify `tests/test_audio_stereo.py` — add time_pool shape tests + BC-spike sensitivity.
- Create `src/pos_weight.py` — numpy-only `compute_pos_weight(y_mat, label_names, cap, per_label_cap)`.
- Create `tests/test_pos_weight.py`.
- Modify `src/train.py` — use `compute_pos_weight` + read `pos_weight_cap_per_label`.
- Modify `src/models/multimodal_baseline.py` — pass `time_pool` to `StereoActivityEncoder`.
- Create `configs/whisper_qwen0_6b_lmf_bc.yaml`.

---

## Task 1: configurable time pooling in `StereoActivityEncoder`

**Files:**
- Modify: `src/audio_stereo.py`
- Test: `tests/test_audio_stereo.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_audio_stereo.py`)**

```python
# --- BC-targeted: time_pool ---
import pytest
from src.audio_stereo import StereoActivityEncoder as _SAE


def _enc_tp(time_pool):
    return _SAE(sample_rate=16000, n_mels=32, conv_channels=(16, 32, 48),
               tail_sec=1.0, dropout=0.0, time_pool=time_pool).eval()


@pytest.mark.parametrize("tp,mult", [("avg", 1), ("max", 1), ("attn", 1), ("attn_max", 2)])
def test_time_pool_out_dim(tp, mult):
    enc = _enc_tp(tp)
    assert enc.out_dim == 48 * mult
    out = enc(torch.randn(2, 2, 32000))
    assert out.shape == (2, 48 * mult)


def test_default_time_pool_is_avg_backward_compat():
    enc = _SAE(sample_rate=16000, n_mels=32, conv_channels=(16, 32, 48),
              tail_sec=1.0, dropout=0.0)
    assert enc.out_dim == 48  # unchanged default behavior


def test_max_pool_preserves_brief_spike_more_than_avg():
    # A brief high-energy spike in the tail should move max/attn_max far more than avg.
    torch.manual_seed(0)
    base = torch.full((1, 2, 16000), 0.01)
    spiked = base.clone()
    spiked[0, 1, 8000:8400] = 3.0           # brief loud blip in ch1 (listener), ~25ms
    diffs = {}
    for tp in ("avg", "max", "attn_max"):
        enc = _enc_tp(tp)
        with torch.no_grad():
            diffs[tp] = (enc(spiked) - enc(base)).abs().mean().item()
    # max-based pooling reacts to the brief spike noticeably more than plain avg
    assert diffs["max"] > diffs["avg"] * 1.5
    assert diffs["attn_max"] > diffs["avg"] * 1.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_audio_stereo.py -k "time_pool or brief_spike or backward_compat" -v`
Expected: FAIL — `StereoActivityEncoder.__init__() got an unexpected keyword argument 'time_pool'`.

- [ ] **Step 3: Implement — replace the `StereoActivityEncoder` class in `src/audio_stereo.py`**

Replace the entire existing `class StereoActivityEncoder(nn.Module):` definition with:

```python
class _AttnTimePool(nn.Module):
    """单查询注意力时间池化：[B, T, C] -> [B, C]。"""
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = (self.query * x).sum(dim=-1) * self.scale     # [B, T]
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)        # [B, T, 1]
        return (x * w).sum(dim=1)                              # [B, C]


class StereoActivityEncoder(nn.Module):
    """[B, 2, T] -> [B, out_dim]：末 tail_sec 秒逐声道 log-mel + GroupNorm conv，
    再做可配置的时间池化（avg|max|attn|attn_max）。

    time_pool='attn_max' 用注意力+max 拼接，保留 BC 的瞬时尖峰（max 是瞬时事件检测器）；
    avg=旧行为（向后兼容，默认）。out_dim = C（avg/max/attn）或 2C（attn_max）。
    """

    def __init__(self, sample_rate: int, n_mels: int = 64,
                 conv_channels=(32, 64, 96), tail_sec: float = 6.0, dropout: float = 0.1,
                 time_pool: str = "avg",
                 n_fft: int = 1024, hop_length: int = 320, win_length: int = 1024):
        super().__init__()
        if time_pool not in ("avg", "max", "attn", "attn_max"):
            raise ValueError(f"bad time_pool: {time_pool}")
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.tail_samples = int(tail_sec * sample_rate)
        self.time_pool = time_pool
        self._mel_cfg = dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self._mel_transform = None
        self.register_buffer("_log_clamp_min", torch.tensor(1e-4), persistent=False)

        c1, c2, c3 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(2, c1, 3, 1, 1), nn.GroupNorm(_num_groups(c1), c1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, 2, 1), nn.GroupNorm(_num_groups(c2), c2), nn.GELU(),
            nn.Conv2d(c2, c3, 3, 2, 1), nn.GroupNorm(_num_groups(c3), c3), nn.GELU(),
        )
        self.attn = _AttnTimePool(c3) if time_pool in ("attn", "attn_max") else None
        self.out_dim = c3 * 2 if time_pool == "attn_max" else c3
        self.norm = nn.LayerNorm(self.out_dim)
        self.drop = nn.Dropout(dropout)

    def _ensure_mel(self, device: torch.device):
        if self._mel_transform is None:
            import torchaudio
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=self.n_mels, **self._mel_cfg,
            )
        self._mel_transform = self._mel_transform.to(device)

    def _pool_time(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, C]
        if self.time_pool == "avg":
            return h.mean(dim=1)
        if self.time_pool == "max":
            return h.max(dim=1).values
        if self.time_pool == "attn":
            return self.attn(h)
        return torch.cat([self.attn(h), h.max(dim=1).values], dim=-1)  # attn_max

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
        h = self.conv(mel)                    # [B, C, F', T']
        h = h.mean(dim=2)                     # 频率轴平均 -> [B, C, T']
        h = h.transpose(1, 2)                 # [B, T', C]
        pooled = self._pool_time(h)           # [B, out_dim]
        return self.drop(self.norm(pooled))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_audio_stereo.py -v`
Expected: PASS (existing 8 tests still pass via default `avg`; new time_pool tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/audio_stereo.py tests/test_audio_stereo.py
git commit -m "feat(bc): configurable time_pool (avg/max/attn/attn_max) in StereoActivityEncoder"
```

---

## Task 2: `compute_pos_weight` with per-label caps

**Files:**
- Create: `src/pos_weight.py`
- Test: `tests/test_pos_weight.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pos_weight.py
import numpy as np
from src.pos_weight import compute_pos_weight


def test_global_cap_applied():
    # label0: pos=1,neg=99 -> raw 99 capped to 8; label1: pos=50,neg=50 -> 1.0
    y = np.zeros((100, 2), np.float32); y[:1, 0] = 1; y[:50, 1] = 1
    pw = compute_pos_weight(y, ["bc", "t"], cap=8.0, per_label_cap=None)
    np.testing.assert_allclose(pw, [8.0, 1.0], rtol=1e-5)


def test_per_label_cap_overrides_for_bc():
    y = np.zeros((100, 2), np.float32); y[:1, 0] = 1; y[:1, 1] = 1   # both raw 99
    pw = compute_pos_weight(y, ["bc", "t"], cap=8.0, per_label_cap={"bc": 16.0})
    np.testing.assert_allclose(pw, [16.0, 8.0], rtol=1e-5)   # bc->16, t->8


def test_zero_pos_no_div_by_zero():
    y = np.zeros((10, 2), np.float32); y[:5, 1] = 1          # label0 has no positives
    pw = compute_pos_weight(y, ["bc", "t"], cap=8.0, per_label_cap=None)
    assert np.isfinite(pw).all() and pw[0] == 8.0            # capped, finite


def test_per_label_cap_none_equals_global():
    y = np.zeros((100, 3), np.float32); y[:2, :] = 1
    a = compute_pos_weight(y, ["a", "b", "c"], cap=8.0, per_label_cap=None)
    b = compute_pos_weight(y, ["a", "b", "c"], cap=8.0, per_label_cap={})
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pos_weight.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pos_weight'`.

- [ ] **Step 3: Implement `src/pos_weight.py`**

```python
"""Per-label BCE pos_weight with optional per-label caps (numpy-only, locally testable)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def compute_pos_weight(y_mat: np.ndarray, label_names: Sequence[str],
                       cap: float, per_label_cap: Optional[Dict[str, float]] = None) -> np.ndarray:
    """pos_weight_i = min(neg_i / max(1, pos_i), cap_i)，cap_i 取 per_label_cap[name] 否则全局 cap。

    y_mat: [N, L] 0/1 多标签矩阵；返回 [L] float32。
    """
    y = np.asarray(y_mat, dtype=np.float32)
    per_label_cap = per_label_cap or {}
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    raw = neg / np.maximum(1.0, pos)
    caps = np.array([float(per_label_cap.get(name, cap)) for name in label_names], dtype=np.float32)
    return np.minimum(raw, caps).astype(np.float32)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pos_weight.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pos_weight.py tests/test_pos_weight.py
git commit -m "feat(bc): add compute_pos_weight with per-label caps"
```

---

## Task 3: wire train.py + model

**Files:**
- Modify: `src/train.py` (import + replace inline pos_weight block)
- Modify: `src/models/multimodal_baseline.py` (pass `time_pool` to StereoActivityEncoder)

Not locally runnable. Verify by `py_compile` + grep; the helpers are covered by Tasks 1–2.

- [ ] **Step 1: Add import in `src/train.py`** (after `from src.utils import (...)` block, add a line)

```python
from src.pos_weight import compute_pos_weight
```

- [ ] **Step 2: Replace the inline pos_weight block in `src/train.py`**

Replace:

```python
    if cfg["train"].get("pos_weight_mode", "per_label") in ("per_label", "capped_per_label"):
        y_mat = np.asarray([s.label_vec for s in train_samples], dtype=np.float32)  # [N,5]
        pos = y_mat.sum(axis=0)
        neg = y_mat.shape[0] - pos
        pw = neg / np.maximum(1.0, pos)
        if cfg["train"].get("pos_weight_mode") == "capped_per_label":
            cap = float(cfg["train"].get("pos_weight_cap", 5.0))
            pw = np.minimum(pw, cap)
        pos_weight = torch.tensor(pw, device=device, dtype=torch.float32)
    else:
        pos_weight = torch.ones(len(multi_targets), device=device, dtype=torch.float32)
```

With:

```python
    if cfg["train"].get("pos_weight_mode", "per_label") in ("per_label", "capped_per_label"):
        y_mat = np.asarray([s.label_vec for s in train_samples], dtype=np.float32)  # [N,5]
        if cfg["train"].get("pos_weight_mode") == "capped_per_label":
            cap = float(cfg["train"].get("pos_weight_cap", 5.0))
            per_label_cap = cfg["train"].get("pos_weight_cap_per_label", None)
        else:
            cap = float("inf")
            per_label_cap = None
        pw = compute_pos_weight(y_mat, metric_label_names, cap=cap, per_label_cap=per_label_cap)
        pos_weight = torch.tensor(pw, device=device, dtype=torch.float32)
    else:
        pos_weight = torch.ones(len(multi_targets), device=device, dtype=torch.float32)
```

(Note: `metric_label_names` is the lowercase label list already defined earlier in `main()` as `[x.lower() for x in multi_targets]`; `per_label_cap` keys must match, e.g. `bc`.)

- [ ] **Step 3: Pass `time_pool` in `src/models/multimodal_baseline.py`**

In the `StereoActivityEncoder(...)` construction inside `MultimodalTurnTakingModel.__init__`, after `dropout=float(sb_cfg.get("dropout", 0.1)),` add:

```python
                    time_pool=str(sb_cfg.get("time_pool", "avg")),
```

- [ ] **Step 4: Syntax check + grep**

Run:
```bash
python -m py_compile src/train.py src/models/multimodal_baseline.py && echo OK
grep -n "compute_pos_weight\|pos_weight_cap_per_label" src/train.py
grep -n "time_pool" src/models/multimodal_baseline.py
```
Expected: `OK`; grep shows the import + call in train.py and the `time_pool=` kwarg in the model.

- [ ] **Step 5: Commit**

```bash
git add src/train.py
git commit src/train.py src/models/multimodal_baseline.py -m "feat(bc): wire per-label pos_weight cap + stereo time_pool"
```

(`src/models/` is under a gitignored `models/` glob but tracked — commit by path.)

---

## Task 4: config `lmf_bc`

**Files:**
- Create: `configs/whisper_qwen0_6b_lmf_bc.yaml`

- [ ] **Step 1: Copy the dualch config**

```bash
cp configs/whisper_qwen0_6b_lmf_dualch.yaml configs/whisper_qwen0_6b_lmf_bc.yaml
```

- [ ] **Step 2: Edit `configs/whisper_qwen0_6b_lmf_bc.yaml` — add `time_pool` under `stereo_branch`**

Change the `stereo_branch:` block to:

```yaml
  stereo_branch:
    enabled: true
    n_mels: 64
    conv_channels: [32, 64, 96]
    tail_sec: 6.0
    dropout: 0.1
    time_pool: attn_max
```

- [ ] **Step 3: Add BC per-label cap under `train:`**

Find `pos_weight_cap: 8.0` and add right after it:

```yaml
  pos_weight_cap_per_label:
    bc: 16.0
```

- [ ] **Step 4: Change the three output paths + checkpoint name (lmf_dualch -> lmf_bc)**

```yaml
  output_root: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_bc
  checkpoints_dir: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_bc/checkpoints
  logs_dir: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_bc/logs
```
and
```yaml
  best_checkpoint_name: best_lmf_bc.pt
```

- [ ] **Step 5: Validate + commit**

Run:
```bash
python -c "import yaml; d=yaml.safe_load(open('configs/whisper_qwen0_6b_lmf_bc.yaml',encoding='utf-8')); assert d['audio_encoder']['stereo_branch']['time_pool']=='attn_max'; assert d['train']['pos_weight_cap_per_label']['bc']==16.0; assert 'lmf_bc' in d['paths']['output_root']; print('YAML OK')"
git add configs/whisper_qwen0_6b_lmf_bc.yaml
git commit -m "feat(bc): add lmf_bc config (time_pool=attn_max, bc pos_weight cap 16)"
```
Expected: `YAML OK`.

---

## Task 5: Full local test sweep

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: PASS (test_audio_stereo.py + test_pos_weight.py).

- [ ] **Step 2: Final review**

Confirm `git log --oneline master..HEAD` shows the per-task commits and `git status --porcelain --untracked-files=no` is empty.

---

## Server smoke (run by user, after merge)

1. Train: `python -m src.train --config configs/whisper_qwen0_6b_lmf_bc.yaml` for the same budget as dualch (`max_steps_per_epoch=500`, ~6–9 ep).
2. Compare `outputs/lmf_bc/logs/eval_epoch_*.json` vs `outputs/lmf_dualch/logs`: **primary = bc_best_f1 / bc_roc_auc**; check i/t don't regress, macro doesn't drop.
3. If BC rises locally, submit (lmf_bc, same v2 flow with stereo_branch + single-frame VAP). Per the score-levers lesson, confirm on the real leaderboard.
