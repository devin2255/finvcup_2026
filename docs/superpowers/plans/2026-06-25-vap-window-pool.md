# VAP Window Learned-Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the VAP 5th-modality from a single boundary frame `[18]` to an `[N=20, 18]` trajectory pooled inside the model with conv1d + attention, to lift the BC/I/T classes that drag down macro-F1.

**Architecture:** Two new dependency-light modules hold all new logic so they are unit-testable in the local py3.8 / torch-1.7.1 env: `src/vap_window.py` (numpy: window extraction + last-N-frame streaming) and `src/vap_pool.py` (torch: `VapWindowEncoder`). The heavy modules (`dataset.py`, `multimodal_baseline.py`, `precompute_vap_test.py`, `train.py`, `infer_ensemble.py`) only get thin wiring that calls these tested helpers. The encoder is length-agnostic, so window size lives entirely in the data layer + config.

**Tech Stack:** Python, PyTorch, NumPy, pytest. Local tests run on CPU (torch 1.7.1); real A/B training runs on the L20 server (torch 2.x).

---

## Environment Reality (read first)

- Local repo is Windows, **no GPU/data**, Python 3.8, torch 1.7.1+cpu, transformers 4.5.1.
- `src.data.dataset` and `src.models.multimodal_baseline` **cannot be imported locally** (py3.8 lowercase-generic annotation + transformers lacks `WhisperFeatureExtractor`). That is why new logic goes into `src/vap_window.py` and `src/vap_pool.py`, whose package path (`src/__init__.py` is empty) lets them import locally.
- Run tests with: `python -m pytest tests/ -v` from the repo root (puts repo root on `sys.path`).
- Tasks 5–9 (wiring/config) are **not locally runnable**; verify them by code review + a server smoke after merge. Their correctness is carried by the unit-tested helpers from Tasks 1–4.

## File Structure

- Create `src/vap_window.py` — numpy/collections only: `_flat_result`, `_extract_vap_window`, `vap_last_n_frames`. One responsibility: turn raw VAP frames into a fixed `[N,18]` window.
- Create `src/vap_pool.py` — torch only: `_AttnPool1d`, `VapWindowEncoder`. One responsibility: encode `[B,N,18] -> [B,hidden]`.
- Create `tests/test_vap_window.py`, `tests/test_vap_pool.py`.
- Modify `src/data/dataset.py` — call `_extract_vap_window`; add `vap_window` ctor param to both datasets.
- Modify `src/models/multimodal_baseline.py` — swap the `vap_feat_proj` Linear for `VapWindowEncoder`; store `self.vap_window`; reshape zeros in forward.
- Modify `src/precompute_vap_test.py` — save last `N` frames `(N,18)` via `vap_last_n_frames`; add `--window`.
- Modify `src/train.py` — read `vap_feat.window`; pass `vap_window=` to train+valid datasets.
- Modify `src/infer_ensemble.py` — pass `vap_window=` to the test dataset.
- Create `configs/whisper_qwen0_6b_lmf_vapwin.yaml` — clean A/B output dir + `vap_feat.window: 20`.

---

## Task 1: VAP window extraction (`_extract_vap_window`)

**Files:**
- Create: `src/vap_window.py`
- Test: `tests/test_vap_window.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vap_window.py
import numpy as np
from src.vap_window import _extract_vap_window


def test_normal_window_is_last_n_inclusive_of_fr():
    arr = np.arange(100 * 18, dtype=np.float32).reshape(100, 18)
    w = _extract_vap_window(arr, fr=50, N=20)
    assert w.shape == (20, 18)
    np.testing.assert_array_equal(w, arr[31:51])  # inclusive of fr=50


def test_near_start_left_replicate_pads_earliest_frame():
    arr = np.arange(10 * 18, dtype=np.float32).reshape(10, 18)
    w = _extract_vap_window(arr, fr=5, N=20)
    assert w.shape == (20, 18)
    np.testing.assert_array_equal(w[-6:], arr[0:6])     # real frames at the end
    for i in range(20 - 6):                              # left pad = replicate arr[0]
        np.testing.assert_array_equal(w[i], arr[0])


def test_none_returns_zeros():
    w = _extract_vap_window(None, fr=0, N=20)
    assert w.shape == (20, 18) and not w.any()


def test_empty_returns_zeros():
    w = _extract_vap_window(np.zeros((0, 18), np.float32), fr=0, N=20)
    assert w.shape == (20, 18) and not w.any()


def test_fr_clamped_above_range():
    arr = np.arange(5 * 18, dtype=np.float32).reshape(5, 18)
    w = _extract_vap_window(arr, fr=999, N=3)
    np.testing.assert_array_equal(w, arr[2:5])           # fr clamps to 4


def test_old_1d_single_frame_treated_as_one_frame():
    arr = np.arange(18, dtype=np.float32)                 # shape (18,)
    w = _extract_vap_window(arr, fr=0, N=4)
    assert w.shape == (4, 18)
    for i in range(4):
        np.testing.assert_array_equal(w[i], arr)


def test_train_and_test_paths_agree_on_same_frames():
    arr = np.random.RandomState(0).randn(60, 18).astype(np.float32)
    fr, N = 40, 20
    train_w = _extract_vap_window(arr, fr=fr, N=N)
    cache = arr[fr - N + 1: fr + 1]                       # what precompute saves: [N,18]
    test_w = _extract_vap_window(cache, fr=cache.shape[0] - 1, N=N)
    np.testing.assert_array_equal(train_w, test_w)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_vap_window.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.vap_window'`.

- [ ] **Step 3: Implement `src/vap_window.py` (window part)**

```python
"""VAP 窗口提取与逐帧聚合（纯 numpy，可离线复用且本地可单测，不引入 torch/torchaudio）。"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

VAP_FEAT_DIM = 18


def _extract_vap_window(arr, fr: int, N: int, feat_dim: int = VAP_FEAT_DIM) -> np.ndarray:
    """取以帧 ``fr`` 结尾、长度 ``N`` 的窗口 -> [N, feat_dim]。

    - ``arr`` 为整段 ``[F, feat_dim]``（训练缓存）或测试缓存 ``[M, feat_dim]``；
      旧的单帧 ``(feat_dim,)`` 视为 1 帧。``None``/空 -> 全零。
    - 不足 N 帧时**左侧复制最早一帧**补齐，保持末帧=边界的时序方向。
    - ``fr`` 自动 clamp 到 ``[0, len-1]``。
    """
    if arr is None:
        return np.zeros((N, feat_dim), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return np.zeros((N, feat_dim), dtype=np.float32)
    fr = int(min(max(fr, 0), arr.shape[0] - 1))
    start = max(0, fr - N + 1)
    window = arr[start: fr + 1]                       # [<=N, D]
    if window.shape[0] < N:
        pad = np.repeat(window[:1], N - window.shape[0], axis=0)
        window = np.concatenate([pad, window], axis=0)
    return np.ascontiguousarray(window[-N:], dtype=np.float32)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_vap_window.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vap_window.py tests/test_vap_window.py
git commit -m "feat(vap): add _extract_vap_window with left replicate-pad"
```

---

## Task 2: Streaming last-N-frame aggregation (`vap_last_n_frames`)

**Files:**
- Modify: `src/vap_window.py`
- Test: `tests/test_vap_window.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_vap_window.py`)**

```python
from collections import deque as _dq
from src.vap_window import vap_last_n_frames, _flat_result


class _FakeQueue:
    def __init__(self): self._items = []
    def empty(self): return not self._items
    def get(self): return self._items.pop(0)
    def put(self, x): self._items.append(x)


def _frame_result(k: int) -> dict:
    # vad[0] = k acts as a frame-index tag; _flat_result puts vad at index 4
    return {"p_now": [0.0, 0.0], "p_future": [0.0, 0.0], "vad": [float(k), 0.0],
            "p_bins": [[0, 0, 0, 0], [0, 0, 0, 0]],
            "p_bins_now": [0.0, 0.0], "p_bins_future": [0.0, 0.0]}


class _FakeMaai:
    """Emits exactly one result dict per process() call, tagged by call index."""
    def __init__(self): self.result_dict_queue = _FakeQueue(); self._i = 0
    def reset_runtime_state(self): self._i = 0
    def process(self, c1, c2):
        self.result_dict_queue.put(_frame_result(self._i)); self._i += 1


def test_flat_result_is_18_dims_with_vad_tag():
    v = _flat_result(_frame_result(7))
    assert len(v) == 18 and v[4] == 7.0


def test_keeps_last_n_frames_when_more_than_n():
    fs, M, N = 4, 10, 5
    audio2 = np.zeros((2, fs * M), dtype=np.float32)     # 10 process() calls
    out = vap_last_n_frames(_FakeMaai(), audio2, frame_samples=fs, N=N)
    assert out.shape == (N, 18)
    np.testing.assert_array_equal(out[:, 4], np.array([5, 6, 7, 8, 9], np.float32))


def test_left_pads_when_fewer_than_n():
    fs, M, N = 4, 3, 5
    audio2 = np.zeros((2, fs * M), dtype=np.float32)     # 3 process() calls -> frames 0,1,2
    out = vap_last_n_frames(_FakeMaai(), audio2, frame_samples=fs, N=N)
    assert out.shape == (N, 18)
    np.testing.assert_array_equal(out[:, 4], np.array([0, 0, 0, 1, 2], np.float32))


def test_empty_audio_returns_zeros():
    out = vap_last_n_frames(_FakeMaai(), np.zeros((2, 0), np.float32), frame_samples=4, N=5)
    assert out.shape == (5, 18) and not out.any()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_vap_window.py -k "last_n or flat_result" -v`
Expected: FAIL — `ImportError: cannot import name 'vap_last_n_frames'`.

- [ ] **Step 3: Implement (append to `src/vap_window.py`)**

```python
def _flat_result(r: dict) -> List[float]:
    """把一帧 VAP result dict 拍平成 18 维（缺字段用 0 兜底）。"""
    pn = [float(x) for x in r.get("p_now", [0.0, 0.0])]
    pf = [float(x) for x in r.get("p_future", [0.0, 0.0])]
    vd = [float(x) for x in r.get("vad", [0.0, 0.0])]
    pb = r.get("p_bins")
    pb_flat: List[float] = []
    if pb is not None:
        for spk in pb:
            pb_flat.extend(float(x) for x in spk)
    pb_flat = (pb_flat + [0.0] * 8)[:8]
    pbn = [float(x) for x in r.get("p_bins_now", [0.0, 0.0])]
    pbf = [float(x) for x in r.get("p_bins_future", [0.0, 0.0])]
    return pn + pf + vd + pb_flat + pbn + pbf


def vap_last_n_frames(maai, audio2: np.ndarray, frame_samples: int, N: int,
                      feat_dim: int = VAP_FEAT_DIM) -> np.ndarray:
    """流式跑整段，保留最后 N 帧 VAP 特征 -> [N, feat_dim]（不足左 replicate-pad）。"""
    maai.reset_runtime_state()
    q = maai.result_dict_queue
    while not q.empty():
        q.get()
    T = audio2.shape[1]
    buf: deque = deque(maxlen=N)
    for i in range(0, T, frame_samples):
        c1 = np.ascontiguousarray(audio2[0, i:i + frame_samples])
        c2 = np.ascontiguousarray(audio2[1, i:i + frame_samples])
        if c1.shape[0] == 0:
            break
        maai.process(c1, c2)
        while not q.empty():
            buf.append(_flat_result(q.get()))
    if not buf:
        return np.zeros((N, feat_dim), dtype=np.float32)
    arr = np.asarray(list(buf), dtype=np.float32)         # [<=N, D]
    return _extract_vap_window(arr, arr.shape[0] - 1, N, feat_dim)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_vap_window.py -v`
Expected: PASS (all tests, 11 total).

- [ ] **Step 5: Commit**

```bash
git add src/vap_window.py tests/test_vap_window.py
git commit -m "feat(vap): add vap_last_n_frames streaming aggregator"
```

---

## Task 3: Learned pooling encoder (`VapWindowEncoder`)

**Files:**
- Create: `src/vap_pool.py`
- Test: `tests/test_vap_pool.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vap_pool.py
import torch
from src.vap_pool import VapWindowEncoder


def test_output_shape_is_batch_by_hidden():
    enc = VapWindowEncoder(feat_dim=18, hidden=320, conv_channels=64)
    out = enc(torch.randn(4, 20, 18))
    assert out.shape == (4, 320)


def test_length_agnostic():
    enc = VapWindowEncoder(feat_dim=18, hidden=320, conv_channels=64)
    for N in (1, 5, 20, 30):
        assert enc(torch.randn(2, N, 18)).shape == (2, 320)


def test_zeros_input_is_finite():
    enc = VapWindowEncoder(feat_dim=18, hidden=320, conv_channels=64)
    out = enc(torch.zeros(3, 20, 18))
    assert out.shape == (3, 320) and torch.isfinite(out).all()


def test_deterministic_in_eval():
    enc = VapWindowEncoder(feat_dim=18, hidden=16, conv_channels=8).eval()
    x = torch.randn(2, 10, 18)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_vap_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.vap_pool'`.

- [ ] **Step 3: Implement `src/vap_pool.py`**

```python
"""VAP 窗口学习式池化编码器（torch-only，本地可单测，不引入 transformers/torchaudio）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class _AttnPool1d(nn.Module):
    """单查询注意力池化：[B, T, H] -> [B, H]。"""
    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.scale = hidden ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = (self.query * x).sum(dim=-1) * self.scale     # [B, T]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)                        # [B, H]


class VapWindowEncoder(nn.Module):
    """[B, N, feat_dim] -> [B, hidden]：conv1d 时序编码 + 注意力池化。长度无关。"""
    def __init__(self, feat_dim: int = 18, hidden: int = 320,
                 conv_channels: int = 64, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = _AttnPool1d(hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))     # [B, hidden, N]
        h = h.transpose(1, 2)                # [B, N, hidden]
        return self.drop(self.norm(self.pool(h)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_vap_pool.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vap_pool.py tests/test_vap_pool.py
git commit -m "feat(vap): add VapWindowEncoder (conv1d + attention pooling)"
```

---

## Task 4: Wire datasets to window extraction

**Files:**
- Modify: `src/data/dataset.py` (import; `TurnTakingTrainDataset.__init__` + `__getitem__`; `TurnTakingTestDataset.__init__` + `__getitem__`)

Not locally runnable (module import blocked by env). Verify by code review; behaviour is covered by Task 1 tests.

- [ ] **Step 1: Add import near the other `src` imports at the top of `src/data/dataset.py`**

```python
from src.vap_window import _extract_vap_window
```

- [ ] **Step 2: Add `vap_window` param to `TurnTakingTrainDataset.__init__`**

In the signature, after `vap_feat_dim: int = 18,` add:

```python
        vap_window: int = 20,
```

In the body, after `self.vap_feat_dim = int(vap_feat_dim)` add:

```python
        self.vap_window = int(vap_window)
```

- [ ] **Step 3: Replace the train `vap_feat` block in `TurnTakingTrainDataset.__getitem__`**

Replace:

```python
        if self.vap_feat_dir is not None:
            arr = self._load_vap_feats(sample.conv_id)
            vf = np.zeros(self.vap_feat_dim, dtype=np.float32)
            if arr is not None and arr.shape[0] > 0:
                fr = int(round(end_idx * self.chunk_ms * self.vap_frame_rate / 1000.0))
                fr = min(max(fr, 0), arr.shape[0] - 1)
                vf = np.asarray(arr[fr], dtype=np.float32)
            out["vap_feat"] = torch.from_numpy(vf)
```

With:

```python
        if self.vap_feat_dir is not None:
            arr = self._load_vap_feats(sample.conv_id)
            fr = int(round(end_idx * self.chunk_ms * self.vap_frame_rate / 1000.0))
            win = _extract_vap_window(arr, fr, self.vap_window, self.vap_feat_dim)  # [N, D]
            out["vap_feat"] = torch.from_numpy(win)
```

- [ ] **Step 4: Add `vap_window` param to `TurnTakingTestDataset.__init__`**

In the signature, after `vap_feat_dim: int = 18,` add:

```python
        vap_window: int = 20,
```

In the body, after `self.vap_feat_dim = int(vap_feat_dim)` add:

```python
        self.vap_window = int(vap_window)
```

- [ ] **Step 5: Replace the test `vap_feat` block in `TurnTakingTestDataset.__getitem__`**

Replace:

```python
        if self.vap_feat_dir is not None:
            vf = np.zeros(self.vap_feat_dim, dtype=np.float32)
            vp = self.vap_feat_dir / f"{seg_id}.npy"
            if vp.exists():
                arr = np.load(vp).astype(np.float32).reshape(-1)
                if arr.shape[0] >= self.vap_feat_dim:
                    vf = arr[: self.vap_feat_dim]
                else:
                    vf[: arr.shape[0]] = arr
            out["vap_feat"] = torch.from_numpy(vf)
```

With:

```python
        if self.vap_feat_dir is not None:
            arr = None
            vp = self.vap_feat_dir / f"{seg_id}.npy"
            if vp.exists():
                arr = np.load(vp).astype(np.float32)
            fr = (arr.shape[0] - 1) if (arr is not None and arr.ndim == 2 and arr.shape[0] > 0) else 0
            win = _extract_vap_window(arr, fr, self.vap_window, self.vap_feat_dim)  # [N, D]
            out["vap_feat"] = torch.from_numpy(win)
```

- [ ] **Step 6: Sanity-check the diff and commit**

Run: `git diff --stat src/data/dataset.py` (expect one file changed).

```bash
git add src/data/dataset.py
git commit -m "feat(vap): datasets emit [N,18] window via _extract_vap_window"
```

---

## Task 5: Wire model to `VapWindowEncoder`

**Files:**
- Modify: `src/models/multimodal_baseline.py` (import; `use_vap_feat` block in `__init__`; forward branch)

Not locally runnable (transformers import blocked). Verify by code review; encoder behaviour is covered by Task 3 tests.

- [ ] **Step 1: Add import near the top (with the other imports)**

```python
from src.vap_pool import VapWindowEncoder
```

- [ ] **Step 2: Replace the `use_vap_feat` init block in `MultimodalTurnTakingModel.__init__`**

Replace:

```python
        if self.use_vap_feat:
            self.vap_feat_dim = int(vf_cfg.get("feat_dim", 18))
            _h = self.fusion.out_dim
            self.vap_feat_proj = nn.Sequential(
                nn.Linear(self.vap_feat_dim, _h), nn.LayerNorm(_h), nn.GELU(),
            )
            self.vap_feat_merge = nn.Sequential(
                nn.Linear(_h * 2, _h), nn.LayerNorm(_h), nn.GELU(),
            )
```

With:

```python
        if self.use_vap_feat:
            self.vap_feat_dim = int(vf_cfg.get("feat_dim", 18))
            self.vap_window = int(vf_cfg.get("window", 20))
            _h = self.fusion.out_dim
            self.vap_feat_encoder = VapWindowEncoder(
                feat_dim=self.vap_feat_dim,
                hidden=_h,
                conv_channels=int(vf_cfg.get("conv_channels", 64)),
                dropout=float(cfg.get("fusion", {}).get("dropout", 0.0)),
            )
            self.vap_feat_merge = nn.Sequential(
                nn.Linear(_h * 2, _h), nn.LayerNorm(_h), nn.GELU(),
            )
```

- [ ] **Step 3: Replace the forward `use_vap_feat` branch**

Replace:

```python
        if getattr(self, "use_vap_feat", False):
            if vap_feat is None:
                vap_feat = fused.new_zeros(fused.shape[0], self.vap_feat_dim)
            v = self.vap_feat_proj(vap_feat.to(fused.dtype))
            fused = self.vap_feat_merge(torch.cat([fused, v], dim=-1))
```

With:

```python
        if getattr(self, "use_vap_feat", False):
            if vap_feat is None:
                vap_feat = fused.new_zeros(fused.shape[0], self.vap_window, self.vap_feat_dim)
            elif vap_feat.dim() == 2:
                # 兼容旧单帧 [B, feat_dim] 输入：升一维成 [B, 1, feat_dim]
                vap_feat = vap_feat.unsqueeze(1)
            v = self.vap_feat_encoder(vap_feat.to(fused.dtype))
            fused = self.vap_feat_merge(torch.cat([fused, v], dim=-1))
```

- [ ] **Step 4: Sanity-check and commit**

Run: `git diff --stat src/models/multimodal_baseline.py` (expect one file changed).

```bash
git add src/models/multimodal_baseline.py
git commit -m "feat(vap): use VapWindowEncoder over [B,N,18] in model"
```

---

## Task 6: Test-set precompute saves last-N frames

**Files:**
- Modify: `src/precompute_vap_test.py` (import; drop local `_flat_result`/`vap_last_frame_feature`; add `--window`; thread `N`)

Not locally runnable (imports `src.data.dataset`). Verify by code review; `vap_last_n_frames` is covered by Task 2 tests.

- [ ] **Step 1: Replace the local helpers with an import**

Delete the local `_flat_result(...)` and `vap_last_frame_feature(...)` definitions. Add near the top imports:

```python
from src.vap_window import vap_last_n_frames
```

- [ ] **Step 2: Add `--window` CLI arg (in `main`, near the other `ap.add_argument` calls)**

```python
    ap.add_argument("--window", type=int, default=20, help="保留最后 N 帧 VAP 特征 -> (N,18)")
```

- [ ] **Step 3: Thread `window` into `init_args`**

In the `init_args = dict(...)` block add:

```python
        window=int(args.window),
```

In `_worker_init`, in the `_W_CFG = {...}` dict add:

```python
        "window": int(init_args["window"]),
```

- [ ] **Step 4: Use `vap_last_n_frames` in `_worker_process_seg`**

Replace:

```python
        feat = vap_last_frame_feature(_W_MAAI, audio2, _W_CFG["frame_samples"])
```

With:

```python
        feat = vap_last_n_frames(_W_MAAI, audio2, _W_CFG["frame_samples"], _W_CFG["window"])
```

- [ ] **Step 5: Commit**

```bash
git add src/precompute_vap_test.py
git commit -m "feat(vap): test precompute saves last-N frames (N,18)"
```

---

## Task 7: Thread `vap_window` through training

**Files:**
- Modify: `src/train.py` (read config; pass to both datasets)

Not locally runnable. Verify by code review.

- [ ] **Step 1: Read the window after `vap_feat_dim_cfg` is read**

After the line `vap_feat_dim_cfg = int(vapfeat_cfg.get("feat_dim", 18))` add:

```python
    vap_feat_window_cfg = int(vapfeat_cfg.get("window", 20))
```

- [ ] **Step 2: Pass `vap_window` to the train dataset**

In the `train_dataset = TurnTakingTrainDataset(...)` call, after `vap_feat_dim=vap_feat_dim_cfg,` add:

```python
        vap_window=vap_feat_window_cfg,
```

- [ ] **Step 3: Pass `vap_window` to the valid dataset**

In the `valid_dataset = TurnTakingTrainDataset(...)` call, after `vap_feat_dim=vap_feat_dim_cfg,` add:

```python
        vap_window=vap_feat_window_cfg,
```

- [ ] **Step 4: Commit**

```bash
git add src/train.py
git commit -m "feat(vap): pass vap_feat.window to train/valid datasets"
```

---

## Task 8: Thread `vap_window` through ensemble inference

**Files:**
- Modify: `src/infer_ensemble.py` (pass to test dataset)

Not locally runnable. Verify by code review.

- [ ] **Step 1: Pass `vap_window` in the `TurnTakingTestDataset(...)` call**

After `vap_feat_dim=int((cfg.get("vap_feat", {}) or {}).get("feat_dim", 18)),` add:

```python
        vap_window=int((cfg.get("vap_feat", {}) or {}).get("window", 20)),
```

- [ ] **Step 2: Commit**

```bash
git add src/infer_ensemble.py
git commit -m "feat(vap): pass vap_feat.window to ensemble test dataset"
```

---

## Task 9: Config for clean A/B

**Files:**
- Create: `configs/whisper_qwen0_6b_lmf_vapwin.yaml`

- [ ] **Step 1: Copy the baseline config**

```bash
cp configs/whisper_qwen0_6b_lmf_vapfeat.yaml configs/whisper_qwen0_6b_lmf_vapwin.yaml
```

- [ ] **Step 2: Edit `configs/whisper_qwen0_6b_lmf_vapwin.yaml`**

In `paths:`, change the three output paths from `outputs/lmf_vapfeat` to `outputs/lmf_vapwin`:

```yaml
  output_root: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_vapwin
  checkpoints_dir: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_vapwin/checkpoints
  logs_dir: /mnt/workspace/dorihue/finvcup_2026/outputs/lmf_vapwin/logs
```

In the `vap_feat:` block, add `window` and `conv_channels` (keep `cache_dir` — the train cache `[F,18]` is reusable as-is):

```yaml
vap_feat:
  enabled: true
  feat_dim: 18
  window: 20
  conv_channels: 64
  frame_rate: 10
  cache_dir: /mnt/workspace/dorihue/finvcup_2026/.cache/vap_ch_kyoto
```

- [ ] **Step 3: Commit**

```bash
git add configs/whisper_qwen0_6b_lmf_vapwin.yaml
git commit -m "feat(vap): add lmf_vapwin config (window=20, clean A/B output dir)"
```

---

## Task 10: Full local test sweep

- [ ] **Step 1: Run all new tests**

Run: `python -m pytest tests/ -v`
Expected: PASS (15 tests: 11 in test_vap_window.py + 4 in test_vap_pool.py).

- [ ] **Step 2: Final review**

Confirm `git log --oneline` shows the per-task commits and the working tree is clean (`git status`).

---

## Server smoke (run by user, after merge)

Not part of local execution; documents how to validate end-to-end on the L20 server (torch 2.x, data present):

1. Re-precompute test VAP with window: `python -m src.precompute_vap_test ... --window 20` → check a `<seg>.npy` has shape `(20, 18)`.
2. Short A/B training: `python -m src.train --config configs/whisper_qwen0_6b_lmf_vapwin.yaml` for the same step budget as `lmf_vapfeat` (`max_steps_per_epoch=500`, ~6–8 epochs).
3. Compare `outputs/lmf_vapwin/logs/eval_epoch_*.json` vs `outputs/lmf_vapfeat/logs`: focus on **Δmacro_roc_auc** (threshold-free) and per-class `bc/i/t` `best_f1`/`roc_auc`. C/NA should stay flat.
