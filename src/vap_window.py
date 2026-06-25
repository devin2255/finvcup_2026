"""VAP 窗口提取与逐帧聚合（纯 numpy，可离线复用且本地可单测，不引入 torch/torchaudio）。"""
from __future__ import annotations

from collections import deque
from typing import List

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
