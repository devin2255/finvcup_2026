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
