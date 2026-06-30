"""VAP window extraction and streaming aggregation helpers."""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from src.vap_feature_layout import (
    VAP_BC_FEAT_DIM,
    VAP_FEAT_DIM,
    append_bc_tail_features,
    flat_bc_result,
    flat_vap_result,
)


def _extract_vap_window(arr, fr: int, N: int, feat_dim: int = VAP_FEAT_DIM) -> np.ndarray:
    """Return the length-N window ending at frame fr, with left replicate padding."""

    if arr is None:
        return np.zeros((N, feat_dim), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return np.zeros((N, feat_dim), dtype=np.float32)
    fr = int(min(max(fr, 0), arr.shape[0] - 1))
    start = max(0, fr - N + 1)
    window = arr[start: fr + 1]
    if window.shape[0] < N:
        pad = np.repeat(window[:1], N - window.shape[0], axis=0)
        window = np.concatenate([pad, window], axis=0)
    return np.ascontiguousarray(window[-N:], dtype=np.float32)


def _flat_result(r: dict) -> List[float]:
    """Backward-compatible alias for flattening one VAP result to 18 dims."""

    return flat_vap_result(r)


def vap_last_n_frames(
    maai,
    audio2: np.ndarray,
    frame_samples: int,
    N: int,
    feat_dim: int = VAP_FEAT_DIM,
) -> np.ndarray:
    """Stream a segment through VAP and keep the final N frames."""

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
    arr = np.asarray(list(buf), dtype=np.float32)
    return _extract_vap_window(arr, arr.shape[0] - 1, N, feat_dim)


def vap_bc_last_n_frames(
    vap_maai,
    bc_maai,
    audio2: np.ndarray,
    frame_samples: int,
    N: int,
    frame_rate: float,
    bc_tail_sec: float,
) -> np.ndarray:
    """Run VAP and BC streams, then return the final [N, 21] causal window."""

    vap_maai.reset_runtime_state()
    bc_maai.reset_runtime_state()
    vap_q = vap_maai.result_dict_queue
    bc_q = bc_maai.result_dict_queue
    while not vap_q.empty():
        vap_q.get()
    while not bc_q.empty():
        bc_q.get()

    T = audio2.shape[1]
    vap_feats: list[list[float]] = []
    bc_values: list[float] = []
    for i in range(0, T, frame_samples):
        c1 = np.ascontiguousarray(audio2[0, i:i + frame_samples])
        c2 = np.ascontiguousarray(audio2[1, i:i + frame_samples])
        if c1.shape[0] == 0:
            break
        vap_maai.process(c1, c2)
        bc_maai.process(c1, c2)
        while not vap_q.empty():
            vap_feats.append(_flat_result(vap_q.get()))
        while not bc_q.empty():
            bc_values.append(flat_bc_result(bc_q.get()))

    if not vap_feats:
        return np.zeros((N, VAP_BC_FEAT_DIM), dtype=np.float32)
    full = append_bc_tail_features(
        np.asarray(vap_feats, dtype=np.float32),
        bc_values,
        frame_rate=frame_rate,
        tail_sec=bc_tail_sec,
    )
    return _extract_vap_window(full, full.shape[0] - 1, N, VAP_BC_FEAT_DIM)
