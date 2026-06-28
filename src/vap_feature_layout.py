"""Shared layout helpers for offline VAP and BC feature caches."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np

VAP_FEAT_DIM = 18
BC_FEAT_DIM = 3
VAP_BC_FEAT_DIM = VAP_FEAT_DIM + BC_FEAT_DIM

VAP_FEAT_LAYOUT = [
    "p_now_0",
    "p_now_1",
    "p_future_0",
    "p_future_1",
    "vad_0",
    "vad_1",
    "p_bins_s0_b0",
    "p_bins_s0_b1",
    "p_bins_s0_b2",
    "p_bins_s0_b3",
    "p_bins_s1_b0",
    "p_bins_s1_b1",
    "p_bins_s1_b2",
    "p_bins_s1_b3",
    "p_bins_now_0",
    "p_bins_now_1",
    "p_bins_future_0",
    "p_bins_future_1",
]

BC_FEAT_LAYOUT = [
    "bc_p_last",
    "bc_p_max_tail_2s",
    "bc_p_mean_tail_2s",
]

VAP_BC_FEAT_LAYOUT = VAP_FEAT_LAYOUT + BC_FEAT_LAYOUT


def _to_float_list(value: Any, default: Sequence[float]) -> List[float]:
    if value is None:
        return [float(x) for x in default]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(x) for x in arr.tolist()]


def flat_vap_result(result: dict) -> List[float]:
    """Flatten one MaAI vap/vap_mc result into the stable 18-dim layout."""

    p_now = _to_float_list(result.get("p_now"), [0.0, 0.0])[:2]
    p_future = _to_float_list(result.get("p_future"), [0.0, 0.0])[:2]
    vad = _to_float_list(result.get("vad"), [0.0, 0.0])[:2]

    p_bins_flat: List[float] = []
    p_bins = result.get("p_bins")
    if p_bins is not None:
        if hasattr(p_bins, "detach"):
            p_bins = p_bins.detach().cpu().numpy()
        for speaker_bins in np.asarray(p_bins, dtype=np.float32):
            p_bins_flat.extend(float(x) for x in np.asarray(speaker_bins).reshape(-1))
    p_bins_flat = (p_bins_flat + [0.0] * 8)[:8]

    p_bins_now = _to_float_list(result.get("p_bins_now"), [0.0, 0.0])[:2]
    p_bins_future = _to_float_list(result.get("p_bins_future"), [0.0, 0.0])[:2]

    values = p_now + p_future + vad + p_bins_flat + p_bins_now + p_bins_future
    return (values + [0.0] * VAP_FEAT_DIM)[:VAP_FEAT_DIM]


def flat_bc_result(result: dict) -> float:
    """Extract scalar p_bc from a MaAI bc result."""

    value = result.get("p_bc", result.get("p_bc_detect", 0.0))
    values = _to_float_list(value, [0.0])
    return float(values[-1] if values else 0.0)


def bc_tail_summary(
    values: Iterable[float],
    frame_rate: float,
    tail_sec: float = 2.0,
) -> np.ndarray:
    """Return [last, max_tail, mean_tail] over the causal tail window."""

    arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((BC_FEAT_DIM,), dtype=np.float32)
    tail_frames = max(1, int(round(float(frame_rate) * float(tail_sec))))
    tail = arr[-tail_frames:]
    return np.asarray([arr[-1], float(np.max(tail)), float(np.mean(tail))], dtype=np.float32)


def append_bc_tail_features(
    vap_feats: np.ndarray,
    bc_values: Sequence[float],
    frame_rate: float,
    tail_sec: float = 2.0,
) -> np.ndarray:
    """Append causal BC timing summaries to each row of an [F,18] VAP cache."""

    vap_arr = np.asarray(vap_feats, dtype=np.float32)
    if vap_arr.ndim != 2:
        raise ValueError(f"vap_feats must be 2-D, got shape={vap_arr.shape}")
    if vap_arr.shape[1] != VAP_FEAT_DIM:
        raise ValueError(f"vap_feats must have {VAP_FEAT_DIM} columns, got {vap_arr.shape[1]}")

    frames = int(vap_arr.shape[0])
    bc_arr = np.asarray(bc_values, dtype=np.float32).reshape(-1)
    if bc_arr.size < frames:
        bc_arr = np.pad(bc_arr, (0, frames - bc_arr.size), mode="constant")
    elif bc_arr.size > frames:
        bc_arr = bc_arr[:frames]

    bc_feats = np.zeros((frames, BC_FEAT_DIM), dtype=np.float32)
    tail_frames = max(1, int(round(float(frame_rate) * float(tail_sec))))
    for idx in range(frames):
        start = max(0, idx - tail_frames + 1)
        tail = bc_arr[start: idx + 1]
        bc_feats[idx] = np.asarray(
            [bc_arr[idx], float(np.max(tail)), float(np.mean(tail))],
            dtype=np.float32,
        )
    return np.concatenate([vap_arr, bc_feats], axis=1).astype(np.float32, copy=False)


def append_bc_tail_to_last_feature(
    vap_feat: np.ndarray,
    bc_values: Sequence[float],
    frame_rate: float,
    tail_sec: float = 2.0,
) -> np.ndarray:
    """Append BC tail summary to a single final-frame VAP feature."""

    vap_arr = np.asarray(vap_feat, dtype=np.float32).reshape(-1)
    if vap_arr.size != VAP_FEAT_DIM:
        raise ValueError(f"vap_feat must have {VAP_FEAT_DIM} values, got {vap_arr.size}")
    bc_feat = bc_tail_summary(bc_values, frame_rate, tail_sec)
    return np.concatenate([vap_arr, bc_feat]).astype(np.float32, copy=False)
