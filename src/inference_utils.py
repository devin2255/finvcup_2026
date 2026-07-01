from __future__ import annotations

from typing import Sequence

import numpy as np


def combine_threshold_vectors(
    threshold_vectors: Sequence[np.ndarray],
    metrics: Sequence[float],
    mode: str,
) -> np.ndarray:
    """Combine per-member label thresholds for soft ensemble inference."""
    if not threshold_vectors:
        raise ValueError("threshold_vectors must not be empty")
    arr = np.stack([np.asarray(v, dtype=np.float64) for v in threshold_vectors], axis=0)
    if mode == "mean":
        return arr.mean(axis=0)
    if mode == "best":
        return arr[0].copy()
    if mode == "weighted_mean":
        weights = np.asarray(metrics, dtype=np.float64)
        if (
            weights.shape[0] != arr.shape[0]
            or not np.isfinite(weights).all()
            or weights.sum() <= 0
        ):
            weights = np.ones(arr.shape[0], dtype=np.float64)
        return np.average(arr, axis=0, weights=weights)
    raise ValueError(f"unknown threshold combine mode: {mode}")
