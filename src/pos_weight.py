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
