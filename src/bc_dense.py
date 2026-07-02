"""BC 逐 chunk 密集监督目标（纯 numpy，本地可单测，不引入 torch/torchaudio）。"""
from __future__ import annotations

import numpy as np


def bc_dense_target(labels, end_idx: int, target_chunks: int, bc_id: int) -> np.ndarray:
    """未来 target_chunks 内逐 chunk 的 BC 0/1 目标 -> float32 [target_chunks]。

    - 取 labels[end_idx : end_idx + target_chunks]，逐 chunk 判 == bc_id。
    - 末尾越界时右侧补 0（无标签视为无 BC），保证输出定长。
    - 语义保证：dense.any() == 窗口级 BC 标签（build_train_samples_multitask 口径）。
    """
    future = np.asarray(labels[end_idx : end_idx + target_chunks])
    dense = (future == bc_id).astype(np.float32)
    if dense.shape[0] < target_chunks:
        dense = np.concatenate(
            [dense, np.zeros(target_chunks - dense.shape[0], dtype=np.float32)]
        )
    return dense
