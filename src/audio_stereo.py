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
