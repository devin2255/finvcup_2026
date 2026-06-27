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
