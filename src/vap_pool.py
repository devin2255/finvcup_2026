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
