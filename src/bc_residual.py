from __future__ import annotations

import torch
import torch.nn as nn


class BcResidualHead(nn.Module):
    """Small BC-only residual head over causal BC tail features."""

    def __init__(self, feat_dim: int = 3, hidden: int = 16, dropout: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.scale = float(scale)
        in_dim = self.feat_dim * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, bc_window: torch.Tensor) -> torch.Tensor:
        if bc_window.dim() != 3:
            raise ValueError(f"bc_window must be [B,N,D], got shape={tuple(bc_window.shape)}")
        last = bc_window[:, -1, :]
        max_tail = bc_window.amax(dim=1)
        mean_tail = bc_window.mean(dim=1)
        summary = torch.cat([last, max_tail, mean_tail], dim=-1)
        return self.net(summary) * self.scale


def apply_label_residual(logits: torch.Tensor, residual: torch.Tensor, target_index: int) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B,C], got shape={tuple(logits.shape)}")
    out = logits.clone()
    out[:, int(target_index)] = out[:, int(target_index)] + residual.reshape(-1).to(out.dtype)
    return out
