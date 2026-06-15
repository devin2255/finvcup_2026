"""Lightweight unit tests for the new improvements (no Whisper/Qwen download).

Covers the pure-logic parts of:
- WhisperAudioEncoder._tail_mask  (变长 padding mask)
- stereo channel concat reshape
- aux chunk head reshape + cross-entropy plumbing
- ModelEMA  (真权重 EMA: update / copy_to / restore)
- BC oversampling list-duplication snippet
- CollateFn wave_len / chunk_labels

Run:  python -m tests.test_new_features
"""
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


def test_tail_mask():
    from src.models.multimodal_baseline import WhisperAudioEncoder

    # Fake "self" with just the attributes _tail_mask needs.
    enc = SimpleNamespace(tail_ratio=0.25, _samples_per_frame=320)
    fn = WhisperAudioEncoder._tail_mask.__get__(enc, WhisperAudioEncoder)

    T = 1500  # whisper-large-v3 encoder frames for 30s

    # (a) wave_len=None -> full-length tail, last 25% of T
    m = fn(T, None, n_items=2, device=torch.device("cpu"))
    assert m.shape == (2, T)
    assert m.sum(dim=1).tolist() == [375, 375], m.sum(dim=1).tolist()
    assert m[0, :1125].sum().item() == 0 and m[0, 1125:].all()

    # (b) variable length: 30s and 10s clips in one batch.
    wl = torch.tensor([30 * 16000, 10 * 16000])  # samples
    m = fn(T, wl, n_items=2, device=torch.device("cpu"))
    valid = (wl.float() / 320).round().long()        # [1500, 500]
    tail = (valid.float() * 0.25).round().long()     # [375, 125]
    assert m.sum(dim=1).tolist() == tail.tolist(), (m.sum(dim=1).tolist(), tail.tolist())
    # 10s clip: tail must sit at the END of the real audio (before zero-pad), not at T.
    last_true = m[1].nonzero().max().item()
    assert last_true == valid[1].item() - 1 == 499, last_true
    assert m[1, 500:].sum().item() == 0  # never attends to the silence padding
    print("[ok] _tail_mask: full-length + variable-length masking correct")


def test_stereo_concat_shape():
    # Mirror the concat in WhisperAudioEncoder.forward: [B*2, D] -> [B, 3D]
    B, D = 4, 8
    pooled = torch.randn(B * 2, D)
    pooled = pooled.view(B, 2, D)
    ch0, ch1 = pooled[:, 0, :], pooled[:, 1, :]
    out = torch.cat([ch0, ch1, (ch0 - ch1).abs()], dim=-1)
    assert out.shape == (B, 3 * D)
    # diff block must equal |ch0-ch1|
    assert torch.allclose(out[:, 2 * D:], (ch0 - ch1).abs())
    print("[ok] stereo concat -> [B, 3*hidden]")


def test_aux_head_loss():
    B, Tc, Cc = 3, 25, 5
    head = nn.Linear(16, Tc * Cc)
    fused = torch.randn(B, 16)
    chunk_logits = head(fused).view(B, Tc, Cc)
    chunk_labels = torch.randint(0, Cc, (B, Tc))
    loss = nn.CrossEntropyLoss()(chunk_logits.reshape(B * Tc, Cc), chunk_labels.reshape(B * Tc))
    assert loss.ndim == 0 and torch.isfinite(loss)
    print(f"[ok] aux chunk head CE loss = {loss.item():.4f}")


def test_model_ema():
    from src.utils import ModelEMA

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    # freeze bias -> EMA must ignore it
    model.bias.requires_grad = False
    init_w = model.weight.detach().clone()

    ema = ModelEMA(model, decay=0.9)
    assert "bias" not in ema.shadow and "weight" in ema.shadow

    # simulate an optimizer step changing the weight
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    expected = 0.9 * init_w + 0.1 * (init_w + 1.0)
    assert torch.allclose(ema.shadow["weight"], expected), "EMA update math wrong"

    # copy_to swaps EMA in; restore brings the live weights back
    live = model.weight.detach().clone()  # == init_w + 1
    ema.copy_to(model)
    assert torch.allclose(model.weight, expected)
    ema.restore(model)
    assert torch.allclose(model.weight, live)
    print("[ok] ModelEMA: shadow/update/copy_to/restore correct, frozen param skipped")


def test_bc_oversample():
    # Mirror train.py BC oversampling on synthetic samples.
    multi_targets = ["C", "NA", "I", "BC", "T"]
    bc_idx = multi_targets.index("BC")

    class S:
        def __init__(self, v):
            self.label_vec = v

    samples = [S((1, 1, 0, 0, 0)) for _ in range(90)] + [S((1, 0, 0, 1, 0)) for _ in range(10)]
    oversample_bc = 3
    bc_pos = [s for s in samples if int(s.label_vec[bc_idx]) == 1]
    extra = len(bc_pos) * (oversample_bc - 1)
    out = list(samples) + bc_pos * (oversample_bc - 1)
    assert len(bc_pos) == 10 and extra == 20 and len(out) == 120
    n_bc = sum(1 for s in out if s.label_vec[bc_idx] == 1)
    assert n_bc == 30  # 10 -> 30 (x3)
    print(f"[ok] BC oversample x{oversample_bc}: {len(samples)} -> {len(out)} samples, BC pos 10 -> {n_bc}")


def test_collate_wave_len_and_chunks():
    try:
        from src.data.dataset import CollateFn
    except ModuleNotFoundError as e:
        # dataset.py imports torchaudio at module top; skip if unavailable
        # (e.g. local CPU box). The training env has it.
        print(f"[skip] CollateFn test ({e}) — torchaudio not installed in this env")
        return

    class FakeTok:
        truncation_side = "right"
        padding_side = "right"

        def __call__(self, texts, **kw):
            n = len(texts)
            return {"input_ids": torch.zeros(n, 3, dtype=torch.long),
                    "attention_mask": torch.ones(n, 3, dtype=torch.long)}

    collate = CollateFn(FakeTok(), text_max_length=8)
    batch = [
        {"waveform": torch.randn(2, 16000), "text": "a",
         "context_labels": torch.zeros(375, dtype=torch.long),
         "chunk_labels": torch.zeros(25, dtype=torch.long),
         "label": torch.zeros(5), "conv_id": "x", "end_idx": 375},
        {"waveform": torch.randn(2, 8000), "text": "b",
         "context_labels": torch.zeros(375, dtype=torch.long),
         "chunk_labels": torch.ones(25, dtype=torch.long),
         "label": torch.ones(5), "conv_id": "y", "end_idx": 400},
    ]
    out = collate(batch)
    assert out["wave_len"].tolist() == [16000, 8000]          # true lengths before padding
    assert out["waveform"].shape == (2, 2, 16000)             # padded to max
    assert out["chunk_labels"].shape == (2, 25)
    print("[ok] CollateFn: wave_len captured pre-pad, chunk_labels stacked")


def test_plan_ensemble_update():
    from src.utils import plan_ensemble_update

    topk, gap = 5, 2

    # Simulate epoch-by-epoch metric stream; replicate train.py's apply logic.
    members: list[dict] = []

    def apply(epoch, metric):
        add, evict = plan_ensemble_update(members, epoch, metric, topk, gap)
        if add:
            for e in evict:
                members.remove(e)
            members.append({"name": f"ep{epoch}", "epoch": epoch, "metric": metric})

    # rising metric on consecutive epochs -> must NOT keep adjacent members
    for ep, mt in [(0, 0.50), (1, 0.55), (2, 0.60), (3, 0.62), (4, 0.61),
                   (5, 0.65), (6, 0.64), (7, 0.70), (8, 0.69), (9, 0.66)]:
        apply(ep, mt)

    eps = sorted(m["epoch"] for m in members)
    # invariant: every pair of members is >= gap epochs apart
    assert all(b - a >= gap for a, b in zip(eps, eps[1:])), eps
    assert len(members) <= topk
    # the global best epoch (7, 0.70) must be retained
    assert any(m["epoch"] == 7 for m in members), eps

    # a strictly-better neighbor should evict the adjacent weaker one
    members2 = [{"name": "ep10", "epoch": 10, "metric": 0.60}]
    add, evict = plan_ensemble_update(members2, 11, 0.65, topk, gap)
    assert add and evict and evict[0]["epoch"] == 10  # 11 beats neighbor 10 -> replace

    # a weaker neighbor must be rejected (keeps spacing, keeps the better one)
    add, evict = plan_ensemble_update(members2, 11, 0.55, topk, gap)
    assert not add and not evict
    print("[ok] plan_ensemble_update: spacing invariant + keeps best + neighbor competition")


if __name__ == "__main__":
    test_tail_mask()
    test_stereo_concat_shape()
    test_aux_head_loss()
    test_model_ema()
    test_bc_oversample()
    test_plan_ensemble_update()
    test_collate_wave_len_and_chunks()
    print("\nALL TESTS PASSED")
