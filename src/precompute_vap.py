"""Precompute frame-level VAP features for training caches.

Default output is [F, 18] VAP. With --bc_enabled, output is [F, 21]:
the original VAP layout plus causal BC tail [last, max_tail, mean_tail].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import _read_wav_slice, list_conv_ids
from src.utils import load_config
from src.vap_feature_layout import (
    VAP_BC_FEAT_DIM,
    VAP_BC_FEAT_LAYOUT,
    VAP_FEAT_DIM,
    VAP_FEAT_LAYOUT,
    append_bc_tail_features,
    flat_bc_result,
    flat_vap_result,
)


def _load_conv_2ch(audio_path: Path, target_sr: int) -> np.ndarray:
    data, sr = _read_wav_slice(audio_path, 0, 10**9)
    w = torch.from_numpy(data.T.copy())
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    elif w.shape[0] > 2:
        w = w[:2]
    if sr != target_sr:
        import torchaudio

        w = torchaudio.functional.resample(w, sr, target_sr)
    return w.numpy().astype(np.float32)


def vap_features_for_conv(maai, audio2: np.ndarray, frame_samples: int) -> np.ndarray:
    maai.reset_runtime_state()
    q = maai.result_dict_queue
    while not q.empty():
        q.get()
    feats = []
    for i in range(0, audio2.shape[1], frame_samples):
        c1 = np.ascontiguousarray(audio2[0, i:i + frame_samples])
        c2 = np.ascontiguousarray(audio2[1, i:i + frame_samples])
        if c1.shape[0] == 0:
            break
        maai.process(c1, c2)
        while not q.empty():
            feats.append(flat_vap_result(q.get()))
    if not feats:
        return np.zeros((0, VAP_FEAT_DIM), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def bc_values_for_conv(maai, audio2: np.ndarray, frame_samples: int) -> np.ndarray:
    maai.reset_runtime_state()
    q = maai.result_dict_queue
    while not q.empty():
        q.get()
    values = []
    for i in range(0, audio2.shape[1], frame_samples):
        c1 = np.ascontiguousarray(audio2[0, i:i + frame_samples])
        c2 = np.ascontiguousarray(audio2[1, i:i + frame_samples])
        if c1.shape[0] == 0:
            break
        maai.process(c1, c2)
        while not q.empty():
            values.append(flat_bc_result(q.get()))
    return np.asarray(values, dtype=np.float32)


def _add_maai_to_path(maai_dir: str) -> None:
    maai_path = Path(maai_dir).resolve()
    for cand in (maai_path / "src", maai_path):
        if (cand / "maai").is_dir():
            cand_s = str(cand)
            if cand_s not in sys.path:
                sys.path.insert(0, cand_s)
            return


def main():
    ap = argparse.ArgumentParser(description="Precompute VAP / VAP+BC feature caches")
    ap.add_argument("--config", type=str, required=True, help="reads paths/sample_rate/chunk_ms")
    ap.add_argument("--maai_dir", type=str, default="./MaAI")
    ap.add_argument("--lang", type=str, default="ch_kyoto")
    ap.add_argument("--mode", type=str, default="vap_mc")
    ap.add_argument("--frame_rate", type=float, default=10)
    ap.add_argument("--context_sec", type=float, default=20)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cpc_model", type=str, default="~/.cache/cpc/60k_epoch4-d0f474de.pt")
    ap.add_argument("--local_model", type=str, default=None)
    ap.add_argument("--bc_enabled", action="store_true", help="append causal MaAI BC tail features")
    ap.add_argument("--bc_lang", type=str, default="ch")
    ap.add_argument("--bc_mode", type=str, default="bc")
    ap.add_argument("--bc_local_model", type=str, default=None)
    ap.add_argument("--bc_tail_sec", type=float, default=2.0)
    ap.add_argument("--out_dir", type=str, required=True, help="feature cache directory")
    ap.add_argument("--max_convs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sample_rate = int(cfg["sample_rate"])
    chunk_ms = int(cfg["chunk_ms"])
    audio_dir = Path(cfg["paths"]["train_audio_dir"])
    labels_dir = Path(cfg["paths"]["train_labels_dir"])
    frame_samples = int(round(sample_rate / float(args.frame_rate)))

    _add_maai_to_path(args.maai_dir)
    from maai import Maai, MaaiInput

    print(
        f"[load] VAP Maai(mode={args.mode}, lang={args.lang}, "
        f"frame_rate={args.frame_rate}, context_len_sec={args.context_sec})"
    )
    vap_maai = Maai(
        mode=args.mode,
        lang=args.lang,
        frame_rate=args.frame_rate,
        context_len_sec=int(args.context_sec),
        audio_ch1=MaaiInput.Zero(),
        audio_ch2=MaaiInput.Zero(),
        device=args.device,
        cpc_model=os.path.expanduser(args.cpc_model),
        local_model=args.local_model,
        return_p_bins=True,
    )
    bc_maai = None
    if args.bc_enabled:
        print(
            f"[load] BC Maai(mode={args.bc_mode}, lang={args.bc_lang}, "
            f"tail_sec={args.bc_tail_sec})"
        )
        bc_maai = Maai(
            mode=args.bc_mode,
            lang=args.bc_lang,
            frame_rate=args.frame_rate,
            context_len_sec=int(args.context_sec),
            audio_ch1=MaaiInput.Zero(),
            audio_ch2=MaaiInput.Zero(),
            device=args.device,
            cpc_model=os.path.expanduser(args.cpc_model),
            local_model=args.bc_local_model,
        )

    feat_dim = VAP_BC_FEAT_DIM if args.bc_enabled else VAP_FEAT_DIM
    feat_layout = VAP_BC_FEAT_LAYOUT if args.bc_enabled else VAP_FEAT_LAYOUT
    print(f"[info] frame_samples={frame_samples}, feat_dim={feat_dim}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "frame_rate": float(args.frame_rate),
        "chunk_ms": chunk_ms,
        "sample_rate": sample_rate,
        "feat_dim": feat_dim,
        "feat_layout": feat_layout,
        "lang": args.lang,
        "mode": args.mode,
        "context_sec": float(args.context_sec),
        "bc_enabled": bool(args.bc_enabled),
        "bc_lang": args.bc_lang,
        "bc_mode": args.bc_mode,
        "bc_tail_sec": float(args.bc_tail_sec),
        "frame_index_formula": "vap_frame = round(end_idx * chunk_ms * frame_rate / 1000)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    conv_ids = list_conv_ids(labels_dir)
    if args.max_convs:
        conv_ids = conv_ids[: args.max_convs]
    print(f"[run] {len(conv_ids)} conversations -> {out_dir}")

    t0 = time.time()
    done = 0
    for k, conv in enumerate(conv_ids):
        out_path = out_dir / f"{conv}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        wav = audio_dir / f"{conv}.wav"
        if not wav.exists():
            print(f"  [skip] missing audio: {wav}")
            continue
        try:
            audio2 = _load_conv_2ch(wav, sample_rate)
            feats = vap_features_for_conv(vap_maai, audio2, frame_samples)
            if bc_maai is not None:
                bc_values = bc_values_for_conv(bc_maai, audio2, frame_samples)
                feats = append_bc_tail_features(
                    feats,
                    bc_values,
                    frame_rate=args.frame_rate,
                    tail_sec=args.bc_tail_sec,
                )
            np.save(out_path, feats)
            done += 1
            if done <= 3 or done % 50 == 0:
                dur = audio2.shape[1] / sample_rate
                print(
                    f"  [{k + 1}/{len(conv_ids)}] {conv}: {feats.shape} "
                    f"audio={dur:.0f}s elapsed={time.time() - t0:.0f}s"
                )
        except Exception as exc:
            print(f"  [ERR] {conv}: {exc!r}")

    print(f"[done] processed={done}, cache={out_dir}, total={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
