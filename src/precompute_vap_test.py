"""Precompute per-segment VAP windows for the test/submission set.

Default output is [window, 18]. With --bc_enabled, output is [window, 21]
using causal BC tail features appended to each VAP frame.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import _read_wav_slice
from src.vap_window import vap_bc_last_n_frames, vap_last_n_frames

_W_MAAI = None
_W_BC_MAAI = None
_W_CFG = None


def _load_seg_2ch(audio_path: Path, target_sr: int) -> np.ndarray:
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


def _add_maai_to_path(maai_dir: str) -> None:
    maai_path = Path(maai_dir)
    for cand in (maai_path / "src", maai_path):
        if (cand / "maai").is_dir():
            cand_s = str(cand)
            if cand_s not in sys.path:
                sys.path.insert(0, cand_s)
            return


def _worker_init(init_args: dict):
    global _W_MAAI, _W_BC_MAAI, _W_CFG
    _add_maai_to_path(init_args["maai_dir"])
    from maai import Maai, MaaiInput

    _W_MAAI = Maai(
        mode=init_args["mode"],
        lang=init_args["lang"],
        frame_rate=init_args["frame_rate"],
        context_len_sec=int(init_args["context_sec"]),
        audio_ch1=MaaiInput.Zero(),
        audio_ch2=MaaiInput.Zero(),
        device=init_args["device"],
        cpc_model=os.path.expanduser(init_args["cpc_model"]),
        local_model=init_args["vap_local_model"],
        return_p_bins=True,
    )
    _W_BC_MAAI = None
    if init_args["bc_enabled"]:
        _W_BC_MAAI = Maai(
            mode=init_args["bc_mode"],
            lang=init_args["bc_lang"],
            frame_rate=init_args["frame_rate"],
            context_len_sec=int(init_args["context_sec"]),
            audio_ch1=MaaiInput.Zero(),
            audio_ch2=MaaiInput.Zero(),
            device=init_args["device"],
            cpc_model=os.path.expanduser(init_args["cpc_model"]),
            local_model=init_args["bc_local_model"],
        )
    _W_CFG = {
        "frame_samples": int(round(init_args["sample_rate"] / float(init_args["frame_rate"]))),
        "sample_rate": int(init_args["sample_rate"]),
        "audio_dir": init_args["audio_dir"],
        "out_dir": init_args["out_dir"],
        "overwrite": bool(init_args["overwrite"]),
        "window": int(init_args["window"]),
        "frame_rate": float(init_args["frame_rate"]),
        "bc_enabled": bool(init_args["bc_enabled"]),
        "bc_tail_sec": float(init_args["bc_tail_sec"]),
    }
    mode = "VAP+BC" if _W_BC_MAAI is not None else "VAP"
    print(
        f"[worker pid={os.getpid()}] {mode} loaded, "
        f"frame_samples={_W_CFG['frame_samples']}, window={_W_CFG['window']}",
        flush=True,
    )


def _worker_process_seg(sid: str) -> tuple[str, bool, str]:
    global _W_MAAI, _W_BC_MAAI, _W_CFG
    out_path = Path(_W_CFG["out_dir"]) / f"{sid}.npy"
    if out_path.exists() and not _W_CFG["overwrite"]:
        return sid, True, "skip-exists"
    try:
        audio2 = _load_seg_2ch(Path(_W_CFG["audio_dir"]) / f"{sid}.wav", _W_CFG["sample_rate"])
        if _W_BC_MAAI is not None:
            feat = vap_bc_last_n_frames(
                _W_MAAI,
                _W_BC_MAAI,
                audio2,
                frame_samples=_W_CFG["frame_samples"],
                N=_W_CFG["window"],
                frame_rate=_W_CFG["frame_rate"],
                bc_tail_sec=_W_CFG["bc_tail_sec"],
            )
        else:
            feat = vap_last_n_frames(_W_MAAI, audio2, _W_CFG["frame_samples"], _W_CFG["window"])
        np.save(out_path, feat)
        return sid, True, f"ok shape={tuple(feat.shape)}"
    except Exception as exc:
        return sid, False, repr(exc)


def main():
    ap = argparse.ArgumentParser(description="precompute per-segment VAP windows")
    ap.add_argument("--maai_dir", type=str, required=True)
    ap.add_argument("--lang", type=str, default="ch_kyoto")
    ap.add_argument("--mode", type=str, default="vap_mc")
    ap.add_argument("--frame_rate", type=float, default=10)
    ap.add_argument("--context_sec", type=float, default=20)
    ap.add_argument("--cpc_model", type=str, required=True)
    ap.add_argument("--vap_local_model", type=str, required=True)
    ap.add_argument("--bc_enabled", action="store_true", help="append causal MaAI BC tail features")
    ap.add_argument("--bc_lang", type=str, default="ch")
    ap.add_argument("--bc_mode", type=str, default="bc")
    ap.add_argument("--bc_local_model", type=str, default=None)
    ap.add_argument("--bc_tail_sec", type=float, default=2.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--test_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_segments", type=int, default=None)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    audio_dir = Path(args.test_root) / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"test audio dir not found: {audio_dir}")
    seg_ids = sorted([p.stem for p in audio_dir.glob("*.wav")])
    if args.max_segments:
        seg_ids = seg_ids[: args.max_segments]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_args = dict(
        maai_dir=str(Path(args.maai_dir).resolve()),
        lang=args.lang,
        mode=args.mode,
        frame_rate=args.frame_rate,
        context_sec=args.context_sec,
        cpc_model=args.cpc_model,
        vap_local_model=args.vap_local_model,
        bc_enabled=bool(args.bc_enabled),
        bc_lang=args.bc_lang,
        bc_mode=args.bc_mode,
        bc_local_model=args.bc_local_model,
        bc_tail_sec=args.bc_tail_sec,
        device=args.device,
        sample_rate=args.sample_rate,
        audio_dir=str(audio_dir),
        out_dir=str(out_dir),
        overwrite=bool(args.overwrite),
        window=int(args.window),
    )

    print(
        f"[run] segments={len(seg_ids)} workers={args.workers} window={args.window} "
        f"bc={bool(args.bc_enabled)} -> {out_dir}",
        flush=True,
    )
    t0 = time.time()

    if args.workers <= 1:
        _worker_init(init_args)
        done = ok = 0
        for sid in seg_ids:
            sid_, success, msg = _worker_process_seg(sid)
            done += 1
            ok += int(success)
            if not success:
                print(f"  [ERR] {sid_}: {msg}", flush=True)
            if done <= 3 or done % 100 == 0:
                print(
                    f"  [{done}/{len(seg_ids)}] {sid_}: {msg} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
        print(f"[done] wrote {ok}/{done} segments. total {time.time() - t0:.0f}s", flush=True)
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_worker_init, initargs=(init_args,)) as pool:
        done = ok = 0
        for sid_, success, msg in pool.imap_unordered(_worker_process_seg, seg_ids, chunksize=4):
            done += 1
            ok += int(success)
            if not success:
                print(f"  [ERR] {sid_}: {msg}", flush=True)
            if done <= 5 or done % 100 == 0:
                rate = done / max(1.0, time.time() - t0)
                eta = (len(seg_ids) - done) / max(1e-6, rate)
                print(
                    f"  [{done}/{len(seg_ids)}] {sid_}: {msg} "
                    f"elapsed={time.time() - t0:.0f}s ETA={eta:.0f}s",
                    flush=True,
                )

    print(f"[done] wrote {ok}/{done} segments. total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
