"""复赛测试集 VAP 预计算：为 /xydata/audio/<seg_id>.wav 每条样本算 18 维 VAP 特征。

输出：<out_dir>/<seg_id>.npy（dtype float32, shape (18,)）—— 该 segment 流式跑完后的最后一帧。

性能：单进程在 GPU 上约 ~80 帧/秒、4 sec/段，1000 段约 66 分钟超出 60 分钟硬限。
本脚本支持 `--workers N` 多进程并行，每个进程各持一份 MaAI 模型共用同一 GPU。
对 1000 段实测 N=4 时约 18-22 分钟（取决于 GPU 占用率）。
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
from src.vap_window import vap_last_n_frames

VAP_FEAT_DIM = 18


def _load_seg_2ch(audio_path: Path, target_sr: int) -> np.ndarray:
    data, sr = _read_wav_slice(audio_path, 0, 10 ** 9)
    w = torch.from_numpy(data.T.copy())
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    elif w.shape[0] > 2:
        w = w[:2]
    if sr != target_sr:
        import torchaudio
        w = torchaudio.functional.resample(w, sr, target_sr)
    return w.numpy().astype(np.float32)


# ============================================================
# 多进程 worker：每个 worker 进程持一份 MaAI 实例（同一 GPU 共享）
# ============================================================
_W_MAAI = None  # worker-local Maai 实例
_W_CFG = None   # {frame_samples, sample_rate, audio_dir, out_dir, overwrite}


def _worker_init(init_args: dict):
    """spawn 子进程的初始化：加载 MaAI 模型（每进程一份）。"""
    global _W_MAAI, _W_CFG
    maai_dir = Path(init_args["maai_dir"])
    for cand in (maai_dir / "src", maai_dir):
        if (cand / "maai").is_dir():
            sys.path.insert(0, str(cand))
            break
    from maai import Maai, MaaiInput

    _W_MAAI = Maai(
        mode=init_args["mode"], lang=init_args["lang"],
        frame_rate=init_args["frame_rate"],
        context_len_sec=int(init_args["context_sec"]),
        audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(),
        device=init_args["device"],
        cpc_model=os.path.expanduser(init_args["cpc_model"]),
        local_model=init_args["vap_local_model"], return_p_bins=True,
    )
    _W_CFG = {
        "frame_samples": int(round(init_args["sample_rate"] / float(init_args["frame_rate"]))),
        "sample_rate": init_args["sample_rate"],
        "audio_dir": init_args["audio_dir"],
        "out_dir": init_args["out_dir"],
        "overwrite": init_args["overwrite"],
        "window": int(init_args["window"]),
    }
    print(f"[worker pid={os.getpid()}] MaAI loaded, frame_samples={_W_CFG['frame_samples']}",
          flush=True)


def _worker_process_seg(sid: str) -> tuple[str, bool, str]:
    """单段处理：返回 (seg_id, ok, msg)。"""
    global _W_MAAI, _W_CFG
    out_path = Path(_W_CFG["out_dir"]) / f"{sid}.npy"
    if out_path.exists() and not _W_CFG["overwrite"]:
        return sid, True, "skip-exists"
    try:
        audio2 = _load_seg_2ch(Path(_W_CFG["audio_dir"]) / f"{sid}.wav", _W_CFG["sample_rate"])
        feat = vap_last_n_frames(_W_MAAI, audio2, _W_CFG["frame_samples"], _W_CFG["window"])
        np.save(out_path, feat)
        return sid, True, "ok"
    except Exception as e:
        return sid, False, repr(e)


# ============================================================
# 入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="per-segment VAP precompute for test set")
    ap.add_argument("--maai_dir", type=str, required=True)
    ap.add_argument("--lang", type=str, default="ch_kyoto")
    ap.add_argument("--mode", type=str, default="vap_mc")
    ap.add_argument("--frame_rate", type=float, default=10)
    ap.add_argument("--context_sec", type=float, default=20)
    ap.add_argument("--cpc_model", type=str, required=True)
    ap.add_argument("--vap_local_model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--test_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_segments", type=int, default=None)
    ap.add_argument("--window", type=int, default=20, help="保留最后 N 帧 VAP 特征 -> (N,18)")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 worker 进程数。每进程占一份 MaAI(GPU 共享)。建议 2-4。")
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
        lang=args.lang, mode=args.mode,
        frame_rate=args.frame_rate, context_sec=args.context_sec,
        cpc_model=args.cpc_model, vap_local_model=args.vap_local_model,
        device=args.device, sample_rate=args.sample_rate,
        audio_dir=str(audio_dir), out_dir=str(out_dir),
        overwrite=bool(args.overwrite),
        window=int(args.window),
    )

    print(f"[run] segments={len(seg_ids)} workers={args.workers} "
          f"lang={args.lang} ctx={args.context_sec}s -> {out_dir}", flush=True)
    t0 = time.time()

    if args.workers <= 1:
        # 单进程：原地跑（节省 spawn 开销，便于调试）
        _worker_init(init_args)
        done = ok = 0
        for k, sid in enumerate(seg_ids):
            sid_, success, msg = _worker_process_seg(sid)
            done += 1
            if success:
                ok += 1
            else:
                print(f"  [ERR] {sid_}: {msg}", flush=True)
            if done <= 3 or done % 100 == 0:
                print(f"  [{done}/{len(seg_ids)}] elapsed={time.time()-t0:.0f}s "
                      f"rate={done/max(1, time.time()-t0):.2f} seg/s", flush=True)
        print(f"[done] wrote {ok}/{done} segments. total {time.time()-t0:.0f}s", flush=True)
        return

    # 多进程：CUDA 必须用 spawn
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_worker_init, initargs=(init_args,)) as pool:
        done = ok = 0
        # imap_unordered: 完成即返回，便于打印进度
        for sid_, success, msg in pool.imap_unordered(_worker_process_seg, seg_ids, chunksize=4):
            done += 1
            if success:
                ok += 1
            else:
                print(f"  [ERR] {sid_}: {msg}", flush=True)
            if done <= 5 or done % 100 == 0:
                rate = done / max(1, time.time() - t0)
                eta = (len(seg_ids) - done) / max(1e-6, rate)
                print(f"  [{done}/{len(seg_ids)}] elapsed={time.time()-t0:.0f}s "
                      f"rate={rate:.2f} seg/s ETA={eta:.0f}s", flush=True)

    print(f"[done] wrote {ok}/{done} segments. total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
