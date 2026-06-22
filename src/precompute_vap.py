"""离线预计算 VAP(vap_mc_ch_kyoto) 逐帧话轮先验，缓存到盘，供训练/推理当"第5模态"特征。

为什么离线预计算：VAP 冻结，且 maai 依赖 torch>=2.6，与训练解耦最稳；每通对话只需
编码一次（复用 maai 的流式 process() + KV-cache），比"每个样本重编码 20s 窗口"快得多。

每帧特征(18 维)=
  [p_now(2), p_future(2), vad(2), p_bins(2x4=8), p_bins_now(2), p_bins_future(2)]
其中 p_now/p_future = 未来 0-600ms / 600-2000ms 两说话人语音活动概率（turn-taking 先验）。

VAP 帧率 = frame_rate(默认10Hz=100ms/帧)。训练/推理时按时间把样本的 end_idx(80ms/chunk)
映射到 vap 帧：vap_frame = round(end_idx * chunk_ms * frame_rate / 1000)。

用法（服务器，已下好 CPC + vap_mc_ch_kyoto 权重）：
  export HF_ENDPOINT=https://hf-mirror.com
  python -m src.precompute_vap --config configs/whisper_qwen0_6b_lmf_vap.yaml \
      --maai_dir ./MaAI --lang ch_kyoto --frame_rate 10 --context_sec 20 --device cuda \
      --cpc_model /mnt/workspace/dorihue/modelscope/60k_epoch4-d0f474de.pt \
      --out_dir /mnt/workspace/dorihue/finvcup_2026/.cache/vap_ch_kyoto
  # 先小规模验证： --max_convs 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import _read_wav_slice, list_conv_ids
from src.utils import load_config

VAP_FEAT_DIM = 18
FEAT_LAYOUT = ["p_now_0", "p_now_1", "p_future_0", "p_future_1", "vad_0", "vad_1",
               "p_bins_s0_b0", "p_bins_s0_b1", "p_bins_s0_b2", "p_bins_s0_b3",
               "p_bins_s1_b0", "p_bins_s1_b1", "p_bins_s1_b2", "p_bins_s1_b3",
               "p_bins_now_0", "p_bins_now_1", "p_bins_future_0", "p_bins_future_1"]


def _flat_result(r: dict) -> list:
    """把一帧的 result dict 拍平成 18 维。缺字段用 0 兜底。"""
    pn = [float(x) for x in r.get("p_now", [0.0, 0.0])]
    pf = [float(x) for x in r.get("p_future", [0.0, 0.0])]
    vd = [float(x) for x in r.get("vad", [0.0, 0.0])]
    pb = r.get("p_bins")
    pb_flat = []
    if pb is not None:
        for spk in pb:
            pb_flat.extend(float(x) for x in spk)
    pb_flat = (pb_flat + [0.0] * 8)[:8]
    pbn = [float(x) for x in r.get("p_bins_now", [0.0, 0.0])]
    pbf = [float(x) for x in r.get("p_bins_future", [0.0, 0.0])]
    return pn + pf + vd + pb_flat + pbn + pbf


def _load_conv_2ch(audio_path: Path, target_sr: int) -> np.ndarray:
    data, sr = _read_wav_slice(audio_path, 0, 10 ** 9)  # [T, C] in [-1,1]
    w = torch.from_numpy(data.T.copy())  # [C, T]
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    elif w.shape[0] > 2:
        w = w[:2]
    if sr != target_sr:
        import torchaudio
        w = torchaudio.functional.resample(w, sr, target_sr)
    return w.numpy().astype(np.float32)


def vap_features_for_conv(maai, audio2: np.ndarray, frame_samples: int) -> np.ndarray:
    """对整通 [2,T] 逐帧跑 VAP，返回 [F, 18]。复用 maai.process()（KV-cache）。"""
    maai.reset_runtime_state()
    q = maai.result_dict_queue
    while not q.empty():
        q.get()
    T = audio2.shape[1]
    feats = []
    for i in range(0, T, frame_samples):
        c1 = np.ascontiguousarray(audio2[0, i:i + frame_samples])
        c2 = np.ascontiguousarray(audio2[1, i:i + frame_samples])
        if c1.shape[0] == 0:
            break
        maai.process(c1, c2)
        while not q.empty():
            feats.append(_flat_result(q.get()))
    if not feats:
        return np.zeros((0, VAP_FEAT_DIM), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="预计算 VAP 逐帧话轮先验特征并缓存")
    ap.add_argument("--config", type=str, required=True, help="取 paths/sample_rate/chunk_ms")
    ap.add_argument("--maai_dir", type=str, default="./MaAI")
    ap.add_argument("--lang", type=str, default="ch_kyoto")
    ap.add_argument("--mode", type=str, default="vap_mc")
    ap.add_argument("--frame_rate", type=float, default=10)
    ap.add_argument("--context_sec", type=float, default=20)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cpc_model", type=str, default="~/.cache/cpc/60k_epoch4-d0f474de.pt")
    ap.add_argument("--local_model", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True, help="特征缓存目录")
    ap.add_argument("--max_convs", type=int, default=None, help="只处理前 N 通（验证用）")
    ap.add_argument("--overwrite", action="store_true", help="已存在也重算")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sample_rate = int(cfg["sample_rate"])
    chunk_ms = int(cfg["chunk_ms"])
    audio_dir = Path(cfg["paths"]["train_audio_dir"])
    labels_dir = Path(cfg["paths"]["train_labels_dir"])

    maai_dir = Path(args.maai_dir).resolve()
    for cand in (maai_dir / "src", maai_dir):
        if (cand / "maai").is_dir():
            sys.path.insert(0, str(cand))
            break
    import os
    from maai import Maai, MaaiInput

    print(f"[load] Maai(mode={args.mode}, lang={args.lang}, frame_rate={args.frame_rate}, context_len_sec={args.context_sec})")
    maai = Maai(
        mode=args.mode, lang=args.lang, frame_rate=args.frame_rate,
        context_len_sec=int(args.context_sec),
        audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(),
        device=args.device, cpc_model=os.path.expanduser(args.cpc_model),
        local_model=args.local_model, return_p_bins=True,
    )
    frame_samples = int(round(sample_rate / float(args.frame_rate)))
    print(f"[info] frame_samples={frame_samples} (={1000/args.frame_rate:.0f}ms/帧), feat_dim={VAP_FEAT_DIM}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # meta：供训练/推理侧做 end_idx -> vap_frame 映射
    meta = {
        "frame_rate": float(args.frame_rate), "chunk_ms": chunk_ms, "sample_rate": sample_rate,
        "feat_dim": VAP_FEAT_DIM, "feat_layout": FEAT_LAYOUT,
        "lang": args.lang, "mode": args.mode, "context_sec": float(args.context_sec),
        "frame_index_formula": "vap_frame = round(end_idx * chunk_ms * frame_rate / 1000)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    conv_ids = list_conv_ids(labels_dir)
    if args.max_convs:
        conv_ids = conv_ids[: args.max_convs]
    print(f"[run] {len(conv_ids)} 通对话 -> {out_dir}")

    t0 = time.time()
    done = 0
    for k, conv in enumerate(conv_ids):
        out_path = out_dir / f"{conv}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        wav = audio_dir / f"{conv}.wav"
        if not wav.exists():
            print(f"  [skip] 无音频: {wav}")
            continue
        try:
            audio2 = _load_conv_2ch(wav, sample_rate)
            feats = vap_features_for_conv(maai, audio2, frame_samples)
            np.save(out_path, feats)
            done += 1
            if done <= 3 or done % 50 == 0:
                dur = audio2.shape[1] / sample_rate
                print(f"  [{k+1}/{len(conv_ids)}] {conv}: {feats.shape} (音频{dur:.0f}s) "
                      f"耗时累计 {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  [ERR] {conv}: {e!r}")

    print(f"[完成] 处理 {done} 通，缓存在 {out_dir}（meta.json 已写）。总耗时 {time.time()-t0:.0f}s")
    print("下一步：训练侧 dataset 读该缓存按 end_idx 映射取帧 -> 融合第5模态。")


if __name__ == "__main__":
    main()
