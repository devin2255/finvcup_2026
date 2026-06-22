"""VAP 探针（按 MaAI 真实 API 写死）：验证 vap_mc_ch 能加载并跑通，dump 输出结构。

依据本地 clone 的 MaAI 源码（src/maai/model.py）确认的用法：
  from maai import Maai, MaaiInput
  maai = Maai(mode="vap_mc", lang="ch_kyoto", frame_rate=10, context_len_sec=5,
              audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(),
              device="cuda", cpc_model=<CPC权重>, local_model=<可选本地权重>)
  # 批量/离线特征：绕开流式 worker，直接用底层模型：
  e1, e2 = maai.vap.encode_audio(x1, x2)          # x:[1,1,T] @16k
  out, _ = maai.vap.forward(e1, e2, cache=None)   # dict
  out 含: p_now, p_future, vad, (p_bins, p_bins_now, p_bins_future)

p_now  = 未来 0–600ms 两说话人语音活动概率；p_future = 600–2000ms。这正是 turn-taking 先验。

用法（服务器上，已 clone MaAI 且有 CPC + vap_mc_ch_kyoto 权重）：
  python -m src.vap_probe --maai_dir ./MaAI \
      --wav /mnt/workspace/dorihue/finvcup_2026/train/audio/<某conv>.wav \
      --lang ch_kyoto --frame_rate 10 --context_sec 5 --device cuda \
      --cpc_model ~/.cache/cpc/60k_epoch4-d0f474de.pt
把输出贴回来，我据真实 out 形状写 VAPFeatureEncoder + 融合接入。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


def dump_value(name, v, depth=0):
    pad = "  " * (depth + 1)
    shape = getattr(v, "shape", None)
    if shape is not None:
        extra = ""
        try:
            flat = v.reshape(-1)
            if flat.numel() <= 8:
                extra = f"  values={[round(float(x),4) for x in flat.tolist()]}"
        except Exception:
            pass
        print(f"{pad}{name}: {type(v).__name__} shape={tuple(shape)} dtype={getattr(v,'dtype',None)}{extra}")
        return
    if isinstance(v, dict):
        print(f"{pad}{name}: dict keys={list(v.keys())}")
        for k, vv in v.items():
            dump_value(str(k), vv, depth + 1)
        return
    if isinstance(v, (list, tuple)):
        print(f"{pad}{name}: {type(v).__name__} len={len(v)}")
        for i, vv in enumerate(v[:6]):
            dump_value(f"[{i}]", vv, depth + 1)
        return
    print(f"{pad}{name}: {type(v).__name__} = {v}")


def load_2ch_tail(wav_path: str, context_sec: float, target_sr: int = 16000):
    """读 wav，重采样到 16k，取最后 context_sec 秒，返回两路 [1,1,T] tensor。"""
    import torchaudio
    w, sr = torchaudio.load(wav_path)  # [C, T]
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    elif w.shape[0] > 2:
        w = w[:2]
    if sr != target_sr:
        w = torchaudio.functional.resample(w, sr, target_sr)
    n = int(context_sec * target_sr)
    if w.shape[1] > n:
        w = w[:, -n:]
    pad = 320  # frame_contxt_padding，与 Maai.process 一致地前置静音
    w = torch.nn.functional.pad(w, (pad, 0))
    x1 = w[0][None, None, :].contiguous()
    x2 = w[1][None, None, :].contiguous()
    return x1, x2


def main():
    ap = argparse.ArgumentParser(description="VAP(vap_mc_ch) 加载+推理探针")
    ap.add_argument("--maai_dir", type=str, default="./MaAI", help="MaAI 仓库 clone 路径")
    ap.add_argument("--wav", type=str, default=None, help="一条双声道 wav（训练 conv 或测试 segment）")
    ap.add_argument("--lang", type=str, default="ch_kyoto", help="ch_kyoto=vap_mc_ch_kyoto；也可 ch / tri_kyoto")
    ap.add_argument("--mode", type=str, default="vap_mc")
    ap.add_argument("--frame_rate", type=float, default=10)
    ap.add_argument("--context_sec", type=float, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cpc_model", type=str, default=os.path.expanduser("~/.cache/cpc/60k_epoch4-d0f474de.pt"))
    ap.add_argument("--local_model", type=str, default=None, help="可选：本地 vap 权重 .pt（否则按 lang 自动下载）")
    args = ap.parse_args()

    # 让未 pip install 的 clone 也能 import：加入 <maai_dir>/src
    maai_dir = Path(args.maai_dir).resolve()
    for cand in (maai_dir / "src", maai_dir):
        if (cand / "maai").is_dir():
            sys.path.insert(0, str(cand))
            print(f"[path] 使用 maai 包: {cand / 'maai'}")
            break

    try:
        from maai import Maai, MaaiInput
    except Exception as e:
        print(f"[FATAL] import maai 失败: {e!r}")
        print("  确认 --maai_dir 指向 clone 根目录（其下应有 src/maai/）。")
        traceback.print_exc()
        return

    print(f"[load] Maai(mode={args.mode}, lang={args.lang}, frame_rate={args.frame_rate}, "
          f"context_len_sec={args.context_sec}, device={args.device})")
    print(f"[load] cpc_model={args.cpc_model}  local_model={args.local_model}")
    try:
        maai = Maai(
            mode=args.mode,
            lang=args.lang,
            frame_rate=args.frame_rate,
            context_len_sec=int(args.context_sec),
            audio_ch1=MaaiInput.Zero(),
            audio_ch2=MaaiInput.Zero(),
            device=args.device,
            cpc_model=args.cpc_model,
            local_model=args.local_model,
        )
    except Exception as e:
        print(f"[FATAL] 构造 Maai 失败: {e!r}")
        print("  常见原因：CPC 权重缺失(--cpc_model)、vap 权重需联网下载(或给 --local_model)、frame_rate 档位不被该 lang 支持。")
        traceback.print_exc()
        return
    print(f"[ok] 模型已加载；底层模型类型 = {type(maai.vap).__name__}")
    n_params = sum(p.numel() for p in maai.vap.parameters())
    print(f"[info] VAP 参数量 ≈ {n_params/1e6:.1f}M（用于核对 8B 预算）")

    if not args.wav:
        print("\n未提供 --wav，仅验证加载成功。给一条双声道 wav 可 dump 输出结构。")
        return

    print(f"\n[run] 读取并推理: {args.wav}")
    try:
        x1, x2 = load_2ch_tail(args.wav, args.context_sec)
        dev = maai.device
        x1, x2 = x1.to(dev), x2.to(dev)
        print(f"  输入每路 shape={tuple(x1.shape)} (含 320 前置 padding)")
        with torch.inference_mode():
            e1, e2 = maai.vap.encode_audio(x1, x2)
            print(f"  encode_audio -> e1 {tuple(e1.shape)}, e2 {tuple(e2.shape)} (应为 [1, n_frames, dim])")
            out, _ = maai.vap.forward(e1, e2, cache=None)
    except Exception:
        print("[FATAL] 推理失败:\n" + traceback.format_exc())
        return

    print("\n========== VAP 输出结构（out）==========")
    dump_value("out", out)

    # 展示"取上下文末端一帧"后我们会抽的特征
    print("\n========== 末端帧特征（我们要喂进融合层的 vap_feat 雏形）==========")
    for k in ("p_now", "p_future", "vad"):
        v = out.get(k) if isinstance(out, dict) else None
        if v is None:
            continue
        try:
            last = v[:, -1] if v.dim() >= 2 else v
            print(f"  {k} 末端帧: shape={tuple(last.shape)} values={[round(float(x),4) for x in last.reshape(-1).tolist()[:8]]}")
        except Exception as e:
            print(f"  {k}: 取末端帧失败 {e!r}")
    print("\n[完成] 把以上 out 结构贴回来，我据真实形状写 VAPFeatureEncoder + 融合接入（第5模态）。")


if __name__ == "__main__":
    main()
