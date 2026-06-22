"""VAD 自检：核对 VAP 辅助头用的"能量级 VAD"在真实数据上是否合理。

VAP 目标(src/data/dataset.py: _future_va_grid)用的判据与本脚本完全一致：
    每 chunk 对数能量 loge = log(mean(frame^2) + 1e-8)
    每声道自适应噪声底 floor = 该声道 loge 的 20% 分位
    有声(VA=1)  <=>  loge > floor + vad_log_offset

本脚本对若干整通对话计算逐 chunk 双声道 VA，并与事件标签 C/T/BC/I/NA 对照，
帮助你挑一个合理的 vad_log_offset（写进 configs/...lmf_vap.yaml 的 vap_aux.vad_log_offset）。

判断标准（VAD 阈值合理时应满足）：
  - NA(静音)    -> 双声道都"无声"(neither) 占比高
  - I(打断/重叠) -> 双声道"都有声"(both) 占比高
  - 非 NA       -> 至少一路有声；"任一有声 vs 非NA" 的 F1 越高越好

用法（在服务器、有训练数据处运行）：
  python -m src.check_vad --config configs/whisper_qwen0_6b_lmf_vap.yaml
  python -m src.check_vad --config configs/whisper_qwen0_6b_lmf_vap.yaml \
      --num_convs 12 --offsets 1.0,1.5,2.0,2.5,3.0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.data.dataset import _read_wav_slice, list_conv_ids
from src.utils import load_config, set_env_paths


def parse_args():
    p = argparse.ArgumentParser(description="VAP 用的能量级 VAD 自检（与事件标签对照）")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--num_convs", type=int, default=8, help="采样多少通对话")
    p.add_argument("--max_seconds_per_conv", type=float, default=600.0, help="每通最多读多少秒")
    p.add_argument("--offsets", type=str, default="1.0,1.5,2.0,2.5,3.0", help="待扫的 vad_log_offset 列表")
    p.add_argument("--floor_quantile", type=float, default=0.2, help="噪声底分位（与训练一致=0.2）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chunk_log_energy(x: np.ndarray, spc: int) -> np.ndarray:
    """镜像 TurnTakingTrainDataset._chunk_log_energy：log(mean(frame^2)+1e-8)。"""
    n = x.shape[0] // spc
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    frames = x[: n * spc].reshape(n, spc).astype(np.float64)
    return np.log(np.mean(frames ** 2, axis=1) + 1e-8)


def load_conv_2ch(audio_path: Path, sample_rate: int, max_seconds: float) -> np.ndarray | None:
    """读整通(截断 max_seconds)，重采样到 sample_rate，返回 [2, T] in [-1,1]。"""
    audio, src_sr = _read_wav_slice(audio_path, 0, int(max_seconds * 1000))  # [T, C]
    wave = torch.from_numpy(audio.T)  # [C, T]
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    elif wave.shape[0] > 2:
        wave = wave[:2]
    if src_sr != sample_rate:
        wave = torchaudio.functional.resample(wave, src_sr, sample_rate)
    return wave.numpy()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)

    paths = cfg["paths"]
    labels_dir = Path(paths["train_labels_dir"])
    audio_dir = Path(paths["train_audio_dir"])
    sample_rate = int(cfg["sample_rate"])
    chunk_ms = int(cfg["chunk_ms"])
    spc = int(chunk_ms * sample_rate / 1000)
    offsets = [float(x) for x in args.offsets.split(",") if x.strip()]
    cfg_offset = float(cfg.get("vap_aux", {}).get("vad_log_offset", 2.0))

    label_names = ["c", "t", "bc", "i", "na"]  # id 0..4 = C,T,BC,I,NA
    na_id = 4
    i_id = 3

    conv_ids = list_conv_ids(labels_dir)
    if not conv_ids:
        raise RuntimeError(f"{labels_dir} 下没有 .npy 标签文件")
    rng = random.Random(args.seed)
    sampled = rng.sample(conv_ids, min(args.num_convs, len(conv_ids)))
    print(f"采样 {len(sampled)} 通对话；spc={spc} (chunk={chunk_ms}ms @ {sample_rate}Hz)")
    print(f"配置当前 vad_log_offset = {cfg_offset}\n")

    # cat[offset][label_id] = [neither, ch0only, ch1only, both]
    cat = {o: np.zeros((5, 4), dtype=np.int64) for o in offsets}
    total_chunks = 0

    for conv in sampled:
        labels = np.load(labels_dir / f"{conv}.npy")
        wave = load_conv_2ch(audio_dir / f"{conv}.wav", sample_rate, args.max_seconds_per_conv)
        if wave is None:
            continue
        loge = [chunk_log_energy(wave[ch], spc) for ch in range(2)]
        floor = [np.quantile(loge[ch], args.floor_quantile) if loge[ch].size else -18.0 for ch in range(2)]
        n = min(len(labels), loge[0].shape[0], loge[1].shape[0])
        if n == 0:
            continue
        total_chunks += n
        lab = labels[:n].astype(int)
        for o in offsets:
            va0 = (loge[0][:n] > floor[0] + o).astype(int)
            va1 = (loge[1][:n] > floor[1] + o).astype(int)
            code = va0 + 2 * va1  # 0=neither,1=ch0only,2=ch1only,3=both
            for lid in range(5):
                m = lab == lid
                if not m.any():
                    continue
                c = code[m]
                cat[o][lid, 0] += int((c == 0).sum())
                cat[o][lid, 1] += int((c == 1).sum())
                cat[o][lid, 2] += int((c == 2).sum())
                cat[o][lid, 3] += int((c == 3).sum())

    print(f"总 chunk 数 = {total_chunks}\n")
    print("=== 阈值扫描（每行一个 vad_log_offset）===")
    print(f"{'offset':>7} | {'F1(any↔¬NA)':>12} | {'P(neither|NA)':>13} | {'P(both|I)':>10} | {'ch0率':>6} {'ch1率':>6}")
    print("-" * 70)
    best = (None, -1.0)
    for o in offsets:
        m = cat[o]
        per_class_total = m.sum(axis=1)  # [5]
        any_active = m[:, 1] + m[:, 2] + m[:, 3]
        neither = m[:, 0]
        non_na_ids = [0, 1, 2, 3]
        tp = sum(int(any_active[c]) for c in non_na_ids)
        fn = sum(int(neither[c]) for c in non_na_ids)
        fp = int(any_active[na_id])
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        p_neither_na = neither[na_id] / max(1, per_class_total[na_id])
        p_both_i = m[i_id, 3] / max(1, per_class_total[i_id])
        ch0 = (m[:, 1].sum() + m[:, 3].sum()) / max(1, total_chunks)
        ch1 = (m[:, 2].sum() + m[:, 3].sum()) / max(1, total_chunks)
        tag = "  <- 当前" if abs(o - cfg_offset) < 1e-6 else ""
        print(f"{o:>7.2f} | {f1:>12.3f} | {p_neither_na:>13.3f} | {p_both_i:>10.3f} | {ch0:>6.3f} {ch1:>6.3f}{tag}")
        if f1 > best[1]:
            best = (o, f1)

    rec_o = best[0]
    print(f"\n>>> 推荐 vad_log_offset = {rec_o}（'任一有声 ↔ 非NA' 的 F1 最高 = {best[1]:.3f}）")
    print("    判断口径：F1 高 + P(neither|NA) 高 + P(both|I) 高，三者兼顾。\n")

    # 推荐阈值下的逐类 VA 模式（用于人工核对语义是否对得上）
    m = cat[rec_o]
    print(f"=== offset={rec_o} 时各事件类的 VA 模式占比 ===")
    print(f"{'类':>4} | {'neither':>8} {'ch0only':>8} {'ch1only':>8} {'both':>8} | 期望")
    print("-" * 62)
    expect = {"c": "恰一路有声", "t": "过渡(混合)", "bc": "both 偏多", "i": "both 高", "na": "neither 高"}
    for lid, name in enumerate(label_names):
        tot = max(1, m[lid].sum())
        row = m[lid] / tot
        print(f"{name.upper():>4} | {row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f} {row[3]:>8.3f} | {expect[name]}")
    print("\n核对要点：NA 行 neither 应最高；I 行 both 应明显高；C 行应以 ch0only/ch1only 为主。")
    print(f"满意后把 vap_aux.vad_log_offset 设为 {rec_o}（或你判断更合理的值），再跑 run_train_vap.sh。")


if __name__ == "__main__":
    main()
