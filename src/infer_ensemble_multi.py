"""跨模型软投票推理：结构不同的多组模型（如 vapwin / dualch / dualvap_bcd）概率级融合。

与 src.infer_ensemble 的区别：
- infer_ensemble 假设所有成员共享同一个 config（同一架构、同一 VAP 窗口），
  只能做"同配置多 epoch"集成；
- 本脚本吃一个 JSON spec，每个 model group 有**自己的 config**（决定架构：
  stereo_branch 开关、vap_feat window=1 单帧 / window=20 窗口等）、自己的
  manifest/checkpoint 与可选权重，组内先做概率均值，组间再做加权概率均值，
  最后用同权重的"阈值加权均值"二值化。

spec JSON 格式：
{
  "models": [
    {
      "name": "vapwin",                       # 仅日志用
      "config": "configs/xxx.yaml",           # 该组的模型/数据配置
      "manifest": "/path/ensemble_manifest.json",   # 可选；默认 <logs_dir>/ensemble_manifest.json
      "checkpoints_dir": "/path/ckpts",       # 可选；默认 config 的 paths.checkpoints_dir
      "checkpoints": ["/path/a.pt"],          # 可选；直接指定成员文件（跳过 manifest）
      "topk": 3,                              # 可选；只用 manifest 前 K 个成员
      "weight": 1.0,                          # 可选；组间权重，默认 1.0
      "vap_feat_dir": "/path/vap_test"        # 可选；该组的测试 VAP 缓存目录
    }
  ]
}

注意：
- 各组 labels.multi_targets 必须一致（输出列一致）。
- 单帧模型（旧 dualch，config 里 vap_feat.window: 1）与窗口模型可共用同一份
  window>=N 的测试缓存：dataset 输出 [N,18]，模型侧 window<=1 时自动取末帧。
- 60 分钟预算：每组每成员各过一遍测试集，总耗时 ~ 成员总数 × 单模型耗时，
  提交时用 topk 控制成员数。

用法：
  python -m src.infer_ensemble_multi \
      --spec configs/submit_multi_spec.json \
      --test_root /xydata \
      --output_csv /app/submit/submit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import TurnTakingTestDataset, build_collate_fn
from src.models import MultimodalTurnTakingModel
from src.utils import load_config, set_env_paths


def parse_args():
    p = argparse.ArgumentParser(description="跨模型软投票推理（每组模型有独立 config/架构）")
    p.add_argument("--spec", type=str, required=True, help="模型组 spec JSON 路径")
    p.add_argument("--test_root", type=str, required=True, help="测试数据根目录")
    p.add_argument("--output_csv", type=str, required=True, help="输出 pred.csv 路径")
    p.add_argument("--batch_size", type=int, default=None, help="默认取各组 config train.eval_batch_size")
    p.add_argument("--max_segments", type=int, default=None, help="仅处理前 N 条（冒烟测试）")
    p.add_argument("--default_threshold", type=float, default=0.5)
    return p.parse_args()


def _load_group_members(group: dict, cfg: dict, default_thr: float) -> list[dict]:
    """解析该组的成员 checkpoint 列表：[{path, thresholds?}, ...]。"""
    if group.get("checkpoints"):
        return [{"name": Path(c).name, "path": Path(c)} for c in group["checkpoints"]]

    paths = cfg["paths"]
    ckpt_dir = Path(group.get("checkpoints_dir") or paths["checkpoints_dir"])
    manifest_path = Path(group.get("manifest") or (Path(paths["logs_dir"]) / "ensemble_manifest.json"))
    if not manifest_path.exists():
        raise FileNotFoundError(f"[{group.get('name')}] 找不到 manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        members = json.load(f).get("members", [])
    if group.get("topk") is not None:
        members = members[: int(group["topk"])]
    resolved = []
    for m in members:
        mp = ckpt_dir / m["name"]
        if not mp.exists():
            print(f"[WARN][{group.get('name')}] 成员缺失，跳过: {mp}")
            continue
        resolved.append({**m, "path": mp})
    if not resolved:
        raise RuntimeError(f"[{group.get('name')}] 没有可用成员（目录 {ckpt_dir}）")
    return resolved


def _member_thresholds(member: dict, ckpt: dict, label_cols: list[str], default: float) -> np.ndarray:
    thr = member.get("thresholds")
    if not isinstance(thr, dict):
        thr = ckpt.get("thresholds") if isinstance(ckpt, dict) else None
    thr = thr if isinstance(thr, dict) else {}
    return np.array([float(thr.get(name, default)) for name in label_cols], dtype=np.float64)


def _build_loader(cfg: dict, group: dict, test_root: Path, batch_size: int | None):
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    vf_cfg = cfg.get("vap_feat", {}) or {}
    vap_feat_dir = group.get("vap_feat_dir")
    if vap_feat_dir is None and bool(vf_cfg.get("enabled", False)):
        vap_feat_dir = vf_cfg.get("test_cache_dir") or None
    # 单帧模型（window<=1）也按窗口读缓存，模型 forward 里自动取末帧
    ds_window = max(1, int(vf_cfg.get("window", 20)))
    ds = TurnTakingTestDataset(
        test_root=test_root,
        sample_rate=int(cfg["sample_rate"]),
        context_chunks=int(cfg["context_chunks"]),
        vap_feat_dir=vap_feat_dir,
        vap_feat_dim=int(vf_cfg.get("feat_dim", 18)),
        vap_window=ds_window,
    )
    bs = int(batch_size or cfg["train"]["eval_batch_size"])
    # num_workers=0：评测容器 /dev/shm 只有 64MB，多 worker 会 Bus error（同 infer_ensemble）
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=True)
    return loader, vap_feat_dir


def main():
    args = parse_args()
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    groups = spec.get("models", [])
    if not groups:
        raise RuntimeError(f"spec 中没有任何模型组: {args.spec}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_root = Path(args.test_root)

    label_cols: list[str] | None = None
    n_labels = 0

    # 组间加权累加：final_p = Σ w_g * mean_p_g / Σ w_g；阈值同权重加权。
    sum_w = 0.0
    wsum_probs: dict[str, np.ndarray] = {}
    wsum_thr: np.ndarray | None = None
    seg_order: list[str] = []

    for gi, group in enumerate(groups):
        gname = group.get("name", f"group{gi}")
        cfg = load_config(group["config"])
        set_env_paths(cfg)
        cols = [x.lower() for x in cfg["labels"]["multi_targets"]]
        if label_cols is None:
            label_cols = cols
            n_labels = len(cols)
            wsum_thr = np.zeros(n_labels, dtype=np.float64)
        elif cols != label_cols:
            raise RuntimeError(f"[{gname}] multi_targets 与首组不一致: {cols} vs {label_cols}")

        members = _load_group_members(group, cfg, args.default_threshold)
        weight = float(group.get("weight", 1.0))
        loader, vap_dir = _build_loader(cfg, group, test_root, args.batch_size)
        use_amp = bool(cfg["train"].get("use_amp", False))
        print(
            f"[multi] group={gname} config={group['config']} members={len(members)} "
            f"weight={weight} vap_feat_dir={vap_dir}"
        )

        g_sum_probs: dict[str, np.ndarray] = {}
        g_sum_thr = np.zeros(n_labels, dtype=np.float64)

        for mi, member in enumerate(members):
            ckpt = torch.load(member["path"], map_location="cpu")
            g_sum_thr += _member_thresholds(member, ckpt, label_cols, args.default_threshold)

            model = MultimodalTurnTakingModel(cfg).to(device)
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            if unexpected:
                print(f"[WARN][{gname}] {member['name']} 含未预期键 {len(unexpected)} 个（已忽略）")
            model.eval()

            done = 0
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"{gname} {mi + 1}/{len(members)} {member['name']}"):
                    waveform = batch["waveform"].to(device, non_blocking=True)
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    context_labels = batch["context_labels"].to(device, non_blocking=True)
                    segment_ids = batch["segment_id"]
                    vap_feat = batch.get("vap_feat")
                    if vap_feat is not None:
                        vap_feat = vap_feat.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(
                            waveform=waveform,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            context_labels=context_labels,
                            vap_feat=vap_feat,
                        )
                    probs = torch.sigmoid(logits).float().cpu().numpy()
                    if probs.ndim == 1:
                        probs = probs.reshape(-1, 1)
                    for i, seg_id in enumerate(segment_ids):
                        p = probs[i]
                        if p.shape[0] != n_labels:
                            raise RuntimeError(f"[{gname}] logits dim {p.shape[0]} != {n_labels}")
                        if seg_id not in g_sum_probs:
                            g_sum_probs[seg_id] = np.zeros(n_labels, dtype=np.float64)
                        g_sum_probs[seg_id] += p.astype(np.float64)
                        done += 1
                        if args.max_segments is not None and done >= args.max_segments:
                            break
                    if args.max_segments is not None and done >= args.max_segments:
                        break

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n_mem = len(members)
        g_mean_thr = g_sum_thr / n_mem
        print(f"[multi] group={gname} mean_thr="
              f"{ {name: round(float(g_mean_thr[j]), 3) for j, name in enumerate(label_cols)} }")
        for seg_id, sp in g_sum_probs.items():
            if seg_id not in wsum_probs:
                wsum_probs[seg_id] = np.zeros(n_labels, dtype=np.float64)
                seg_order.append(seg_id)
            wsum_probs[seg_id] += weight * (sp / n_mem)
        wsum_thr += weight * g_mean_thr
        sum_w += weight

    assert label_cols is not None and wsum_thr is not None
    final_thr = wsum_thr / sum_w
    print(f"[multi] final weighted thresholds="
          f"{ {name: round(float(final_thr[j]), 3) for j, name in enumerate(label_cols)} }")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["segment_id"] + label_cols
    rows = []
    pos_count = np.zeros(n_labels, dtype=np.int64)
    for seg_id in seg_order:
        mean_p = wsum_probs[seg_id] / sum_w
        final = (mean_p >= final_thr).astype(int)
        pos_count += final
        row = {"segment_id": seg_id}
        for j, col in enumerate(label_cols):
            row[col] = int(final[j])
        rows.append(row)
    rows = sorted(rows, key=lambda r: r["segment_id"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    pos_rate = ", ".join(
        f"{name}={pos_count[j] / max(1, len(rows)):.3f}" for j, name in enumerate(label_cols)
    )
    print(f"[multi] final positive rate per label: [{pos_rate}]")
    print(f"[multi] {len(groups)} 组模型软投票完成，写出 {len(rows)} 行 -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
