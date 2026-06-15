"""集成推理：多 checkpoint 概率平均 -> per-label 阈值 -> pred.csv。

训练侧（train.py, ensemble_topk>0）会保存 top-N 个 `ensemble_ep*.pt` 及
`logs/ensemble_manifest.json`，每个成员自带 valid 上拟合的 per-label 阈值。
但**集成平均后的概率分布与任一单模型都不同**，直接套用单模型阈值并不最优，因此本
脚本支持 `--tune_thresholds_on_valid`：先用同一组模型在验证集上做集成平均，重新拟合
每个标签的最优阈值，再应用到测试集。这通常是集成能否真正提分的关键。

显存安全：**一次只加载一个模型**跑完整个数据集、累加概率，再 del 释放，故 N 个 1.24B
的大模型也不会同时占用显存。

用法示例：
  # 用 manifest 里的 top-N + best.pt，在 valid 上重拟合阈值后预测 test
  python -m src.infer_ensemble \
      --config configs/whisper_qwen0_6b_lmf_2xL20.yaml \
      --manifest outputs/lmf_2xL20/logs/ensemble_manifest.json \
      --include_best outputs/lmf_2xL20/checkpoints/best_lmf_2xL20.pt \
      --test_root /path/to/test \
      --output_csv outputs/pred_ensemble.csv \
      --tune_thresholds_on_valid --weighted

  # 显式给定 checkpoint 列表 + 固定阈值文件
  python -m src.infer_ensemble --config <cfg> \
      --checkpoints a.pt b.pt c.pt \
      --test_root /path/to/test --output_csv pred.csv \
      --threshold_file outputs/thresholds.json
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

from src.data import (
    TurnTakingTestDataset,
    TurnTakingTrainDataset,
    build_collate_fn,
    build_train_samples_multitask,
    list_conv_ids,
    split_conversation_ids,
)
from src.models import MultimodalTurnTakingModel
from src.utils import find_best_f1_threshold, load_config, set_env_paths


def parse_args():
    p = argparse.ArgumentParser(description="集成推理（多 checkpoint 概率平均）")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--test_root", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    # --- 成员来源（可叠加；自动去重） ---
    p.add_argument("--checkpoints", type=str, nargs="*", default=[],
                   help="显式 checkpoint 路径列表")
    p.add_argument("--manifest", type=str, default=None,
                   help="ensemble_manifest.json；成员名相对 config 的 checkpoints_dir 解析")
    p.add_argument("--include_best", type=str, default=None,
                   help="额外把单个 best.pt 也纳入集成")
    p.add_argument("--weighted", action="store_true",
                   help="按 manifest 中各成员的 valid metric 加权平均（默认等权）")
    # --- 阈值策略（优先级从高到低） ---
    p.add_argument("--tune_thresholds_on_valid", action="store_true",
                   help="在验证集上对集成平均概率重新拟合 per-label 阈值（推荐）")
    p.add_argument("--threshold_file", type=str, default=None,
                   help="per-label 阈值 JSON：{'thresholds': {label: thr}}")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="所有标签统一阈值（最低优先级兜底）")
    # --- 其它 ---
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_segments", type=int, default=None)
    p.add_argument("--max_valid_batches", type=int, default=None,
                   help="调阈值时验证集最多跑多少 batch（None=全量，BC 稀疏建议全量）")
    return p.parse_args()


def resolve_members(args, cfg) -> list[dict]:
    """返回 [{path, weight, thresholds}]，已去重。"""
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    members: list[dict] = []
    seen: set[str] = set()

    def add(path: str, weight: float = 1.0, thresholds: dict | None = None):
        rp = str(Path(path).resolve())
        if rp in seen:
            return
        seen.add(rp)
        members.append({"path": path, "weight": float(weight), "thresholds": thresholds})

    if args.manifest:
        man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for m in man.get("members", []):
            mp = ckpt_dir / m["name"]
            add(str(mp), weight=float(m.get("metric", 1.0)), thresholds=m.get("thresholds"))
    for c in args.checkpoints:
        add(c)
    if args.include_best:
        add(args.include_best)

    if not members:
        raise SystemExit("未解析到任何 checkpoint，请用 --manifest / --checkpoints / --include_best 指定。")

    # 等权时把 weight 归一；加权时用 metric 归一（缺失的按均值兜底）。
    if args.weighted:
        ws = np.array([m["weight"] for m in members], dtype=np.float64)
        if not np.isfinite(ws).all() or ws.sum() <= 0:
            ws = np.ones(len(members))
    else:
        ws = np.ones(len(members))
    ws = ws / ws.sum()
    for m, w in zip(members, ws):
        m["weight"] = float(w)
    return members


@torch.no_grad()
def run_model_probs(model, loader, device, use_amp, id_key: str,
                    max_batches: int | None = None):
    """跑一个模型，返回 (ids, probs[N,L])，以及（若有）labels[N,L]。顺序即 loader 顺序。"""
    ids: list = []
    probs_list: list = []
    labels_list: list = []
    for bi, batch in enumerate(tqdm(loader, desc="infer", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        waveform = batch["waveform"].to(device, non_blocking=True)
        wave_len = batch["wave_len"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        context_labels = batch["context_labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(
                waveform=waveform, input_ids=input_ids,
                attention_mask=attention_mask, context_labels=context_labels,
                wave_len=wave_len,
            )
        probs = torch.sigmoid(logits).float().cpu().numpy()
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        probs_list.append(probs)
        ids.extend(batch[id_key])
        if "label" in batch:
            labels_list.append(batch["label"].cpu().numpy())
    probs = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, 1))
    labels = np.concatenate(labels_list, axis=0) if labels_list else None
    return ids, probs, labels


def build_model_from_ckpt(ckpt: dict, fallback_cfg: dict, device):
    """优先用 checkpoint 自带 config 构建（保证 stereo 等架构开关与训练一致）。"""
    model_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    model = MultimodalTurnTakingModel(model_cfg or fallback_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def ensemble_over_loader(members, loader, cfg, device, use_amp, id_key, max_batches=None):
    """逐个模型加载->推理->累加加权概率，返回 (ids, ensemble_probs, labels)。"""
    ref_ids = None
    sum_probs = None
    labels = None
    for i, m in enumerate(members):
        ckpt = torch.load(m["path"], map_location="cpu")
        model = build_model_from_ckpt(ckpt, cfg, device)
        ids, probs, lbls = run_model_probs(model, loader, device, use_amp, id_key, max_batches)
        if ref_ids is None:
            ref_ids, sum_probs = ids, np.zeros_like(probs)
            labels = lbls
        elif ids != ref_ids:
            raise RuntimeError("各模型推理样本顺序不一致（loader 必须 shuffle=False）。")
        sum_probs += m["weight"] * probs
        print(f"  [{i+1}/{len(members)}] {Path(m['path']).name} weight={m['weight']:.4f}")
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ref_ids, sum_probs, labels


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)

    multi_targets = list(cfg["labels"]["multi_targets"])
    label_cols = [x.lower() for x in multi_targets]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("use_amp", False))
    bs = int(args.batch_size or cfg["train"]["eval_batch_size"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    members = resolve_members(args, cfg)
    print(f"集成成员 {len(members)} 个，加权={'是' if args.weighted else '否(等权)'}")

    # ---- 阈值：可选地在验证集上对集成概率重新拟合 ----
    thresholds: dict | None = None
    if args.tune_thresholds_on_valid:
        paths = cfg["paths"]
        labels_dir = Path(paths["train_labels_dir"])
        conv_ids = list_conv_ids(labels_dir)
        split_ids = split_conversation_ids(
            conv_ids=conv_ids, valid_ratio=float(cfg["split"]["valid_ratio"]), seed=int(cfg["seed"]),
        )
        valid_samples = build_train_samples_multitask(
            labels_dir=labels_dir, conv_ids=split_ids["valid"],
            context_chunks=int(cfg["context_chunks"]), target_chunks=int(cfg["target_chunks"]),
            stride=int(cfg["stride"]), label_ids=cfg["labels"], target_labels=multi_targets,
            max_samples=cfg.get("max_valid_samples"),
        )
        valid_ds = TurnTakingTrainDataset(
            samples=valid_samples,
            train_audio_dir=Path(paths["train_audio_dir"]),
            train_text_dir=Path(paths["train_text_dir"]),
            train_labels_dir=labels_dir,
            context_chunks=int(cfg["context_chunks"]), target_chunks=int(cfg["target_chunks"]),
            chunk_ms=int(cfg["chunk_ms"]), sample_rate=int(cfg["sample_rate"]),
            augment_audio=False,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=bs, shuffle=False, num_workers=int(cfg["num_workers"]),
            collate_fn=collate_fn, pin_memory=True,
        )
        print(f"在验证集（{len(valid_samples)} 样本）上拟合集成阈值...")
        _, vprobs, vlabels = ensemble_over_loader(
            members, valid_loader, cfg, device, use_amp, "conv_id", args.max_valid_batches,
        )
        vlabels = vlabels.astype(int)
        thresholds = {}
        print(f"\n{'Label':<6}{'Thr':>10}{'F1':>10}{'PosRate':>10}")
        for i, name in enumerate(label_cols):
            t, f1 = find_best_f1_threshold(vprobs[:, i], vlabels[:, i])
            thresholds[name] = float(t)
            print(f"{name:<6}{t:>10.4f}{f1:>10.4f}{vlabels[:, i].mean():>10.4f}")
        thr_out = Path(args.output_csv).with_suffix(".thresholds.json")
        thr_out.parent.mkdir(parents=True, exist_ok=True)
        thr_out.write_text(json.dumps({"thresholds": thresholds}, indent=2), encoding="utf-8")
        print(f"集成阈值已保存 -> {thr_out.resolve()}\n")
    elif args.threshold_file:
        thresholds = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))["thresholds"]
    else:
        # 退而求其次：平均各成员自带阈值（若有），否则用统一 --threshold
        per_member = [m["thresholds"] for m in members if m.get("thresholds")]
        if per_member:
            thresholds = {
                name: float(np.mean([pm.get(name, args.threshold) for pm in per_member]))
                for name in label_cols
            }
            print(f"使用各成员阈值的平均值: {thresholds}")

    # ---- 测试集集成推理 ----
    test_ds = TurnTakingTestDataset(test_root=Path(args.test_root), sample_rate=int(cfg["sample_rate"]))
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn, pin_memory=True,
    )
    print(f"在测试集（{len(test_ds)} 样本）上集成推理...")
    seg_ids, probs, _ = ensemble_over_loader(members, test_loader, cfg, device, use_amp, "segment_id")

    if args.max_segments is not None:
        seg_ids = seg_ids[: args.max_segments]
        probs = probs[: args.max_segments]

    rows = []
    for k, seg_id in enumerate(seg_ids):
        row = {"segment_id": seg_id}
        for j, name in enumerate(label_cols):
            thr = float(thresholds[name]) if thresholds and name in thresholds else args.threshold
            row[name] = int(float(probs[k, j]) >= thr)
        rows.append(row)
    rows.sort(key=lambda r: r["segment_id"])

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["segment_id"] + label_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
