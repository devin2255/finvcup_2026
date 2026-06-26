"""
集成推理（多标签 event-level）：用 ensemble_manifest.json 里的多个模型做集成。

支持两种集成方式（`--vote`）：

- soft（默认，**概率平均软投票**）：
  逐成员算出 sigmoid 概率，对每个 segment 的每个标签**跨成员求平均概率**，
  再用 **best pt 的逐标签最优阈值**统一二值化为 0/1。
  这里的 "best pt" = manifest 里验证指标(metric)最高的成员（members 已按 metric 降序，
  即 members[0]），它就是训练保存的全局 best checkpoint，自带 best_thresholds。

- hard（旧行为，**逐标签多数硬投票**）：
  每个模型用**自己的最优阈值**把每个标签二值化，再做逐标签多数投票：
  得票 > 半数 的标签判为 1（5 个模型即 >=3 票）。

不论哪种方式，同一时刻 GPU 上只放一个模型，显存峰值=单模型。

成员 checkpoint 为“瘦身”格式（只存可训练参数）：构建模型时从本地预训练骨干
（Whisper/Qwen）加载冻结权重，再用 strict=False 叠加成员的可训练权重即可还原。
完整 state_dict 的旧 checkpoint 同样兼容（strict=False 会全部覆盖）。

输出 CSV 格式（与 src/infer_test.py 一致）：
- segment_id, <label_1>, <label_2>, ...（小写列名，值 0/1）

用法示例：
  python -m src.infer_ensemble \
      --config configs/whisper_qwen0_6b_lmf_ensemble.yaml \
      --test_root /path/to/test \
      --vote soft \
      --output_csv outputs/lmf_ensemble/pred_ensemble.csv
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
    p = argparse.ArgumentParser(
        description="集成推理：soft=概率平均后用 best pt 阈值二值化；hard=逐标签多数投票"
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--test_root", type=str, required=True, help="测试数据根目录")
    p.add_argument(
        "--vote",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="集成方式：soft=概率平均软投票（默认），hard=逐标签多数硬投票",
    )
    p.add_argument(
        "--soft_threshold_source",
        type=str,
        default="best",
        choices=["best", "mean"],
        help="soft 模式下统一阈值来源：best=best pt(指标最高成员)的阈值（默认）；"
        "mean=参与成员阈值的逐标签平均",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="ensemble_manifest.json 路径；默认取 <logs_dir>/ensemble_manifest.json",
    )
    p.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="成员 checkpoint 所在目录；默认取 config 的 paths.checkpoints_dir",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=None,
        help="只用 manifest 中前 K 个成员（已按指标降序）；默认用全部成员",
    )
    p.add_argument("--batch_size", type=int, default=None, help="默认取 config train.eval_batch_size")
    p.add_argument("--max_segments", type=int, default=None, help="仅处理前 N 条（冒烟测试）")
    p.add_argument(
        "--default_threshold",
        type=float,
        default=0.5,
        help="某成员缺少某标签阈值时的兜底阈值",
    )
    p.add_argument("--output_csv", type=str, required=True, help="输出 pred.csv 路径")
    return p.parse_args()


def _load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    members = data.get("members", [])
    if not members:
        raise RuntimeError(f"manifest 中没有任何成员: {manifest_path}")
    return members


def _resolve_members(args, cfg) -> tuple[list[dict], Path]:
    paths = cfg["paths"]
    ckpt_dir = Path(args.checkpoints_dir or paths["checkpoints_dir"])
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(paths["logs_dir"]) / "ensemble_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"找不到 manifest: {manifest_path}（训练时需设置 train.ensemble_topk > 0）"
        )
    members = _load_manifest(manifest_path)
    if args.topk is not None:
        members = members[: int(args.topk)]

    resolved = []
    for m in members:
        member_path = ckpt_dir / m["name"]
        if not member_path.exists():
            print(f"[WARN] 成员文件缺失，跳过: {member_path}")
            continue
        resolved.append({**m, "path": member_path})
    if not resolved:
        raise RuntimeError(f"没有可用的成员 checkpoint（检查目录 {ckpt_dir}）")
    return resolved, manifest_path


def _member_thresholds(member: dict, ckpt: dict, label_cols: list[str], default: float) -> dict:
    """优先用 manifest 里的阈值，其次用 checkpoint 内自带的，最后兜底。"""
    thr = member.get("thresholds")
    if not isinstance(thr, dict):
        thr = ckpt.get("thresholds") if isinstance(ckpt, dict) else None
    thr = thr if isinstance(thr, dict) else {}
    return {name: float(thr.get(name, default)) for name in label_cols}


def _best_pt_threshold_vec(
    members: list[dict], cfg, label_cols: list[str], default: float
) -> tuple[np.ndarray, str]:
    """soft 软投票用的统一阈值 = best pt（验证 metric 最高成员）的逐标签阈值。

    members 由 manifest 按 metric 降序保存，故 members[0] 即 best pt（训练时保存的全局
    best checkpoint 对应同一 epoch）。其 thresholds 与 logs/best_thresholds.json 一致。
    若该成员缺 thresholds（旧 manifest），退而读 <logs_dir>/best_thresholds.json，再兜底。
    """
    best = max(members, key=lambda m: m.get("metric", float("-inf")))
    thr = best.get("thresholds")
    src = f"manifest:{best.get('name')}(metric={best.get('metric')})"
    if not isinstance(thr, dict) or not thr:
        bt = Path(cfg["paths"]["logs_dir"]) / "best_thresholds.json"
        if bt.exists():
            with open(bt, "r", encoding="utf-8") as f:
                thr = json.load(f).get("thresholds", {})
            src = str(bt)
        else:
            thr = {}
            src = f"default={default}"
    vec = np.array([float(thr.get(name, default)) for name in label_cols], dtype=np.float32)
    return vec, src


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)  # 设置离线/缓存等环境变量

    multi_targets = list(cfg["labels"]["multi_targets"])
    label_cols = [x.lower() for x in multi_targets]
    n_labels = len(label_cols)

    members, manifest_path = _resolve_members(args, cfg)
    num_members = len(members)
    is_soft = args.vote == "soft"

    # hard 投票用：严格多数（得票 > 半数）。5 个成员 => >=3 票。
    majority_need = num_members // 2 + 1

    # soft 投票用：统一阈值（best pt 或 成员平均）。
    soft_thr_vec = None
    if is_soft:
        if args.soft_threshold_source == "mean":
            mats = []
            for m in members:
                thr = m.get("thresholds") if isinstance(m.get("thresholds"), dict) else {}
                mats.append([float(thr.get(name, args.default_threshold)) for name in label_cols])
            soft_thr_vec = np.mean(np.array(mats, dtype=np.float32), axis=0)
            soft_src = "mean(参与成员阈值平均)"
        else:
            soft_thr_vec, soft_src = _best_pt_threshold_vec(
                members, cfg, label_cols, args.default_threshold
            )

    print(f"[ensemble] manifest={manifest_path}  vote={args.vote}  成员数={num_members}")
    if is_soft:
        thr_show = {c: round(float(soft_thr_vec[j]), 3) for j, c in enumerate(label_cols)}
        print(f"[ensemble] soft：跨成员概率平均，统一阈值来源={soft_src}，阈值={thr_show}")
    else:
        print(
            f"[ensemble] hard：逐标签多数投票阈值=>={majority_need} 票（每个模型用各自最优阈值）"
        )
    for m in members:
        print(f"  - {m['name']} (epoch={m.get('epoch')}, metric={m.get('metric')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    test_root = Path(args.test_root)
    ds = TurnTakingTestDataset(
        test_root=test_root,
        sample_rate=int(cfg["sample_rate"]),
        context_chunks=int(cfg["context_chunks"]),  # 归一化变长上下文到定长，支持复赛 (0,30] 动态时长
    )
    bs = int(args.batch_size or cfg["train"]["eval_batch_size"])
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    use_amp = bool(cfg["train"].get("use_amp", False))
    limit = args.max_segments

    # 逐标签累计：
    #   soft -> prob_sum[seg] 累计各成员概率(float64)，count[seg] 记成员数用于求平均；
    #   hard -> votes[seg] 累计各成员二值票(int)。
    agg: dict[str, np.ndarray] = {}
    count: dict[str, int] = {}
    seg_order: list[str] = []  # 保持首次出现顺序（loader 确定性，各成员一致）

    for mi, member in enumerate(members):
        # weights_only=False：checkpoint 由本项目训练产出（可信），且含 config/thresholds
        # 等非张量对象；torch>=2.6 默认 weights_only=True 会拒绝加载，这里显式关闭。
        ckpt = torch.load(member["path"], map_location="cpu", weights_only=False)
        # hard 模式每个成员用自己的阈值（soft 模式仅用于诊断打印正例率）。
        thr_map = _member_thresholds(member, ckpt, label_cols, args.default_threshold)
        thr_vec = np.array([thr_map[name] for name in label_cols], dtype=np.float32)

        model = MultimodalTurnTakingModel(cfg).to(device)
        # 瘦身/完整 checkpoint 均兼容：strict=False 叠加可训练参数到预训练骨干上。
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if unexpected:
            print(f"[WARN] {member['name']} 含未预期的键 {len(unexpected)} 个（已忽略）")
        model.eval()

        done = 0
        member_pos = np.zeros(n_labels, dtype=np.int64)  # 该成员各标签预测正例数（诊断用）
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"member {mi + 1}/{num_members} {member['name']}"):
                waveform = batch["waveform"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                context_labels = batch["context_labels"].to(device, non_blocking=True)
                segment_ids = batch["segment_id"]

                with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                    logits = model(
                        waveform=waveform,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        context_labels=context_labels,
                    )
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim == 1:
                    probs = probs.reshape(-1, 1)

                for i, seg_id in enumerate(segment_ids):
                    p = probs[i]
                    if p.shape[0] != n_labels:
                        raise RuntimeError(
                            f"logits dim {p.shape[0]} != len(multi_targets) {n_labels}"
                        )
                    if seg_id not in agg:
                        agg[seg_id] = np.zeros(n_labels, dtype=np.float64)
                        count[seg_id] = 0
                        seg_order.append(seg_id)
                    if is_soft:
                        agg[seg_id] += p.astype(np.float64)  # 累计概率，最后求平均
                    else:
                        agg[seg_id] += (p >= thr_vec).astype(np.float64)  # 累计二值票
                    count[seg_id] += 1
                    member_pos += (p >= thr_vec).astype(np.int64)
                    done += 1
                    if limit is not None and done >= limit:
                        break
                if limit is not None and done >= limit:
                    break

        pos_rate = ", ".join(
            f"{name}={member_pos[j] / max(1, done):.3f}" for j, name in enumerate(label_cols)
        )
        print(f"[member {mi + 1}] thr={ {k: round(v, 3) for k, v in thr_map.items()} } pos_rate[{pos_rate}]")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总 -> 最终 0/1
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["segment_id"] + label_cols
    rows: list[dict] = []
    for seg_id in seg_order:
        n = max(1, count[seg_id])
        if is_soft:
            avg_prob = agg[seg_id] / n  # 跨成员平均概率
            final = (avg_prob >= soft_thr_vec).astype(int)
        else:
            final = (agg[seg_id] >= majority_need).astype(int)
        row = {"segment_id": seg_id}
        for j, col in enumerate(label_cols):
            row[col] = int(final[j])
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["segment_id"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    mode_desc = (
        f"soft 概率平均(阈值来源={soft_src})" if is_soft else f"hard 多数投票(>={majority_need}票)"
    )
    print(
        f"\n[ensemble] {num_members} 个模型 {mode_desc} 完成，写出 {len(rows)} 行 -> {out_path.resolve()}"
    )


if __name__ == "__main__":
    main()
