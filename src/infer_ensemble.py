"""
集成推理（多标签 event-level）：用 ensemble_manifest.json 里的多个模型做投票。

核心逻辑（与训练保存逻辑配套）：
- 训练时 `ensemble_topk: N` 会按 valid best_f1 排名保留 top-N 个 checkpoint，
  每个成员自带各标签最优阈值，并写入 <logs_dir>/ensemble_manifest.json。
- 本脚本逐个加载成员（同一时刻 GPU 上只放一个模型，显存峰值=单模型），
  **每个模型用自己的最优阈值**把每个标签二值化为 0/1，再做**逐标签多数投票**：
  得票 > 半数 的标签判为 1（5 个模型即 >=3 票）。

成员 checkpoint 为“瘦身”格式（只存可训练参数）：构建模型时从本地预训练骨干
（Whisper/Qwen）加载冻结权重，再用 strict=False 叠加成员的可训练权重即可还原。
完整 state_dict 的旧 checkpoint 同样兼容（strict=False 会全部覆盖）。

输出 CSV 格式（与 src/infer_test.py 一致）：
- segment_id, <label_1>, <label_2>, ...（小写列名，值 0/1）

用法示例：
  python -m src.infer_ensemble \
      --config configs/whisper_qwen0_6b_lmf_ensemble.yaml \
      --test_root /path/to/test \
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
        description="集成推理：多模型逐标签多数投票（每个模型用自己的最优阈值），导出 pred.csv"
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--test_root", type=str, required=True, help="测试数据根目录")
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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)  # 设置离线/缓存等环境变量

    multi_targets = list(cfg["labels"]["multi_targets"])
    label_cols = [x.lower() for x in multi_targets]
    n_labels = len(label_cols)

    members, manifest_path = _resolve_members(args, cfg)
    num_members = len(members)
    # 严格多数：得票 > 半数。5 个成员 => >=3 票。
    majority_need = num_members // 2 + 1
    print(
        f"[ensemble] manifest={manifest_path}\n"
        f"[ensemble] 成员数={num_members}，逐标签多数投票阈值=>={majority_need} 票（每个模型用各自最优阈值）"
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

    # 逐标签累计票数：seg_id -> np.ndarray[n_labels]（int 计票）
    votes: dict[str, np.ndarray] = {}
    seg_order: list[str] = []  # 保持首次出现顺序（loader 确定性，各成员一致）

    for mi, member in enumerate(members):
        ckpt = torch.load(member["path"], map_location="cpu")
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

                with torch.amp.autocast("cuda", enabled=use_amp):
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
                    vote = (p >= thr_vec).astype(np.int64)  # 用该模型自己的阈值
                    if seg_id not in votes:
                        votes[seg_id] = np.zeros(n_labels, dtype=np.int64)
                        seg_order.append(seg_id)
                    votes[seg_id] += vote
                    member_pos += vote
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

    # 逐标签多数投票 -> 最终 0/1
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["segment_id"] + label_cols
    rows: list[dict] = []
    for seg_id in seg_order:
        v = votes[seg_id]
        final = (v >= majority_need).astype(int)
        row = {"segment_id": seg_id}
        for j, col in enumerate(label_cols):
            row[col] = int(final[j])
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["segment_id"])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(
        f"\n[ensemble] {num_members} 个模型多数投票完成，写出 {len(rows)} 行 -> {out_path.resolve()}"
    )


if __name__ == "__main__":
    main()
