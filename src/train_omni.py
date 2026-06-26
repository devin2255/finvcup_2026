"""
Qwen2.5-Omni-3B 原生训练入口（4 卡 DDP）。

与 src/train.py 的差异：
  - 模型：src.models.omni_turntaking.OmniTurnTaking（LoRA + 分类头）
  - 数据：src.data.omni_dataset（Omni processor 打包多模态对话）
  - 精度：bf16 自动混合精度（不用 GradScaler）
其余（样本/标签构造、EMA、阈值寻优、ensemble top-k 瘦身保存、指标）复用 src.train / src.utils。

启动：
  torchrun --nproc_per_node=4 -m src.train_omni --config configs/qwen2_5_omni3b_4xL20.yaml
冒烟（小样本快速验证管线）：
  ... --config ... --max_train_samples 64 --max_valid_samples 64 --epochs 1 --max_steps_per_epoch 5
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.data.dataset import build_train_samples_multitask, list_conv_ids, split_conversation_ids
from src.data.omni_dataset import (
    OmniTurnTakingTrainDataset,
    build_omni_collate,
)
from src.models.omni_turntaking import OmniTurnTaking, build_omni_processor
from src.train import (
    EMA,
    _extract_best_thresholds,
    _format_multilabel_metrics_line,
    _get_best_checkpoint_name,
    _select_eval_samples,
    _trainable_state_dict,
)
from src.utils import (
    cleanup_distributed,
    compute_multilabel_metrics,
    ensure_dirs,
    is_distributed,
    load_config,
    save_json,
    set_env_paths,
    set_seed,
    setup_distributed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/qwen2_5_omni3b_4xL20.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_steps_per_epoch", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_valid_samples", type=int, default=None)
    return p.parse_args()


def _amp_dtype(cfg) -> torch.dtype:
    name = str(cfg["train"].get("amp_dtype", "bfloat16")).lower()
    return torch.float16 if name in ("float16", "fp16") else torch.bfloat16


def _to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _model_inputs(batch_dev: dict) -> dict:
    return {k: v for k, v in batch_dev.items() if k not in ("label", "segment_id")}


@torch.no_grad()
def evaluate(model, data_loader, device, use_amp, amp_dtype, label_names, max_batches=None):
    model.eval()
    all_labels, all_probs = [], []
    for bi, batch in enumerate(tqdm(data_loader, desc="eval", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        batch = _to_device(batch, device)
        labels = batch["label"]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(**_model_inputs(batch))
        probs = torch.sigmoid(logits.float())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())
    return compute_multilabel_metrics(all_labels, all_probs, label_names=label_names)


def _build_criterion(cfg, train_samples, multi_targets, device):
    if cfg["train"].get("pos_weight_mode", "per_label") in ("per_label", "capped_per_label"):
        y_mat = np.asarray([s.label_vec for s in train_samples], dtype=np.float32)
        pos = y_mat.sum(axis=0)
        neg = y_mat.shape[0] - pos
        pw = neg / np.maximum(1.0, pos)
        if cfg["train"].get("pos_weight_mode") == "capped_per_label":
            pw = np.minimum(pw, float(cfg["train"].get("pos_weight_cap", 5.0)))
        pos_weight = torch.tensor(pw, device=device, dtype=torch.float32)
    else:
        pos_weight = torch.ones(len(multi_targets), device=device, dtype=torch.float32)

    focal_gamma = float(cfg["train"].get("focal_gamma", 0.0))
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))

    if focal_gamma > 0:
        class MultiLabelFocalLoss(torch.nn.Module):
            def __init__(self, gamma, pos_weight, label_smoothing=0.0):
                super().__init__()
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                self.register_buffer("pos_weight", pos_weight)

            def forward(self, logits, targets):
                if self.label_smoothing > 0:
                    targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none", pos_weight=self.pos_weight
                )
                probs = torch.sigmoid(logits)
                p_t = targets * probs + (1 - targets) * (1 - probs)
                return ((1 - p_t) ** self.gamma * bce).mean()

        return MultiLabelFocalLoss(focal_gamma, pos_weight, label_smoothing).to(device), pos_weight
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight), pos_weight


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)
    ensure_dirs(cfg)
    set_seed(int(cfg["seed"]))

    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.max_steps_per_epoch is not None:
        cfg["train"]["max_steps_per_epoch"] = int(args.max_steps_per_epoch)
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = int(args.max_train_samples)
    if args.max_valid_samples is not None:
        cfg["max_valid_samples"] = int(args.max_valid_samples)

    local_rank, world_size, rank = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    paths = cfg["paths"]
    labels_dir = Path(paths["train_labels_dir"])
    multi_targets = list(cfg["labels"]["multi_targets"])
    metric_label_names = [x.lower() for x in multi_targets]

    conv_ids = list_conv_ids(labels_dir)
    split_ids = split_conversation_ids(conv_ids, float(cfg["split"]["valid_ratio"]), int(cfg["seed"]))
    train_ids, valid_ids = split_ids["train"], split_ids["valid"]

    train_samples = build_train_samples_multitask(
        labels_dir=labels_dir, conv_ids=train_ids,
        context_chunks=int(cfg["context_chunks"]), target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]), label_ids=cfg["labels"], target_labels=multi_targets,
        max_samples=cfg["max_train_samples"],
    )
    valid_samples = build_train_samples_multitask(
        labels_dir=labels_dir, conv_ids=valid_ids,
        context_chunks=int(cfg["context_chunks"]), target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]), label_ids=cfg["labels"], target_labels=multi_targets,
        max_samples=cfg["max_valid_samples"],
    )
    valid_eval_samples = _select_eval_samples(valid_samples, cfg, seed=int(cfg["seed"]))

    if is_main:
        save_json(Path(paths["logs_dir"]) / "split_ids.json", split_ids)
        save_json(Path(paths["logs_dir"]) / "sample_count.json", {
            "train_samples": len(train_samples),
            "valid_samples": len(valid_samples),
            "valid_eval_samples": len(valid_eval_samples),
        })

    # ---- Omni processor + collate + datasets ----
    processor = build_omni_processor(cfg["omni"]["model_path"])
    aug = cfg.get("data_augmentation", {})
    hist = cfg.get("history", {})
    collate_fn = build_omni_collate(
        processor, sample_rate=int(cfg["sample_rate"]),
        max_text_tokens=int(cfg["omni"].get("max_text_tokens", 512)),
    )

    def _make_ds(samples, dynamic):
        return OmniTurnTakingTrainDataset(
            samples=samples,
            train_audio_dir=Path(paths["train_audio_dir"]),
            train_text_dir=Path(paths["train_text_dir"]),
            train_labels_dir=labels_dir,
            context_chunks=int(cfg["context_chunks"]),
            chunk_ms=int(cfg["chunk_ms"]),
            sample_rate=int(cfg["sample_rate"]),
            labels_cfg=cfg["labels"],
            history_include=bool(hist.get("include", True)),
            history_chunks=int(hist.get("history_chunks", 125)),
            dynamic_context=bool(dynamic and aug.get("dynamic_context", False)),
            min_context_chunks=int(aug.get("min_context_chunks", 125)),
            max_context_chunks=int(aug.get("max_context_chunks", 375)),
            context_prob=float(aug.get("context_prob", 0.5)),
        )

    train_dataset = _make_ds(train_samples, dynamic=True)
    valid_dataset = _make_ds(valid_eval_samples, dynamic=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg["train"]["batch_size"]), sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=int(cfg["train"]["eval_batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), collate_fn=collate_fn, pin_memory=True,
    )

    # ---- model ----
    model = OmniTurnTaking(cfg).to(device)
    if is_main:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[omni] trainable={n_train/1e6:.2f}M / total={n_total/1e9:.2f}B "
              f"({100*n_train/max(1,n_total):.3f}% trainable)")
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=bool(cfg["train"].get("ddp_find_unused", True)))
    raw_model = model.module if hasattr(model, "module") else model

    criterion, pos_weight = _build_criterion(cfg, train_samples, multi_targets, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"]["weight_decay"]),
    )

    max_epochs = int(cfg["train"]["epochs"])
    accum_steps = int(cfg["train"]["gradient_accumulation_steps"])
    max_steps_cfg = cfg["train"].get("max_steps_per_epoch", None)
    max_steps_per_epoch = int(max_steps_cfg) if max_steps_cfg is not None else None
    steps_per_epoch = min(len(train_loader), max_steps_per_epoch) if max_steps_per_epoch else len(train_loader)
    total_update_steps = max(1, (steps_per_epoch * max_epochs) // max(1, accum_steps))
    warmup_steps = int(total_update_steps * float(cfg["train"].get("warmup_ratio", 0.03)))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    use_amp = bool(cfg["train"].get("use_amp", True))
    amp_dtype = _amp_dtype(cfg)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    use_ema = bool(cfg["train"].get("use_ema", False))
    ema = EMA(raw_model, cfg["train"].get("ema_decay", 0.999)) if (use_ema and is_main) else None

    grad_clip = float(cfg["train"]["grad_clip_norm"])
    save_metric = str(cfg["train"]["save_metric"])
    early_stop_patience = int(cfg["train"]["early_stop_patience"])
    ensemble_topk = int(cfg["train"].get("ensemble_topk", 0))
    eval_valid_max_batches = cfg["train"].get("eval_valid_max_batches", None)
    eval_valid_max_batches = int(eval_valid_max_batches) if eval_valid_max_batches is not None else None
    log_every = int(cfg["train"].get("log_every_steps", 20))

    best_metric = -math.inf
    best_path = Path(paths["checkpoints_dir"]) / _get_best_checkpoint_name(cfg)
    topk_members: list[dict] = []
    bad_epochs = 0
    global_step = 0

    if is_main:
        print(f"[omni] amp_dtype={amp_dtype}, use_scaler={use_scaler}, "
              f"effective_batch={int(cfg['train']['batch_size'])*max(1,world_size)*accum_steps}")

    for epoch in range(max_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(train_loader, desc=f"train ep{epoch}", leave=False) if is_main else train_loader

        epoch_loss_sum, epoch_steps = 0.0, 0
        for step, batch in enumerate(iterator):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            batch = _to_device(batch, device)
            labels = batch["label"]
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(**_model_inputs(batch))
                loss = criterion(logits, labels) / accum_steps

            if not torch.isfinite(loss):
                if is_main:
                    print(f"[WARN] non-finite loss ep{epoch} step{step}, skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_value = float(loss.item() * accum_steps)
            epoch_loss_sum += loss_value
            epoch_steps += 1
            global_step += 1

            if (step + 1) % accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(raw_model, global_step)

            if is_main and (step + 1) % log_every == 0:
                lr_now = float(optimizer.param_groups[0]["lr"])
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr_now:.2e}")
                if not sys.stderr.isatty():
                    print(f"[train] ep{epoch} step{step} gstep{global_step} "
                          f"loss={loss_value:.4f} lr={lr_now:.2e}", flush=True)

        if is_distributed():
            torch.distributed.barrier()

        if is_main:
            if ema is not None:
                ema.store_and_copy_to(raw_model)
            eval_model = raw_model
            metrics_valid = evaluate(
                eval_model, valid_loader, device, use_amp, amp_dtype,
                label_names=metric_label_names, max_batches=eval_valid_max_batches,
            )
            metric_value = float(metrics_valid[save_metric])
            save_json(Path(paths["logs_dir"]) / f"valid_epoch_{epoch}.json", metrics_valid)
            valid_thresholds = _extract_best_thresholds(metrics_valid, metric_label_names)
            save_json(Path(paths["logs_dir"]) / f"thresholds_epoch_{epoch}.json", {"thresholds": valid_thresholds})
            print(
                f"[Epoch {epoch}] train_loss={epoch_loss_sum/max(1,epoch_steps):.6f} "
                f"valid_macro_f1={metrics_valid['macro_f1']:.4f} "
                f"valid_macro_best_f1={metrics_valid['macro_best_f1']:.4f} "
                f"valid_macro_auc={metrics_valid['macro_roc_auc']:.4f} "
                f"| valid[{_format_multilabel_metrics_line(metrics_valid, metric_label_names)}] "
                f"best_{save_metric}={max(best_metric, metric_value):.4f}"
            )

            if ensemble_topk > 0:
                worst_kept = min((m["metric"] for m in topk_members), default=-math.inf)
                if len(topk_members) < ensemble_topk or metric_value > worst_kept:
                    member_name = f"ensemble_ep{epoch}.pt"
                    torch.save({
                        "epoch": epoch, "metric": metric_value,
                        "model": _trainable_state_dict(eval_model),
                        "trainable_only": True, "config": cfg, "thresholds": valid_thresholds,
                    }, Path(paths["checkpoints_dir"]) / member_name)
                    topk_members.append({
                        "name": member_name, "epoch": epoch, "metric": metric_value,
                        "thresholds": valid_thresholds,
                    })
                    topk_members.sort(key=lambda m: m["metric"], reverse=True)
                    while len(topk_members) > ensemble_topk:
                        evicted = topk_members.pop()
                        ev_path = Path(paths["checkpoints_dir"]) / evicted["name"]
                        if ev_path.exists():
                            ev_path.unlink()
                    save_json(Path(paths["logs_dir"]) / "ensemble_manifest.json",
                              {"save_metric": save_metric, "members": topk_members})

            if metric_value > best_metric:
                bad_epochs = 0
                best_metric = metric_value
                torch.save({
                    "epoch": epoch, "best_metric": best_metric,
                    "model": _trainable_state_dict(eval_model),  # 瘦身：只存 LoRA+头
                    "trainable_only": True, "config": cfg, "thresholds": valid_thresholds,
                }, best_path)
                save_json(Path(paths["logs_dir"]) / "best_thresholds.json", {"thresholds": valid_thresholds})
            else:
                bad_epochs += 1

            if ema is not None:
                ema.restore(raw_model)

        # 早停：rank0 决定，广播给其它 rank
        stop = torch.tensor([1 if (is_main and bad_epochs >= early_stop_patience) else 0], device=device)
        if is_distributed():
            torch.distributed.broadcast(stop, src=0)
            torch.distributed.barrier()
        if int(stop.item()) == 1:
            break

    cleanup_distributed()


if __name__ == "__main__":
    main()
