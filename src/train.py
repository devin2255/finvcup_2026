import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.data import (
    TurnTakingTrainDataset,
    build_collate_fn,
    build_train_samples_multitask,
    list_conv_ids,
    split_conversation_ids,
)
from src.models import MultimodalTurnTakingModel
from src.pos_weight import compute_pos_weight
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
    p.add_argument("--config", type=str, default="configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml")
    p.add_argument("--resume", type=str, default=None)
    # 仅用于冒烟/快速迭代：覆盖 config 中的训练参数（不写回文件）
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_steps_per_epoch", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_valid_samples", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    use_amp: bool,
    label_names: list[str] | None = None,
    max_batches: int | None = None,
):
    model.eval()
    all_labels, all_probs = [], []
    for bi, batch in enumerate(tqdm(data_loader, desc="eval", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        waveform = batch["waveform"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        context_labels = batch["context_labels"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
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

        probs = torch.sigmoid(logits)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    labels_np = (
        torch.as_tensor(all_labels).numpy() if len(all_labels) > 0 else np.array([])
    )
    if labels_np.ndim == 2:
        return compute_multilabel_metrics(all_labels, all_probs, label_names=label_names)
    raise RuntimeError("Baseline 固化为多标签训练，evaluate 不应进入二分类分支。")


def _format_multilabel_metrics_line(metrics: dict, label_names: list[str]) -> str:
    parts = []
    for n in label_names:
        parts.append(
            f"{n}:acc={metrics.get(f'{n}_accuracy', 0.0):.4f},"
            f"f1={metrics.get(f'{n}_f1', 0.0):.4f},"
            f"bf1={metrics.get(f'{n}_best_f1', 0.0):.4f},"
            f"auc={metrics.get(f'{n}_roc_auc', 0.5):.4f}"
        )
    return " | ".join(parts)


def _get_valid_loader_shuffle(cfg: dict) -> bool:
    return bool(cfg.get("train", {}).get("eval_valid_shuffle", False))


def _get_best_checkpoint_name(cfg: dict) -> str:
    return str(cfg.get("train", {}).get("best_checkpoint_name", "best.pt"))


def _select_eval_samples(samples: list, cfg: dict, seed: int) -> list:
    sample_count = cfg.get("train", {}).get("eval_valid_sample_count", None)
    if sample_count is None:
        return list(samples)
    sample_count = int(sample_count)
    if sample_count <= 0 or sample_count >= len(samples):
        return list(samples)
    indices = sorted(random.Random(seed + 1009).sample(range(len(samples)), sample_count))
    return [samples[i] for i in indices]


def _extract_best_thresholds(metrics: dict, label_names: list[str]) -> dict[str, float]:
    return {name: float(metrics.get(f"{name}_best_threshold", 0.5)) for name in label_names}


def _trainable_state_dict(model) -> dict:
    """Return only the trainable (requires_grad=True) parameters of `model`.

    Ensemble members are pure inference artifacts: the frozen Whisper/Qwen
    backbones are identical across members and reloaded from the same local
    pretrained paths at inference time, so storing them per-member wastes
    ~5GB each. We keep only the fine-tuned params (~170MB) and overlay them
    with strict=False on a freshly built model (see src/infer_ensemble.py).
    Buffers are excluded; the whisper/qwen path carries no train-critical
    buffers (LayerNorm, no BatchNorm running stats).
    """
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k in trainable_keys
    }


class EMA:
    """Exponential moving average of the model's *trainable* parameters.

    Why trainable-only: the frozen Whisper/Qwen backbones never change, so an
    EMA of them equals themselves — shadowing them would just waste ~5GB. We
    average only the fine-tuned params (~50M). Maintained on rank0 only; under
    DDP all ranks hold identical params right after each optimizer step, so
    rank0's EMA is valid for the whole job.

    Per epoch end (rank0):
        ema.store_and_copy_to(model)   # 备份原权重，把 EMA 权重写入模型
        ... evaluate / 存 best & ensemble（此时捕获的就是 EMA 权重）...
        ema.restore(model)             # 还原原权重，下个 epoch 从原权重继续训练
    """

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model, step):
        # 早期用较小 decay 快速跟随，逐步逼近目标 decay（避免初始权重污染 EMA）。
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def store_and_copy_to(self, model):
        self._backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            return
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = None


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_env_paths(cfg)
    ensure_dirs(cfg)
    set_seed(int(cfg["seed"]))

    # CLI overrides (for smoke runs)
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
    writer = None

    paths = cfg["paths"]
    labels_dir = Path(paths["train_labels_dir"])
    train_audio_dir = Path(paths["train_audio_dir"])
    train_text_dir = Path(paths["train_text_dir"])

    conv_ids = list_conv_ids(labels_dir)
    split_ids = split_conversation_ids(
        conv_ids=conv_ids,
        valid_ratio=float(cfg["split"]["valid_ratio"]),
        seed=int(cfg["seed"]),
    )
    train_ids, valid_ids = split_ids["train"], split_ids["valid"]
    # Baseline 固化：只做 event-level 多标签（未来 2s 内 5 个标签分别是否出现）
    use_multi_label = True
    multi_targets = list(cfg.get("labels", {}).get("multi_targets", ["C", "NA", "I", "BC", "T"]))
    metric_label_names = [x.lower() for x in multi_targets]

    train_samples = build_train_samples_multitask(
        labels_dir=labels_dir,
        conv_ids=train_ids,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]),
        label_ids=cfg["labels"],
        target_labels=multi_targets,
        max_samples=cfg["max_train_samples"],
    )
    valid_samples = build_train_samples_multitask(
        labels_dir=labels_dir,
        conv_ids=valid_ids,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        stride=int(cfg["stride"]),
        label_ids=cfg["labels"],
        target_labels=multi_targets,
        max_samples=cfg["max_valid_samples"],
    )
    valid_eval_samples = _select_eval_samples(valid_samples, cfg, seed=int(cfg["seed"]))

    if is_main:
        save_json(Path(paths["logs_dir"]) / "split_ids.json", split_ids)
        save_json(
            Path(paths["logs_dir"]) / "sample_count.json",
            {
                "train_samples": len(train_samples),
                "valid_samples": len(valid_samples),
                "valid_eval_samples": len(valid_eval_samples),
            },
        )
        writer = SummaryWriter(log_dir=str(Path(paths["logs_dir"]) / "tb"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["text_encoder"]["model_name"], use_fast=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = build_collate_fn(tokenizer, int(cfg["text_encoder"]["max_length"]))

    vap_cfg = cfg.get("vap_aux", {}) or {}
    use_vap = bool(vap_cfg.get("enabled", False))
    vap_weight = float(vap_cfg.get("weight", 0.3))
    vap_bins = int(vap_cfg.get("bins", cfg.get("target_chunks", 25)))
    vad_log_offset = float(vap_cfg.get("vad_log_offset", 2.0))
    if is_main and use_vap:
        print(f"[VAP] aux head enabled: weight={vap_weight}, bins={vap_bins}, vad_log_offset={vad_log_offset}")

    vapfeat_cfg = cfg.get("vap_feat", {}) or {}
    use_vap_feat = bool(vapfeat_cfg.get("enabled", False))
    vap_feat_dir = vapfeat_cfg.get("cache_dir") if use_vap_feat else None
    vap_feat_rate = float(vapfeat_cfg.get("frame_rate", 10.0))
    vap_feat_dim_cfg = int(vapfeat_cfg.get("feat_dim", 18))
    if is_main and use_vap_feat:
        print(f"[VAP-feat] enabled: cache_dir={vap_feat_dir}, frame_rate={vap_feat_rate}, dim={vap_feat_dim_cfg}")

    train_dataset = TurnTakingTrainDataset(
        samples=train_samples,
        train_audio_dir=train_audio_dir,
        train_text_dir=train_text_dir,
        train_labels_dir=labels_dir,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
        augment_audio=True,
        # Phase 1: 动态上下文配置
        dynamic_context=cfg.get("data_augmentation", {}).get("dynamic_context", False),
        min_context_chunks=cfg.get("data_augmentation", {}).get("min_context_chunks", 125),
        max_context_chunks=cfg.get("data_augmentation", {}).get("max_context_chunks", 375),
        context_prob=cfg.get("data_augmentation", {}).get("context_prob", 0.5),
        vap_target=use_vap,
        vap_bins=vap_bins,
        vad_log_offset=vad_log_offset,
        vap_feat_dir=vap_feat_dir,
        vap_frame_rate=vap_feat_rate,
        vap_feat_dim=vap_feat_dim_cfg,
    )
    valid_dataset = TurnTakingTrainDataset(
        samples=valid_eval_samples,
        train_audio_dir=train_audio_dir,
        train_text_dir=train_text_dir,
        train_labels_dir=labels_dir,
        context_chunks=int(cfg["context_chunks"]),
        target_chunks=int(cfg["target_chunks"]),
        chunk_ms=int(cfg["chunk_ms"]),
        sample_rate=int(cfg["sample_rate"]),
        augment_audio=False,
        # 验证集不使用动态上下文
        dynamic_context=False,
        vap_feat_dir=vap_feat_dir,
        vap_frame_rate=vap_feat_rate,
        vap_feat_dim=vap_feat_dim_cfg,
    )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["train"]["eval_batch_size"]),
        shuffle=_get_valid_loader_shuffle(cfg),
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 训练阶段完全不考虑测试集：不加载、不评估
    eval_test_every_steps = 0
    eval_valid_max_batches = cfg["train"].get("eval_valid_max_batches", None)
    eval_valid_max_batches = (
        int(eval_valid_max_batches) if eval_valid_max_batches is not None else None
    )
    test_loader: DataLoader | None = None
    gt_test_labels = None

    model = MultimodalTurnTakingModel(cfg).to(device)
    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if cfg["train"].get("pos_weight_mode", "per_label") in ("per_label", "capped_per_label"):
        y_mat = np.asarray([s.label_vec for s in train_samples], dtype=np.float32)  # [N,5]
        if cfg["train"].get("pos_weight_mode") == "capped_per_label":
            cap = float(cfg["train"].get("pos_weight_cap", 5.0))
            per_label_cap = cfg["train"].get("pos_weight_cap_per_label", None)
        else:
            cap = float("inf")
            per_label_cap = None
        pw = compute_pos_weight(y_mat, metric_label_names, cap=cap, per_label_cap=per_label_cap)
        pos_weight = torch.tensor(pw, device=device, dtype=torch.float32)
    else:
        pos_weight = torch.ones(len(multi_targets), device=device, dtype=torch.float32)

    focal_gamma = float(cfg["train"].get("focal_gamma", 0.0))
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))  # Phase 1: 标签平滑
    
    if focal_gamma > 0:
        # Focal Loss: down-weight easy examples, forcing the model to focus on
        # hard/rare labels like BC (3.65%) and I (14.15%).
        class MultiLabelFocalLoss(torch.nn.Module):
            def __init__(self, gamma, pos_weight, label_smoothing=0.0):
                super().__init__()
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                self.register_buffer("pos_weight", pos_weight)

            def forward(self, logits, targets):
                # Phase 1: 应用标签平滑
                if self.label_smoothing > 0:
                    targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none", pos_weight=self.pos_weight,
                )
                probs = torch.sigmoid(logits)
                p_t = targets * probs + (1 - targets) * (1 - probs)
                return ((1 - p_t) ** self.gamma * bce).mean()

        criterion = MultiLabelFocalLoss(focal_gamma, pos_weight, label_smoothing)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # VAP 辅助损失（多任务）。仅 use_vap 时启用，对未来双声道 VA 做 BCE。
    vap_criterion = torch.nn.BCEWithLogitsLoss() if use_vap else None

    max_steps_per_epoch_cfg = cfg["train"].get("max_steps_per_epoch", None)
    max_steps_per_epoch = (
        int(max_steps_per_epoch_cfg) if max_steps_per_epoch_cfg is not None else None
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    max_epochs = int(cfg["train"]["epochs"])
    accum_steps = int(cfg["train"]["gradient_accumulation_steps"])
    steps_per_epoch_for_sched = (
        min(len(train_loader), max_steps_per_epoch)
        if max_steps_per_epoch is not None
        else len(train_loader)
    )
    total_update_steps = max(
        1, (steps_per_epoch_for_sched * max_epochs) // max(1, accum_steps)
    )
    warmup_ratio = float(cfg["train"].get("warmup_ratio", 0.03))
    warmup_steps = int(total_update_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["use_amp"]))

    start_epoch = 0
    best_metric = -math.inf
    best_path = Path(paths["checkpoints_dir"]) / _get_best_checkpoint_name(cfg)
    # Top-N ensemble checkpoints (besides the single best). Each member keeps its
    # own per-label thresholds; manifest lists them sorted by valid metric desc.
    ensemble_topk = int(cfg["train"].get("ensemble_topk", 0))
    topk_members: list[dict] = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", -math.inf))

    # 权重 EMA（仅可训练参数，rank0）。在 resume 之后创建，shadow 从已加载权重起步。
    # 注：保存的 best/ensemble 权重是 EMA 版（推理更稳）；--resume 会从 EMA 权重热启动，
    #     与原始权重略有出入，但训练能很快恢复（fresh run 不受影响）。
    raw_model = model.module if hasattr(model, "module") else model
    use_ema = bool(cfg["train"].get("use_ema", False))
    ema = EMA(raw_model, cfg["train"].get("ema_decay", 0.999)) if (use_ema and is_main) else None
    if use_ema and is_main:
        print(f"[EMA] enabled, decay={float(cfg['train'].get('ema_decay', 0.999))}")

    grad_clip = float(cfg["train"]["grad_clip_norm"])
    use_amp = bool(cfg["train"]["use_amp"])
    save_metric = str(cfg["train"]["save_metric"])
    early_stop_patience = int(cfg["train"]["early_stop_patience"])
    bad_epochs = 0
    # 真实训练步数（不受 len(train_loader) 误导）；用于 TensorBoard / 打印
    global_train_step = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        iterator = train_loader
        if is_main:
            iterator = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)

        epoch_loss_sum = 0.0
        epoch_step_count = 0
        log_every = int(cfg["train"].get("log_every_steps", 20))
        ema_decay = float(cfg["train"].get("ema_decay", 0.98))
        loss_ema = None
        update_step = 0
        last_metrics_valid = None
        last_metrics_test = None
        for step, batch in enumerate(iterator):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                if is_main:
                    print(
                        f"[Epoch {epoch}] reach max_steps_per_epoch={max_steps_per_epoch}, "
                        "run eval and continue next epoch."
                    )
                break
            waveform = batch["waveform"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            context_labels = batch["context_labels"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            vap_target = batch.get("vap_target")
            if vap_target is not None:
                vap_target = vap_target.to(device, non_blocking=True)
            vap_feat = batch.get("vap_feat")
            if vap_feat is not None:
                vap_feat = vap_feat.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_vap:
                    logits, vap_logits = model(
                        waveform=waveform,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        context_labels=context_labels,
                        return_vap=True,
                        vap_feat=vap_feat,
                    )
                    main_loss = criterion(logits, labels)
                    # vap_target [B,2,bins] -> [B,2*bins]，与 vap_head 输出对齐
                    vap_loss = vap_criterion(vap_logits, vap_target.flatten(1))
                    loss = (main_loss + vap_weight * vap_loss) / accum_steps
                else:
                    logits = model(
                        waveform=waveform,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        context_labels=context_labels,
                        vap_feat=vap_feat,
                    )
                    loss = criterion(logits, labels) / accum_steps

            if not torch.isfinite(loss):
                if is_main:
                    print(
                        f"[WARN] non-finite loss at epoch={epoch} step={step}, "
                        "skip this batch."
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            loss_value = float(loss.item() * accum_steps)
            epoch_loss_sum += loss_value
            epoch_step_count += 1
            global_train_step += 1
            global_step = global_train_step
            loss_ema = loss_value if loss_ema is None else (ema_decay * loss_ema + (1.0 - ema_decay) * loss_value)

            with torch.no_grad():
                probs = torch.sigmoid(logits.detach())
                batch_pos_rate = float(labels.float().mean().item())
                prob_mean = float(probs.mean().item())
                prob_std = float(probs.std(unbiased=False).item())
                logit_mean = float(logits.detach().mean().item())
                logit_std = float(logits.detach().std(unbiased=False).item())
                # Per-batch diagnostic: split loss by positive/negative entries.
                per_entry = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.detach(),
                    labels,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                pos_mask = labels > 0.5
                neg_mask = ~pos_mask
                pos_loss = float(per_entry[pos_mask].mean().item()) if pos_mask.any() else 0.0
                neg_loss = float(per_entry[neg_mask].mean().item()) if neg_mask.any() else 0.0

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if ema is not None:
                    ema.update(raw_model, global_train_step)
            else:
                grad_norm = float("nan")

            if is_main and (step + 1) % log_every == 0:
                lr_now = float(optimizer.param_groups[0]["lr"])
                if writer is not None:
                    writer.add_scalar("train/loss_step", loss_value, global_step)
                    writer.add_scalar("train/loss_ema", float(loss_ema), global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)
                    writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    writer.add_scalar("train/logit_mean", logit_mean, global_step)
                    writer.add_scalar("train/logit_std", logit_std, global_step)
                    writer.add_scalar("train/prob_mean", prob_mean, global_step)
                    writer.add_scalar("train/prob_std", prob_std, global_step)
                    writer.add_scalar("train/batch_pos_rate", batch_pos_rate, global_step)
                    writer.add_scalar("train/pos_loss", pos_loss, global_step)
                    writer.add_scalar("train/neg_loss", neg_loss, global_step)
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        loss=f"{loss_value:.4f}",
                        ema=f"{float(loss_ema):.4f}",
                        lr=f"{lr_now:.2e}",
                        p=f"{batch_pos_rate:.3f}",
                        pm=f"{prob_mean:.3f}",
                    )
                # nohup/重定向到文件时 tqdm 用 \\r 刷新，日志看起来像没动；补一行纯文本便于 tail
                if not sys.stderr.isatty():
                    print(
                        f"[train] epoch={epoch} step={step} global_step={global_step} "
                        f"loss={loss_value:.4f} ema={float(loss_ema):.4f} lr={lr_now:.2e} "
                        f"pos_rate={batch_pos_rate:.3f} prob_mean={prob_mean:.3f}",
                        flush=True,
                    )

            # 训练过程中不做任何测试集相关评估

        if is_distributed():
            torch.distributed.barrier()

        skip_epoch_end_eval = False

        if is_main:
            if ema is not None:
                ema.store_and_copy_to(raw_model)  # 用 EMA 权重做评估与保存
            eval_model = model.module if hasattr(model, "module") else model
            metrics_valid = evaluate(
                eval_model,
                valid_loader,
                device,
                use_amp=use_amp,
                label_names=metric_label_names if use_multi_label else None,
                max_batches=eval_valid_max_batches,
            )
            metrics_test_epoch = None

            metric_value = float(metrics_valid[save_metric])
            save_json(Path(paths["logs_dir"]) / f"valid_epoch_{epoch}.json", metrics_valid)
            # 兼容旧脚本读取路径
            save_json(Path(paths["logs_dir"]) / f"eval_epoch_{epoch}.json", metrics_valid)
            valid_thresholds = _extract_best_thresholds(metrics_valid, metric_label_names)
            save_json(Path(paths["logs_dir"]) / f"thresholds_epoch_{epoch}.json", {"thresholds": valid_thresholds})
            if use_multi_label:
                valid_per_label = _format_multilabel_metrics_line(metrics_valid, metric_label_names)
                print(
                    f"[Epoch {epoch}] train_loss={epoch_loss_sum / max(1, epoch_step_count):.6f} "
                    f"valid_macro_acc={metrics_valid['macro_accuracy']:.4f} "
                    f"valid_macro_f1={metrics_valid['macro_f1']:.4f} "
                    f"valid_macro_best_f1={metrics_valid['macro_best_f1']:.4f} "
                    f"valid_macro_auc={metrics_valid['macro_roc_auc']:.4f} "
                    f"| valid[{valid_per_label}] "
                    f"best_{save_metric}={max(best_metric, metric_value):.4f}"
                )
            else:
                print(
                    f"[Epoch {epoch}] train_loss={epoch_loss_sum / max(1, epoch_step_count):.6f} "
                    f"valid_acc={metrics_valid['accuracy']:.4f} valid_f1={metrics_valid['f1']:.4f} valid_auc={metrics_valid['roc_auc']:.4f} "
                    f"best_{save_metric}={max(best_metric, metric_value):.4f}"
                )
            if writer is not None:
                avg_train_loss = epoch_loss_sum / max(1, epoch_step_count)
                writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
                if use_multi_label:
                    writer.add_scalar("valid/macro_accuracy", metrics_valid["macro_accuracy"], epoch)
                    writer.add_scalar("valid/macro_f1", metrics_valid["macro_f1"], epoch)
                    writer.add_scalar("valid/macro_best_f1", metrics_valid["macro_best_f1"], epoch)
                    writer.add_scalar("valid/macro_roc_auc", metrics_valid["macro_roc_auc"], epoch)
                    for n in metric_label_names:
                        writer.add_scalar(f"valid/{n}_accuracy", metrics_valid[f"{n}_accuracy"], epoch)
                        writer.add_scalar(f"valid/{n}_f1", metrics_valid[f"{n}_f1"], epoch)
                        writer.add_scalar(f"valid/{n}_best_f1", metrics_valid[f"{n}_best_f1"], epoch)
                        writer.add_scalar(f"valid/{n}_best_threshold", metrics_valid[f"{n}_best_threshold"], epoch)
                        writer.add_scalar(f"valid/{n}_roc_auc", metrics_valid[f"{n}_roc_auc"], epoch)
                else:
                    writer.add_scalar("valid/accuracy", metrics_valid["accuracy"], epoch)
                    writer.add_scalar("valid/f1", metrics_valid["f1"], epoch)
                    writer.add_scalar("valid/roc_auc", metrics_valid["roc_auc"], epoch)

            # --- Top-N ensemble checkpoints (runs every epoch, independent of
            #     the global-best bookkeeping below, so a non-best-but-strong
            #     epoch can still join the ensemble pool). ---
            if ensemble_topk > 0:
                worst_kept = min((m["metric"] for m in topk_members), default=-math.inf)
                if len(topk_members) < ensemble_topk or metric_value > worst_kept:
                    member_name = f"ensemble_ep{epoch}.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "metric": metric_value,
                            # Slim: only fine-tuned params (~170MB vs ~5GB full).
                            "model": _trainable_state_dict(eval_model),
                            "trainable_only": True,
                            "config": cfg,
                            "thresholds": valid_thresholds,
                        },
                        Path(paths["checkpoints_dir"]) / member_name,
                    )
                    topk_members.append(
                        {
                            "name": member_name,
                            "epoch": epoch,
                            "metric": metric_value,
                            "thresholds": valid_thresholds,
                        }
                    )
                    topk_members.sort(key=lambda m: m["metric"], reverse=True)
                    while len(topk_members) > ensemble_topk:
                        evicted = topk_members.pop()
                        ev_path = Path(paths["checkpoints_dir"]) / evicted["name"]
                        if ev_path.exists():
                            ev_path.unlink()
                    save_json(
                        Path(paths["logs_dir"]) / "ensemble_manifest.json",
                        {"save_metric": save_metric, "members": topk_members},
                    )

            if metric_value > best_metric:
                bad_epochs = 0
                best_metric = metric_value
                torch.save(
                    {
                        "epoch": epoch,
                        "best_metric": best_metric,
                        "model": eval_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": cfg,
                        "thresholds": valid_thresholds,
                    },
                    best_path,
                )
                save_json(Path(paths["logs_dir"]) / "best_thresholds.json", {"thresholds": valid_thresholds})
            else:
                bad_epochs += 1

            if ema is not None:
                ema.restore(raw_model)  # 还原原始权重，下个 epoch 从原始权重继续训练

            if bad_epochs >= early_stop_patience:
                break

        if is_distributed():
            torch.distributed.barrier()

    cleanup_distributed()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
