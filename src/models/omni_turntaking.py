"""
Qwen2.5-Omni-3B 原生 turn-taking 模型。

设计（与 whisper+qwen 的 LMF 方案完全独立）：
  音频 + ASR + 历史事件 → 一段多模态对话 → Omni **thinker** 联合编码
  → 取 thinker 最后一层 hidden state 池化 → 5 路 sigmoid 分类头。
不做生成解码（结构化多标签 + 逐标签阈值寻优，生成式无法阈值校准且慢）。

微调：LoRA 适配 thinker 的注意力/MLP 投影，骨干冻结；只训练 LoRA + 分类头。
保存时配合 train.py 的 _trainable_state_dict（只存 requires_grad 参数）→ 瘦身 checkpoint，
推理时重建同样结构再 strict=False 叠加即可（与 ensemble/软投票基建一致）。

⚠️ 首次在服务器跑通时重点核对（不同 transformers 版本类名/字段可能不同）：
  1) 类名：Qwen2_5OmniForConditionalGeneration / Qwen2_5OmniProcessor（transformers>=4.52）
  2) enable_audio_output=False 是否被支持（用于不加载 talker）
  3) thinker.forward 是否接受 processor 的全部键并支持 output_hidden_states
都用 try/except 做了兜底，报错时按提示改这一个文件即可。
"""

from __future__ import annotations

import re
from typing import Dict, List

import torch
import torch.nn as nn


def _dtype_from_str(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.bfloat16)


def build_omni_processor(model_path: str):
    """加载 Qwen2.5-Omni processor（含 tokenizer + 音频 feature extractor）。"""
    from transformers import Qwen2_5OmniProcessor

    return Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)


def _load_thinker(omni_cfg: Dict):
    """加载 Omni，仅保留 thinker（可选不加载 talker 以省显存）。"""
    from transformers import Qwen2_5OmniForConditionalGeneration

    dtype = _dtype_from_str(omni_cfg.get("torch_dtype", "bfloat16"))
    load_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    attn = omni_cfg.get("attn_implementation")
    if attn:
        load_kwargs["attn_implementation"] = attn

    enable_audio_output = bool(omni_cfg.get("enable_audio_output", False))
    base = None
    try:
        base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            omni_cfg["model_path"], enable_audio_output=enable_audio_output, **load_kwargs
        )
    except TypeError:
        # 旧/新版本不接受 enable_audio_output 关键字：正常加载后手动丢弃 talker。
        base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            omni_cfg["model_path"], **load_kwargs
        )

    thinker = getattr(base, "thinker", base)  # 若直接是 thinker 类则用其本身
    # 释放 talker（若存在且未被 enable_audio_output 阻止加载）
    if hasattr(base, "talker") and base.talker is not None:
        base.talker = None
    return thinker


def _hidden_size(thinker) -> int:
    cfg = thinker.config
    for attr in ("hidden_size",):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise RuntimeError("无法从 thinker.config 推断 hidden_size，请手动指定。")


def _apply_lora(thinker, lora_cfg: Dict):
    """对 thinker 注入 LoRA。restrict_to_language_model=True 时用正则排除 audio_tower。"""
    from peft import LoraConfig, get_peft_model

    suffixes = list(lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    if bool(lora_cfg.get("restrict_to_language_model", True)):
        # 只匹配不含 audio/visual/tower 的模块路径，避免给音频塔加 LoRA。
        alt = "|".join(re.escape(s) for s in suffixes)
        target = rf"^(?!.*(?:audio|visual|tower)).*\.(?:{alt})$"
    else:
        target = suffixes  # 列表 = 按后缀匹配（含音频塔）

    lconf = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        target_modules=target,
        task_type=None,  # 自定义头，不用 peft 的 CAUSAL_LM 包装
    )
    return get_peft_model(thinker, lconf)


class OmniTurnTaking(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        omni_cfg = cfg["omni"]
        self.pooling = str(omni_cfg.get("pooling", "masked_mean"))

        thinker = _load_thinker(omni_cfg)
        self.hidden_size = _hidden_size(thinker)

        # LoRA（in-place 注入 thinker，返回 PeftModel；self.thinker 持有可训练 LoRA 参数）
        self.thinker = _apply_lora(thinker, cfg.get("lora", {}))

        if bool(omni_cfg.get("gradient_checkpointing", True)):
            try:
                self.thinker.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.thinker.gradient_checkpointing_enable()
            if hasattr(self.thinker, "enable_input_require_grads"):
                self.thinker.enable_input_require_grads()

        n_labels = len(cfg["labels"]["multi_targets"])
        head_cfg = cfg.get("head", {})
        head_hidden = int(head_cfg.get("hidden_dim", 512))
        head_drop = float(head_cfg.get("dropout", 0.2))
        # 分类头保持 fp32（数值稳定）；池化后的 hidden 会 cast 到 fp32 再进头。
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(head_hidden, n_labels),
        )

    def _thinker_last_hidden(self, inputs: Dict) -> torch.Tensor:
        """跑 thinker，返回最后一层 hidden state [B,T,H]。用 logits_to_keep=1 省 lm_head 显存。"""
        kwargs = dict(output_hidden_states=True, return_dict=True, use_cache=False)
        try:
            out = self.thinker(**inputs, logits_to_keep=1, **kwargs)
        except TypeError:
            try:
                out = self.thinker(**inputs, num_logits_to_keep=1, **kwargs)
            except TypeError:
                out = self.thinker(**inputs, **kwargs)
        return out.hidden_states[-1]

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None or attention_mask.shape[1] != hidden.shape[1]:
            return hidden.mean(dim=1)  # 兜底：对齐不上时直接全均值
        if self.pooling == "last_token":
            lengths = attention_mask.sum(dim=1).long() - 1  # 右 padding：最后一个有效位
            idx = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[idx, lengths]
        # masked_mean
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, **inputs) -> torch.Tensor:
        # 不把 label/segment_id 传给 thinker（train 循环已剥离，这里再保险一次）
        inputs.pop("label", None)
        inputs.pop("segment_id", None)
        attention_mask = inputs.get("attention_mask", None)
        hidden = self._thinker_last_hidden(inputs)        # [B,T,H], bf16
        pooled = self._pool(hidden, attention_mask)        # [B,H]
        logits = self.head(pooled.float())                 # [B, n_labels], fp32
        return logits
