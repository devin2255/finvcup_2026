"""
Qwen2.5-Omni-3B 原生数据管线。

与 whisper+qwen 方案的关键区别：**不再拆成多个模态分支**，而是把
「音频 + ASR 文本 + 历史 chunk 标签」铺进一段多模态对话，交给 Omni processor
打包成 thinker 的输入。历史标签没有结构入口，统一做 RLE 文本化塞进 user 文本。

复用 dataset.py 的样本/标签/音频切片/ASR 文本构造（窗口与标签口径与 whisper 方案一致）：
- build_train_samples_multitask：未来 target_chunks 内每个标签是否出现 → label_vec
- _read_wav_slice：按 ms 切音频
- build_text_context：按说话人标签拼 ASR 文本

__getitem__ 只返回 python 原料（mono 音频 np、user 文本、label）；
真正的多模态打包在 OmniCollate 里用 processor 完成（与 whisper 方案在 collate 里
tokenizer 的做法对齐）。
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .dataset import (
    _read_wav_slice,
    build_text_context,
    resolve_test_root,
)

# Omni 默认任务系统提示（不做语音生成，仅用于给 thinker 表征加任务条件）。
DEFAULT_SYSTEM_PROMPT = (
    "你是对话轮次预测助手。给定过去最多30秒的双人对话音频、其ASR转写文本，"
    "以及最近若干80ms片段(chunk)的历史事件标注，请判断【未来2秒(25个chunk)】内是否会出现"
    "以下5类语音事件：C(继续/正常说话)、NA(无活动/静音)、I(插话)、BC(反馈词/backchannel)、T(话轮转换)。"
)


def build_id_to_name(labels_cfg: Dict) -> Dict[int, str]:
    """从 labels 配置(形如 {C:0,T:1,BC:2,I:3,NA:4}) 反推 id->名称。"""
    id2name: Dict[int, str] = {}
    for name, idx in labels_cfg.items():
        if isinstance(idx, int):  # 跳过 positive_ids / multi_targets 这类非整数项
            id2name[int(idx)] = str(name)
    return id2name


def rle_verbalize(seq: Sequence[int], chunk_ms: int, id2name: Dict[int, str]) -> str:
    """把历史标签序列做运行长度编码并带上时长，oldest→newest。

    例: [4,4,4,0,0,1] -> "NA(0.24s) C(0.16s) T(0.08s)"
    """
    if len(seq) == 0:
        return "(无历史)"
    parts: List[str] = []
    prev = int(seq[0])
    run = 1
    for x in list(seq)[1:]:
        x = int(x)
        if x == prev:
            run += 1
        else:
            parts.append(f"{id2name.get(prev, str(prev))}({run * chunk_ms / 1000:.2f}s)")
            prev, run = x, 1
    parts.append(f"{id2name.get(prev, str(prev))}({run * chunk_ms / 1000:.2f}s)")
    return " ".join(parts)


def build_user_text(asr_text: str, history_text: str | None) -> str:
    """组装喂给 Omni 的 user 文本（ASR + 历史事件）。"""
    blocks = [f"【ASR转写】\n{asr_text}"]
    if history_text is not None:
        blocks.append(f"【历史事件(每段~80ms, 由旧到新)】\n{history_text}")
    blocks.append("请预测未来2秒内 C / NA / I / BC / T 是否各自出现。")
    return "\n".join(blocks)


def _to_mono_16k(wave_2ch: torch.Tensor, src_sr: int, sample_rate: int) -> np.ndarray:
    """[C,T] -> 单声道 float32 np（16k）。双声道取均值。"""
    if wave_2ch.shape[0] > 1:
        mono = wave_2ch.mean(dim=0, keepdim=True)
    else:
        mono = wave_2ch
    if src_sr != sample_rate:
        mono = torchaudio.functional.resample(mono, src_sr, sample_rate)
    return mono.squeeze(0).contiguous().to(torch.float32).numpy()


class OmniTurnTakingTrainDataset(Dataset):
    def __init__(
        self,
        samples: Sequence,
        train_audio_dir: Path,
        train_text_dir: Path,
        train_labels_dir: Path,
        context_chunks: int,
        chunk_ms: int,
        sample_rate: int,
        labels_cfg: Dict,
        history_include: bool = True,
        history_chunks: int = 125,
        dynamic_context: bool = False,
        min_context_chunks: int = 125,
        max_context_chunks: int = 375,
        context_prob: float = 0.5,
    ) -> None:
        self.samples = list(samples)
        self.train_audio_dir = Path(train_audio_dir)
        self.train_text_dir = Path(train_text_dir)
        self.train_labels_dir = Path(train_labels_dir)
        self.context_chunks = int(context_chunks)
        self.chunk_ms = int(chunk_ms)
        self.sample_rate = int(sample_rate)
        self.id2name = build_id_to_name(labels_cfg)
        self.history_include = bool(history_include)
        self.history_chunks = int(history_chunks)
        self.dynamic_context = bool(dynamic_context)
        self.min_context_chunks = int(min_context_chunks)
        self.max_context_chunks = int(max_context_chunks)
        self.context_prob = float(context_prob)

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=256)
    def _load_labels(self, conv_id: str) -> np.ndarray:
        return np.load(self.train_labels_dir / f"{conv_id}.npy")

    @lru_cache(maxsize=256)
    def _load_text_json(self, conv_id: str) -> Dict:
        with open(self.train_text_dir / f"{conv_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        labels = self._load_labels(sample.conv_id)
        end_idx = int(sample.end_idx)

        # 上下文窗口（可动态变长，与 whisper 方案一致的因果约束）
        if self.dynamic_context and random.random() < self.context_prob:
            actual_context = random.randint(self.min_context_chunks, self.max_context_chunks)
            start_idx = max(0, end_idx - actual_context)
        else:
            start_idx = max(0, end_idx - self.context_chunks)

        start_ms = start_idx * self.chunk_ms
        end_ms = end_idx * self.chunk_ms

        # 音频（mono 16k）
        wav_path = self.train_audio_dir / f"{sample.conv_id}.wav"
        audio_2ch, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio_2ch.T)  # [C,T]
        audio = _to_mono_16k(wave, src_sr, self.sample_rate)

        # ASR 文本
        text_json = self._load_text_json(sample.conv_id)
        asr_text = build_text_context(text_json.get("utterances", []), start_ms, end_ms)

        # 历史事件 RLE（取最近 history_chunks，因果）
        history_text = None
        if self.history_include:
            h_start = max(0, end_idx - self.history_chunks)
            hist_seq = labels[h_start:end_idx].astype(np.int64)
            history_text = rle_verbalize(hist_seq, self.chunk_ms, self.id2name)

        user_text = build_user_text(asr_text, history_text)
        return {
            "audio": audio,
            "user_text": user_text,
            "label": torch.tensor(sample.label_vec, dtype=torch.float32),
        }


class OmniTurnTakingTestDataset(Dataset):
    """复赛测试集：/xydata/{audio,text,context}，渲染方式与训练完全一致。"""

    PAD_LABEL = 4

    def __init__(
        self,
        test_root: Path,
        sample_rate: int,
        context_chunks: int,
        chunk_ms: int,
        labels_cfg: Dict,
        history_include: bool = True,
        history_chunks: int = 125,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.context_chunks = int(context_chunks)
        self.chunk_ms = int(chunk_ms)
        self.id2name = build_id_to_name(labels_cfg)
        self.history_include = bool(history_include)
        self.history_chunks = int(history_chunks)
        self.base = resolve_test_root(Path(test_root))
        self.audio_dir = self.base / "audio"
        self.text_dir = self.base / "text"
        self.context_dir = self.base / "context"
        self.segment_ids = sorted([p.stem for p in self.context_dir.glob("*.npy")])

    def __len__(self) -> int:
        return len(self.segment_ids)

    @lru_cache(maxsize=512)
    def _load_text_json(self, seg_id: str) -> Dict:
        with open(self.text_dir / f"{seg_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def __getitem__(self, idx: int) -> Dict:
        seg_id = self.segment_ids[idx]
        context_labels = np.load(self.context_dir / f"{seg_id}.npy").astype(np.int64)

        text_json = self._load_text_json(seg_id)
        start_ms = int(text_json.get("start_ms", 0))
        end_ms = int(text_json.get("end_ms", 30000))
        asr_text = build_text_context(text_json.get("utterances", []), start_ms, end_ms)

        history_text = None
        if self.history_include:
            hist_seq = context_labels[-self.history_chunks:]
            history_text = rle_verbalize(hist_seq, self.chunk_ms, self.id2name)

        wav_path = self.audio_dir / f"{seg_id}.wav"
        audio_2ch, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio_2ch.T)
        audio = _to_mono_16k(wave, src_sr, self.sample_rate)

        return {
            "segment_id": seg_id,
            "audio": audio,
            "user_text": build_user_text(asr_text, history_text),
        }


class OmniCollate:
    """用 Qwen2.5-Omni processor 把一个 batch 打包成 thinker 输入。

    每条样本是一段「system + user(音频占位 + 文本)」对话；processor 负责把音频
    展开成音频 token、文本 tokenize、并 padding 对齐。labels（若有）单独 stack。
    """

    def __init__(
        self,
        processor,
        sample_rate: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_text_tokens: int = 512,
    ):
        self.processor = processor
        self.sample_rate = int(sample_rate)
        self.system_prompt = system_prompt
        self.max_text_tokens = int(max_text_tokens)
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "padding_side"):
            tok.padding_side = "right"  # 池化用 attention_mask，右 padding 更直观

    def _conversation(self, audio: np.ndarray, user_text: str) -> List[Dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def __call__(self, batch: List[Dict]) -> Dict:
        audios = [b["audio"] for b in batch]
        conversations = [self._conversation(b["audio"], b["user_text"]) for b in batch]
        texts = [
            self.processor.apply_chat_template(
                c, add_generation_prompt=True, tokenize=False
            )
            for c in conversations
        ]
        inputs = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        out = dict(inputs)
        if "label" in batch[0]:
            out["label"] = torch.stack([b["label"] for b in batch], dim=0)
        else:
            out["segment_id"] = [b["segment_id"] for b in batch]
        return out


def build_omni_collate(
    processor,
    sample_rate: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_text_tokens: int = 512,
) -> OmniCollate:
    return OmniCollate(processor, sample_rate, system_prompt, max_text_tokens)
