import json
import random
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class TrainSample:
    conv_id: str
    end_idx: int
    label: int


@dataclass
class TrainSampleMulti:
    conv_id: str
    end_idx: int
    # (BC, I, T) or any configured order
    label_vec: Tuple[int, ...]


def list_conv_ids(labels_dir: Path) -> List[str]:
    return sorted([p.stem for p in labels_dir.glob("*.npy")])


def split_conversation_ids(
    conv_ids: Sequence[str], valid_ratio: float, seed: int
) -> Dict[str, List[str]]:
    conv_ids = list(conv_ids)
    random.Random(seed).shuffle(conv_ids)
    valid_size = max(1, int(len(conv_ids) * valid_ratio))
    valid_ids = sorted(conv_ids[:valid_size])
    train_ids = sorted(conv_ids[valid_size:])
    return {"train": train_ids, "valid": valid_ids}


def build_train_samples(
    labels_dir: Path,
    conv_ids: Sequence[str],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    positive_ids: Sequence[int],
    max_samples: Optional[int] = None,
) -> List[TrainSample]:
    samples: List[TrainSample] = []
    pos_set = set(positive_ids)
    for conv_id in conv_ids:
        labels = np.load(labels_dir / f"{conv_id}.npy")
        max_end = labels.shape[0] - target_chunks
        for end_idx in range(context_chunks, max_end + 1, stride):
            future = labels[end_idx : end_idx + target_chunks]
            y = int(any(int(x) in pos_set for x in future))
            samples.append(TrainSample(conv_id=conv_id, end_idx=end_idx, label=y))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def build_train_samples_multitask(
    labels_dir: Path,
    conv_ids: Sequence[str],
    context_chunks: int,
    target_chunks: int,
    stride: int,
    label_ids: Dict[str, int],
    target_labels: Sequence[str] = ("BC", "I", "T"),
    max_samples: Optional[int] = None,
) -> List[TrainSampleMulti]:
    samples: List[TrainSampleMulti] = []
    target_id_list = [int(label_ids[k]) for k in target_labels]
    for conv_id in conv_ids:
        labels = np.load(labels_dir / f"{conv_id}.npy")
        max_end = labels.shape[0] - target_chunks
        for end_idx in range(context_chunks, max_end + 1, stride):
            future = labels[end_idx : end_idx + target_chunks]
            y_vec = tuple(int(any(int(x) == tid for x in future)) for tid in target_id_list)
            samples.append(TrainSampleMulti(conv_id=conv_id, end_idx=end_idx, label_vec=y_vec))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def _speaker_token(channel_id: int) -> str:
    return "[SPK1]" if channel_id == 1 else "[SPK2]"


def build_text_context(
    utterances: Iterable[Dict],
    start_ms: int,
    end_ms: int,
    max_utterances: int = 120,
) -> str:
    selected = []
    for utt in utterances:
        utt_start = int(utt.get("start_ms", 0))
        utt_end = int(utt.get("end_ms", utt_start))
        if utt_end <= start_ms or utt_start >= end_ms:
            continue
        text = str(utt.get("text", "")).strip()
        if not text:
            continue
        selected.append(f"{_speaker_token(int(utt.get('channel_id', 1)))} {text}")
    if not selected:
        return "[SPK1] <silence> [SPK2] <silence>"
    if len(selected) > max_utterances:
        selected = selected[-max_utterances:]
    return " ".join(selected)


def _read_wav_slice(path: Path, start_ms: int, end_ms: int) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        n_ch = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        start_frame = max(0, int(start_ms * sr / 1000))
        end_frame = max(start_frame + 1, int(end_ms * sr / 1000))
        total_frames = wf.getnframes()
        end_frame = min(end_frame, total_frames)
        read_frames = max(1, end_frame - start_frame)
        wf.setpos(start_frame)
        raw = wf.readframes(read_frames)

    if sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
    elif sampwidth == 1:
        dtype = np.uint8
        scale = 128.0
    else:
        raise RuntimeError(f"Unsupported wav sample width {sampwidth} for {path}")

    data = np.frombuffer(raw, dtype=dtype)
    if n_ch > 1:
        data = data.reshape(-1, n_ch)
    else:
        data = data[:, None]

    if sampwidth == 1:
        data = (data.astype(np.float32) - 128.0) / scale
    else:
        data = data.astype(np.float32) / scale
    return data, sr


class TurnTakingTrainDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[TrainSample | TrainSampleMulti],
        train_audio_dir: Path,
        train_text_dir: Path,
        train_labels_dir: Path,
        context_chunks: int,
        target_chunks: int,
        chunk_ms: int,
        sample_rate: int,
        augment_audio: bool = True,
        dynamic_context: bool = False,
        min_context_chunks: int = 125,
        max_context_chunks: int = 375,
        context_prob: float = 0.5,
        vap_target: bool = False,
        vap_bins: int = 25,
        vad_log_offset: float = 2.0,
        vap_feat_dir: Optional[str] = None,
        vap_frame_rate: float = 10.0,
        vap_feat_dim: int = 18,
    ) -> None:
        self.samples = list(samples)
        self.train_audio_dir = train_audio_dir
        self.train_text_dir = train_text_dir
        self.train_labels_dir = train_labels_dir
        self.context_chunks = context_chunks
        self.target_chunks = target_chunks
        self.chunk_ms = chunk_ms
        self.sample_rate = sample_rate
        self.augment_audio = augment_audio
        # Phase 1: 动态上下文配置
        self.dynamic_context = dynamic_context
        self.min_context_chunks = min_context_chunks
        self.max_context_chunks = max_context_chunks
        self.context_prob = context_prob
        # VAP 辅助目标（仅训练）：未来 target_chunks 的双声道语音活动 [2, vap_bins]
        self.vap_target = vap_target
        self.vap_bins = int(vap_bins)
        self.vad_log_offset = float(vad_log_offset)
        # VAP 特征(预计算)：晚融合第5模态，按 end_idx 映射到 vap 帧读取
        self.vap_feat_dir = Path(vap_feat_dir) if vap_feat_dir else None
        self.vap_frame_rate = float(vap_frame_rate)
        self.vap_feat_dim = int(vap_feat_dim)

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=256)
    def _load_labels(self, conv_id: str) -> np.ndarray:
        return np.load(self.train_labels_dir / f"{conv_id}.npy")

    @lru_cache(maxsize=256)
    def _load_text_json(self, conv_id: str) -> Dict:
        with open(self.train_text_dir / f"{conv_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @lru_cache(maxsize=64)
    def _load_vap_feats(self, conv_id: str):
        """读该会话预计算的 VAP 逐帧特征 [F, feat_dim]；缺失返回 None。"""
        if self.vap_feat_dir is None:
            return None
        path = self.vap_feat_dir / f"{conv_id}.npy"
        if not path.exists():
            return None
        return np.load(path)

    def _load_wave_segment(self, conv_id: str, start_ms: int, end_ms: int) -> torch.Tensor:
        wav_path = self.train_audio_dir / f"{conv_id}.wav"
        audio, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio.T)  # [C, T]
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]

        if src_sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, src_sr, self.sample_rate)

        expected_frames = int((end_ms - start_ms) * self.sample_rate / 1000)
        if wave.shape[1] < expected_frames:
            pad = expected_frames - wave.shape[1]
            wave = torch.nn.functional.pad(wave, (0, pad))
        elif wave.shape[1] > expected_frames:
            wave = wave[:, :expected_frames]
        return wave

    def _augment_audio(self, wave: torch.Tensor) -> torch.Tensor:
        """SpecAugment-style time masking + Gaussian noise (training only)."""
        if not self.augment_audio:
            return wave
        T = wave.shape[1]
        # Random Gaussian noise
        if torch.rand(1).item() < 0.3:
            wave = wave + 0.005 * torch.randn_like(wave)
        # Random time mask (zero out a segment)
        if torch.rand(1).item() < 0.5:
            mask_len = int(T * 0.05 * torch.rand(1).item())
            mask_start = torch.randint(0, max(1, T - mask_len), (1,)).item()
            wave[:, mask_start:mask_start + mask_len] = 0
        return wave

    def _chunk_log_energy(self, x: torch.Tensor, spc: int) -> torch.Tensor:
        """每 spc 个采样点一个 chunk 的对数能量。x:[T] -> [n_chunks]"""
        n = x.shape[0] // spc
        if n == 0:
            return x.new_zeros(0)
        frames = x[: n * spc].reshape(n, spc)
        return torch.log(frames.pow(2).mean(dim=1) + 1e-8)

    def _future_va_grid(self, wave_ctx: torch.Tensor, wave_fut: torch.Tensor) -> torch.Tensor:
        """VAP 目标：未来双声道语音活动 [2, vap_bins]（能量级 VAD）。
        每声道用 ctx+fut 的低分位做自适应噪声底（抵消增益差异），fut 每 chunk
        能量 > 底 + vad_log_offset 判为有声。bins<n_chunks 时按"任一 chunk 有声"聚合。
        仅训练用，靠未来音频构造监督；推理不需要、不破坏 causal 约束。
        """
        spc = int(self.chunk_ms * self.sample_rate / 1000)
        n_chunks = self.target_chunks
        va = torch.zeros(2, n_chunks)
        for ch in range(min(2, wave_fut.shape[0])):
            loge_fut = self._chunk_log_energy(wave_fut[ch], spc)
            if loge_fut.numel() < n_chunks:
                pad_v = float(loge_fut.min()) if loge_fut.numel() > 0 else -18.0
                loge_fut = torch.nn.functional.pad(
                    loge_fut, (0, n_chunks - loge_fut.numel()), value=pad_v
                )
            loge_fut = loge_fut[:n_chunks]
            loge_ctx = self._chunk_log_energy(wave_ctx[ch], spc)
            ref = torch.cat([t for t in (loge_ctx, loge_fut) if t.numel() > 0])
            floor = torch.quantile(ref, 0.2)
            va[ch] = (loge_fut > (floor + self.vad_log_offset)).float()
        if self.vap_bins != n_chunks and n_chunks % self.vap_bins == 0:
            per = n_chunks // self.vap_bins
            va = va.reshape(2, self.vap_bins, per).amax(dim=2)
        return va  # [2, vap_bins]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        labels = self._load_labels(sample.conv_id)
        end_idx = sample.end_idx
        
        # Phase 1: 动态上下文长度
        if self.dynamic_context and random.random() < self.context_prob:
            actual_context = random.randint(self.min_context_chunks, self.max_context_chunks)
            start_idx = max(0, end_idx - actual_context)
        else:
            start_idx = max(0, end_idx - self.context_chunks)

        context_labels = labels[start_idx:end_idx].astype(np.int64)
        
        # 如果实际长度不足，需要padding
        if len(context_labels) < self.context_chunks:
            pad_len = self.context_chunks - len(context_labels)
            # 用NA标签(4)进行padding
            context_labels = np.concatenate([
                np.full(pad_len, 4, dtype=np.int64),
                context_labels
            ])
        
        start_ms = start_idx * self.chunk_ms
        end_ms = end_idx * self.chunk_ms

        text_json = self._load_text_json(sample.conv_id)
        text = build_text_context(text_json.get("utterances", []), start_ms, end_ms)
        wave = self._load_wave_segment(sample.conv_id, start_ms, end_ms)

        vap_grid = None
        if self.vap_target:
            # 从未增广的 clean 上下文 + 未来 2s 估双声道 VA（VAD 阈值靠 clean 音频）
            fut_end_ms = end_ms + self.target_chunks * self.chunk_ms
            wave_fut = self._load_wave_segment(sample.conv_id, end_ms, fut_end_ms)
            vap_grid = self._future_va_grid(wave, wave_fut)

        wave = self._augment_audio(wave)

        out = {
            "conv_id": sample.conv_id,
            "end_idx": end_idx,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
        }
        if vap_grid is not None:
            out["vap_target"] = vap_grid
        if self.vap_feat_dir is not None:
            arr = self._load_vap_feats(sample.conv_id)
            vf = np.zeros(self.vap_feat_dim, dtype=np.float32)
            if arr is not None and arr.shape[0] > 0:
                fr = int(round(end_idx * self.chunk_ms * self.vap_frame_rate / 1000.0))
                fr = min(max(fr, 0), arr.shape[0] - 1)
                vf = np.asarray(arr[fr], dtype=np.float32)
            out["vap_feat"] = torch.from_numpy(vf)
        if hasattr(sample, "label_vec"):
            out["label"] = torch.tensor(sample.label_vec, dtype=torch.float32)
        else:
            out["label"] = torch.tensor(float(sample.label), dtype=torch.float32)
        return out


class TurnTakingTestDataset(Dataset):
    # 上下文标签的 padding 值（与训练 TurnTakingTrainDataset 一致：labels.NA = 4）。
    PAD_LABEL = 4

    def __init__(
        self,
        test_root: Path,
        sample_rate: int,
        context_chunks: int = 375,
    ) -> None:
        self.sample_rate = sample_rate
        self.context_chunks = int(context_chunks)
        self.base = test_root
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
        # 归一化到固定 context_chunks（与训练 TurnTakingTrainDataset 完全一致）：
        # 截取最后 N 个 chunk；不足则前面用 NA=4 补齐。复赛上下文为 (0,30] 动态时长，
        # 这样无论官方落盘是定长 375 还是变长，batch 内长度都统一且 >= tail_k，
        # 不会在 collate 的 torch.stack 或模型 ContextLabelEncoder 的 tail 分支处报错。
        N = self.context_chunks
        L = len(context_labels)
        if L >= N:
            context_labels = context_labels[-N:]
        else:
            context_labels = np.concatenate(
                [np.full(N - L, self.PAD_LABEL, dtype=np.int64), context_labels]
            )
        text_json = self._load_text_json(seg_id)
        start_ms = int(text_json.get("start_ms", 0))
        end_ms = int(text_json.get("end_ms", 30000))
        text = build_text_context(text_json.get("utterances", []), start_ms, end_ms)

        wav_path = self.audio_dir / f"{seg_id}.wav"
        audio, src_sr = _read_wav_slice(wav_path, start_ms, end_ms)
        wave = torch.from_numpy(audio.T)
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]
        if src_sr != self.sample_rate:
            wave = torchaudio.functional.resample(wave, src_sr, self.sample_rate)

        return {
            "segment_id": seg_id,
            "waveform": wave,
            "text": text,
            "context_labels": torch.from_numpy(context_labels),
        }


class CollateFn:
    """Module-level callable class for DataLoader collate_fn (picklable on Windows)."""

    def __init__(self, tokenizer, text_max_length: int):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = "left"
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        tokenized = self.tokenizer(
            texts,
            max_length=self.text_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        waves = [b["waveform"] for b in batch]
        max_len = max(w.shape[1] for w in waves)
        padded_waves = []
        for w in waves:
            if w.shape[1] < max_len:
                w = torch.nn.functional.pad(w, (0, max_len - w.shape[1]))
            padded_waves.append(w)

        out = {
            "waveform": torch.stack(padded_waves, dim=0),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "context_labels": torch.stack([b["context_labels"] for b in batch], dim=0),
        }

        if "vap_target" in batch[0]:
            out["vap_target"] = torch.stack([b["vap_target"] for b in batch], dim=0)

        if "vap_feat" in batch[0]:
            out["vap_feat"] = torch.stack([b["vap_feat"] for b in batch], dim=0)

        if "label" in batch[0]:
            out["label"] = torch.stack([b["label"] for b in batch], dim=0)
            out["conv_id"] = [b["conv_id"] for b in batch]
            out["end_idx"] = [b["end_idx"] for b in batch]
        else:
            out["segment_id"] = [b["segment_id"] for b in batch]
        return out


def build_collate_fn(tokenizer, text_max_length: int) -> CollateFn:
    return CollateFn(tokenizer, text_max_length)
