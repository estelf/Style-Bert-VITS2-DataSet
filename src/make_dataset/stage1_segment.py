"""Stage 1: 音声セグメント化. Silero-VAD で文単位スライス + エネルギーベース境界補正 (±約70ms) + 無音トリム."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .manifest import Record
from .stage0_ingest import Source


@dataclass
class Segment:
    record: Record
    audio441: np.ndarray  # 補正後境界で切った 44.1k モノラル音声 (スコアリング/出力用)
    audio16k: np.ndarray  # ASR/埋め込み用


def _refine_boundary(audio: np.ndarray, sr: int, pos: int, side: str, window: int, rms_all: float) -> int:
    """VAD 境界を ±window サンプルの範囲で「無音寄りの谷」へ寄せる (強制アライメント代替ドラフト)."""
    lo, hi = max(0, pos - window), min(len(audio), pos + window)
    frame = int(sr * 0.005)  # 5ms
    best, best_e = pos, np.inf
    for p in range(lo, hi, frame):
        e = float(np.sqrt(np.mean(audio[p : p + frame] ** 2))) if p < len(audio) else np.inf
        if e < best_e:
            best, best_e = p, e
    # 明らかな無音区間へしか寄せない (話者ノイズ床より下なら谷を採用、でなければ VAD 境界維持)
    return best if best_e < rms_all * 0.25 else pos


def segment_sources(sources: list[Source], cfg: Config) -> list[Segment]:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    pad_ms = cfg.get("stage1.speech_pad_ms")
    refine_win = cfg.get("stage1.refine_window_ms")

    segments: list[Segment] = []
    for src in sources:
        wavs = [torch.from_numpy(src.audio16k)]
        ts = get_speech_timestamps(
            wavs[0], model, sampling_rate=16000, speech_pad_ms=pad_ms, return_seconds=False
        )
        win441 = int(44100 * refine_win / 1000)
        rms_all = float(np.sqrt(np.mean(src.audio441**2))) if len(src.audio441) else 0.0
        for i, t in enumerate(ts):
            s441, e441 = int(t["start"] * 44100 / 16000), int(t["end"] * 44100 / 16000)
            rs = _refine_boundary(src.audio441, 44100, s441, "start", win441, rms_all)
            re_ = _refine_boundary(src.audio441, 44100, e441, "end", win441, rms_all)
            if re_ - rs >= int(44100 * cfg.get("stage1.min_seg_sec")):  # 補正で極端に縮んだら元境界へ退避
                s441, e441 = rs, re_
            if e441 - s441 < int(44100 * cfg.get("stage1.min_seg_sec")):
                continue
            rec = Record(
                utt_id=f"{src.src_hash}-{i:03d}",
                src=src.path.name,
                start=s441 / 44100,
                end=e441 / 44100,
            )
            segments.append(
                Segment(
                    record=rec,
                    audio441=src.audio441[s441:e441],
                    audio16k=src.audio16k[t["start"] : t["end"]],
                )
            )
    return segments
