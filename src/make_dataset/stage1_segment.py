"""Stage 1: 音声セグメント化. Silero-VAD で文単位スライス + エネルギーベース境界補正 + 中間キャッシュ書き出し.

- D3/D4: silero-vad の主要パラメータ (threshold / min_silence_duration_ms / min_speech_duration_ms /
  max_speech_duration_s) を filters.toml から受け取る。長い塊は破棄せず VAD 側で分割する。
- C8/C11: 境界補正は side を見る。「始端=最初の話声フレーム、終端=最後の話声フレーム」へ寄せ、
  無音パディングをトリムする (語尾を自分で切りにいく挙動をやめる)。閾値 (騒音床) はセグメント
  自身の周辺から推定し、入力の無音率に依存しない (C9)。
- C10: 44.1k 出力スライスと 16k ASR 入力は同一境界から作る (16k は 44.1k 側からリサンプル)。
- E1/A3: 全セグメントを中間キャッシュ .cache/ に書き、納品用 raw/ と分離する。rescore はここから再生成。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import Config
from .manifest import Record
from .stage0_ingest import Source, decode, resample_to


def _frames_rms(audio: np.ndarray, sr: int, frame: int) -> tuple[np.ndarray, int]:
    n = len(audio) // frame * frame
    frames = audio[:n].reshape(-1, frame)
    return np.sqrt((frames**2).mean(axis=1)), frame


def _refine_boundary(audio: np.ndarray, sr: int, pos: int, side: str, window: int) -> int:
    """VAD 境界を「話声の実際の始端/終端」へ寄せる (強制アライメント代替ドラフト, C8/C11).

    start: [pos-w, pos+w] で最初に騒音床を超えるフレーム = 話声開始。それより後ろへは寄せない
           (語頭を削らない) ので、パディング分は削れても語頭自体は落ちない。
    end:   [pos-w, pos+w] で最後に騒音床を超えていたフレームの終端 = 話声終了。語尾切れを作らない。
    騒音床はそのセグメント周辺のエネルギー分布から推定する (C9: ソース全体平均に依存しない)。
    """
    frame = int(sr * 0.005)  # 5ms
    if len(audio) < 2 * window + frame or pos <= 0 or pos >= len(audio) - 1:
        return pos
    lo, hi = max(0, pos - window), min(len(audio), pos + window)
    ctx_lo, ctx_hi = max(0, pos - 8 * window), min(len(audio), pos + 8 * window)
    fe_ctx, _ = _frames_rms(audio[ctx_lo:ctx_hi], sr, frame)
    floor = float(np.percentile(fe_ctx, 20)) * 3.0  # ローカルな騒音床推定 (C9)
    if not np.isfinite(floor) or floor <= 0:
        return pos
    step = window // 4 or 1
    best = pos
    if side == "start":
        for p in range(lo, min(hi, len(audio) - frame), step):
            if float(np.sqrt(np.mean(audio[p : p + frame] ** 2))) > floor:
                return max(0, p - frame)  # 話声始端の少し前から切る (語頭保護)
        return pos
    for p in range(lo, min(hi - frame, len(audio) - frame), step):
        if float(np.sqrt(np.mean(audio[p : p + frame] ** 2))) > floor:
            best = p + frame  # 話声終端の少し後ろまで残す (語尾保護)
    return int(min(best, len(audio)))


def segment_sources(sources: list[Source], cfg: Config, cache_dir: Path) -> list[Record]:
    """ソースを 1 本ずつデコード→分割→キャッシュ書き出し。フル尺波形は同時保持しない (F2)."""
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    s1 = cfg.data["stage1"]
    pad_ms, win_ms = s1["speech_pad_ms"], s1["refine_window_ms"]
    min_seg_sec = s1["min_seg_sec"]
    records: list[Record] = []

    for src in tqdm(sources, desc="stage1 segment", unit="src"):
        try:  # B2: 1 ソースの異常でラン全体を止めない
            x441 = decode(src.path, 44100)
            x16 = resample_to(
                x441, 44100, 16000
            )  # F6: デコードは 1 回だけ (silero は 8k/16k しか扱えない)
            ts = get_speech_timestamps(
                torch.from_numpy(x16),
                model,
                sampling_rate=16000,
                threshold=s1["vad_threshold"],
                min_speech_duration_ms=s1["min_speech_duration_ms"],
                min_silence_duration_ms=s1["min_silence_duration_ms"],
                max_speech_duration_s=s1[
                    "max_speech_duration_s"
                ],  # D3: 長い塊は分割する (破棄しない)
                speech_pad_ms=pad_ms,
                return_seconds=False,
            )
            win441 = int(44100 * win_ms / 1000)
            for i, t in enumerate(ts):  # 16k サンプル → 44.1k サンプルへ換算
                s441, e441 = int(t["start"] * 44100 / 16000), int(t["end"] * 44100 / 16000)
                rs = _refine_boundary(x441, 44100, s441, "start", win441)
                re_ = _refine_boundary(x441, 44100, e441, "end", win441)
                if re_ - rs >= int(44100 * min_seg_sec):  # 補正で極端に縮んだら元境界へ退避
                    s441, e441 = rs, re_
                if e441 - s441 < int(44100 * min_seg_sec):
                    continue
                rec = Record(
                    utt_id=f"{src.src_hash}-{i:03d}",
                    src=src.path.name,
                    start=s441 / 44100,
                    end=e441 / 44100,
                )
                seg441 = x441[
                    s441:e441
                ]  # C10: 出力も ASR 入力も同じ補正後境界 (16k はここから作る)
                cache_dir.mkdir(parents=True, exist_ok=True)
                sf.write(cache_dir / f"{rec.utt_id}.flac", seg441, 44100, format="FLAC")
                rec.scores["seg_dur"] = round((e441 - s441) / 44100, 3)
                records.append(rec)
        except Exception as e:  # noqa: BLE001
            import typer

            typer.echo(f"  [stage1] {src.path.name}: 処理失敗でスキップ: {e!r}")
    return records
