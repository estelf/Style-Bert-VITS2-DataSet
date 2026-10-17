"""Stage 1: 音声セグメント化. Silero-VAD で文単位スライス + エネルギーベース境界補正 + 中間キャッシュ書き出し.

- silero-vad の主要パラメータ (threshold / min_silence_duration_ms / min_speech_duration_ms /
  max_speech_duration_s) を filters.toml から受け取る。長い塊は破棄せず VAD 側で分割する。
- 境界補正は side を見る。「始端=最初の話声フレーム、終端=最後の話声フレーム」へ寄せ、
  無音パディングをトリムする (語尾を自分で切りにいく挙動をやめる)。閾値 (騒音床) はセグメント
  自身の周辺から推定し、入力の無音率に依存しない。
- 44.1k 出力スライスと 16k ASR 入力は同一境界から作る (16k は 44.1k 側からリサンプル)。
- output_pad_ms は境界補正の**後**に効く納品用の無音マージン。speech_pad_ms (VAD 側の切り出し拡張) は
  補正窓の内側でトリムされて出力に残らないため、語頭/語尾のマージンを作りたいときはこちらを使う。
- 全セグメントを中間キャッシュ .cache/ に書き、納品用 raw/ と分離する。rescore はここから再生成。
- キャッシュ下限 (cache_min_sec) は閾値ゲート (min_seg_sec) と分離。下限未満の VAD 出力は
  中間キャッシュしないが、manifest 上にスキップ理由 (too_short) を残すので「どこへ消えたか」が見える。
- 長尺ソースは decode_chunk_min 分ずつ範囲デコードして保持メモリを有界化する (1 本の長尺動画で
  ラン全体が OOM するのを防ぐ)。所有領域 (owned) を開始時刻で決めるので境界の文は取りこぼし/重複しない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import Config
from .manifest import Record
from .stage0_ingest import Source, decode, decode_range, probe_duration, resample_to


def _refine_boundary(audio: np.ndarray, sr: int, pos: int, side: str, window: int) -> int:
    """VAD 境界を「話声の実際の始端/終端」へ寄せる (エネルギーベース境界補正).

    start: [pos-w, pos+w] で最初に騒音床を超えるフレーム = 話声開始。それより後ろへは寄せない
           (語頭を削らない) ので、パディング分は削れても語頭自体は落ちない。
    end:   [pos-w, pos+w] で最後に騒音床を超えていたフレームの終端 = 話声終了。語尾切れを作らない。
    騒音床はそのセグメント周辺のエネルギー分布から推定する (ソース全体平均に依存しない)。
    """
    frame = int(sr * 0.005)  # 5ms
    if len(audio) < 2 * window + frame or pos <= 0 or pos >= len(audio) - 1:
        return pos
    lo, hi = max(0, pos - window), min(len(audio), pos + window)
    ctx_lo, ctx_hi = max(0, pos - 8 * window), min(len(audio), pos + 8 * window)
    fe_ctx = _frames_rms(audio[ctx_lo:ctx_hi], sr, frame)
    floor = float(np.percentile(fe_ctx, 20)) * 3.0  # ローカルな騒音床推定
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


def _frames_rms(audio: np.ndarray, sr: int, frame: int) -> np.ndarray:
    n = len(audio) // frame * frame
    frames = audio[:n].reshape(-1, frame)
    return np.sqrt((frames**2).mean(axis=1))


def _chunk_windows(total: float, chunk_s: float, pad_before: float, pad_after: float):
    """長尺を (win_lo, win_hi, owned_lo, owned_hi) [秒] のチャンク列へ分割する.

    owned = そのチャンクが「開始時刻」で所有する発話の領域。win は owned を pad だけ前後に広げた
    デコード窓で、境界で切れる長い発話も pad_after 内で完結し、次のチャンクとは開始時刻で排他分割される。
    """
    k = 0
    while k * chunk_s < total:
        owned_lo = k * chunk_s
        owned_hi = min((k + 1) * chunk_s, total)
        win_lo = max(0.0, owned_lo - pad_before)
        win_hi = min(total, owned_hi + pad_after)
        yield win_lo, win_hi, owned_lo, owned_hi
        k += 1


def segment_sources(sources: list[Source], cfg: Config, cache_dir: Path) -> list[Record]:
    """ソースを 1 本ずつ (長尺はチャンクずつ) デコード→分割→キャッシュ書き出し。保持メモリを有界化."""
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    s1 = cfg.data["stage1"]
    pad_ms, win_ms = s1["speech_pad_ms"], s1["refine_window_ms"]
    # 補正で詰めた実境界へ付ける納品用の無音マージン [サンプル数]
    out_pad = round(44100 * float(s1.get("output_pad_ms") or 0) / 1000)
    cache_min = s1["cache_min_sec"]
    chunk_s = (s1.get("decode_chunk_min") or 0) * 60.0
    # チャンク境界で VAD のマージ/分割が全体デコードとズレないよう、前後に「最長発話 + 無音/音声判定 + 補正窓」
    # ぶんの文脈を確保する 。保持メモリは (chunk + 2*pad) で有界。
    ctx = (
        s1["max_speech_duration_s"]
        + (s1["min_silence_duration_ms"] + s1["min_speech_duration_ms"]) / 1000
        + win_ms / 1000
        + 0.1
    )
    pad_before = pad_after = ctx
    min_samples = int(44100 * cache_min)
    records: list[Record] = []

    for src in tqdm(sources, desc="stage1 segment", unit="src"):
        idx = 0  # utt_id 連番 (チャンクをまたいで単調。スキップ記録も同じ連番で採番)
        try:  # 1 ソースの異常でラン全体を止めない
            total = probe_duration(src.path) if chunk_s else 0.0
            chunked = bool(chunk_s) and total > chunk_s + pad_before + pad_after
            windows = (
                list(_chunk_windows(total, chunk_s, pad_before, pad_after))
                if chunked
                else [(0.0, total or float("inf"), 0.0, total or float("inf"))]
            )
            for win_lo, win_hi, owned_lo, owned_hi in windows:
                x441 = (
                    decode_range(src.path, 44100, win_lo, win_hi - win_lo)
                    if chunked
                    else decode(src.path, 44100)
                )  # デコードは 1 回だけ (silero は 8k/16k しか扱えない)
                x16 = resample_to(x441, 44100, 16000)
                ts = get_speech_timestamps(
                    torch.from_numpy(x16),
                    model,
                    sampling_rate=16000,
                    threshold=s1["vad_threshold"],
                    min_speech_duration_ms=s1["min_speech_duration_ms"],
                    min_silence_duration_ms=s1["min_silence_duration_ms"],
                    max_speech_duration_s=s1[
                        "max_speech_duration_s"
                    ],  # 長い塊は分割する (破棄しない)
                    speech_pad_ms=pad_ms,
                    return_seconds=False,
                )
                win441 = int(44100 * win_ms / 1000)
                base16 = round(win_lo * 16000)
                own_hi16 = round(owned_hi * 16000) if owned_hi != float("inf") else None
                own_lo16 = round(owned_lo * 16000)
                for t in ts:  # 16k サンプル → 44.1k サンプルへ換算 (チャンク相対)
                    abs_start16 = base16 + int(t["start"])
                    if own_hi16 is not None and not (own_lo16 <= abs_start16 < own_hi16):
                        continue  # 開始時刻が他チャンク所有の発話はここで数えない (重複防止)
                    s441, e441 = int(t["start"] * 44100 / 16000), int(t["end"] * 44100 / 16000)
                    rs = _refine_boundary(x441, 44100, s441, "start", win441)
                    re_ = _refine_boundary(x441, 44100, e441, "end", win441)
                    if re_ - rs >= min_samples:  # 補正で極端に縮んだら元境界へ退避
                        s441, e441 = rs, re_
                    if (
                        out_pad
                    ):  # 補正で詰めた実境界へ納品用の無音マージンを付ける (ソース範囲でクランプ)
                        s441 = max(0, s441 - out_pad)
                        e441 = min(len(x441), e441 + out_pad)
                    abs_start = win_lo + s441 / 44100
                    abs_end = win_lo + e441 / 44100
                    if (
                        e441 - s441 < min_samples
                    ):  # 物理下限未満はキャッシュせずスキップ記録だけ残す
                        rec = Record(
                            utt_id=f"{src.src_hash}-{idx:03d}",
                            src=src.path.name,
                            start=abs_start,
                            end=abs_end,
                        )
                        rec.scores["seg_dur"] = round((e441 - s441) / 44100, 3)
                        rec.scores["stage1_skipped"] = "too_short"
                        records.append(rec)
                        idx += 1
                        continue
                    rec = Record(
                        utt_id=f"{src.src_hash}-{idx:03d}",
                        src=src.path.name,
                        start=abs_start,
                        end=abs_end,
                    )
                    seg441 = x441[s441:e441]  # 出力も ASR 入力も同じ補正後境界 (16k はここから作る)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(cache_dir / f"{rec.utt_id}.flac", seg441, 44100, format="FLAC")
                    rec.scores["seg_dur"] = round((e441 - s441) / 44100, 3)
                    records.append(rec)
                    idx += 1
        except Exception as e:  # noqa: BLE001
            import typer

            typer.echo(f"  [stage1] {src.path.name}: 処理失敗でスキップ: {e!r}")
            rec = Record(utt_id=f"{src.src_hash}-err", src=src.path.name, start=0.0, end=0.0)
            rec.scores["stage1_skipped"] = f"source_error: {e!r}"[:200]
            records.append(rec)
    return records
