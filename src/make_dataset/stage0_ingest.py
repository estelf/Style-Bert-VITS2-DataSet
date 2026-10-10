"""Stage 0: 投入・前処理. バイトハッシュ (ストリーミング計算) で同一ファイル再投入を即除外する.

F1/F2: ハッシュはチャンク読み (数 GB の動画で read_bytes() 全展開しない)。
デコードは Stage 1 がソース単位で行い、全ソースの波形を同時保持しない。セグメント音声は
中間キャッシュ (Data/<モデル>/.cache/) に 44.1k flac で書き、以降の Stage はそこから読む
(E1 の rescore と A3 の納品 raw/ 分離のため、パイプライン中にフル尺波形を常時保持しない)。
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .manifest import Record


@dataclass
class Source:
    src_hash: str  # sha256[:16] が utt_id の接頭辞になる
    path: Path  # デコード済み波形は保持しない (F2)。Stage 1 が単位デコードする


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """F1: ストリーミング sha256[:16]."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def decode(path: Path, sr: int) -> np.ndarray:
    """ffmpeg で 1 回デコード (F6: 44.1k と 16k を 2 回起動しない)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """F6/C10: 44.1k 音声から 16k を作って ASR/埋め込みにも渡す (出力音と ASR 入力のズレ防止)."""
    from math import gcd

    if sr_in == sr_out:
        return x
    import scipy.signal

    g = gcd(sr_in, sr_out)
    return scipy.signal.resample_poly(x, sr_out // g, sr_in // g).astype("float32", copy=False)


def load_cached(record: Record, cache_dir: Path, sr: int = 44100) -> np.ndarray:
    """中間キャッシュ (.cache/<utt_id>.flac) からセグメント音声を取り出し、必要ならリサンプル."""
    import soundfile as sf

    x, file_sr = sf.read(cache_dir / f"{record.utt_id}.flac", dtype="float32")
    return resample_to(x, file_sr, sr)


def ingest(paths: list[Path]) -> tuple[list[Source], list[tuple[Path, str]]]:
    """入力ファイル群のハッシュ計算。バイトハッシュ一致 (同一ファイル再投入) は即除外して返す."""
    sources: list[Source] = []
    skipped: list[tuple[Path, str]] = []
    seen: dict[str, Path] = {}
    for path in [Path(p) for p in paths]:
        digest = file_hash(path)
        if digest in seen:
            skipped.append((path, f"byte-hash duplicate of {seen[digest].name}"))
            continue
        seen[digest] = path
        sources.append(Source(src_hash=digest, path=path))
    return sources, skipped
