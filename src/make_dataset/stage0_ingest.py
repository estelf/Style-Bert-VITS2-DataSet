"""Stage 0: 投入・前処理. ffmpeg で 44.1kHz モノラル float 化、ハッシュID付与、バイトハッシュ一致即除外."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 44100


@dataclass
class Source:
    src_hash: str          # sha256[:16] が utt_id の接頭辞になる
    path: Path
    audio441: "object"     # np.ndarray float32 mono 44.1k
    audio16k: "object"     # VAD/ASR/埋め込み用 16k モノラル


def _decode(path: Path, sr: int) -> "object":
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def ingest(paths: list[Path]) -> tuple[list[Source], list[tuple[Path, str]]]:
    """入力ファイル群をデコード。バイトハッシュ一致 (同一ファイル再投入) は即除外して返す."""
    sources: list[Source] = []
    skipped: list[tuple[Path, str]] = []
    seen: dict[str, Path] = {}
    for path in [Path(p) for p in paths]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if digest in seen:
            skipped.append((path, f"byte-hash duplicate of {seen[digest].name}"))
            continue
        seen[digest] = path
        import soundfile  # noqa: F401  (flac/mp3 デコードは ffmpeg に委譲するが sr=44100 保証のため明示)

        sources.append(
            Source(src_hash=digest, path=path, audio441=_decode(path, SAMPLE_RATE), audio16k=_decode(path, 16000))
        )
    return sources, skipped
