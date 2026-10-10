"""filters.toml 読み込み. 閾値とパイプライン挙動設定を一元管理する (設計原則: フィルタロジックと閾値設定の分離).

E2 対応の設計: 「キーを消せば無効化される」を本当に成立させるため、閾値 (thresholds) に
コード側デフォルトを持たせない。全閾値は filters.toml に明示され、キーが無ければ該当チェックは走らない。
TOML には None を書けないので、「キー自体を消す = 無効」が唯一の無効化手段であることが保証される。
パイプライン挙動 (stage1/asr/dedup の一部) は DEFAULTS に持ってよい (閾値ではなく動作パラメータ)。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

DEFAULTS = {
    "paths": {
        # `make_dataset run <話者名>` が音声を取りに行くルート。voice/<話者名>/ にフラット配置する
        "voice": "voice",
    },
    "output": {"sample_rate": 44100, "channels": 1, "target_lufs": -23.0},
    "stage1": {
        # D4: silero-vad の主要パラメータを露出する。「1行=1文」契約を左右するのは speech_pad_ms よりこちら側。
        "vad_threshold": 0.5,
        "min_silence_duration_ms": 200,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 15.0,  # D3: 長い塊を VAD 側で分割し、seg_dur で捨てない (DESIGN「2文混入の自動分割」)
        "speech_pad_ms": 0,  # C11: 既定は無音パディングなし (出力は音声境界そのまま。ASR に文脈が欲しい時だけ上げる)
        "refine_window_ms": 70,  # 強制アライメント代替の境界補正窓 (語尾切れ/先端ノイズ対策、C8 で side 考慮)
        "min_seg_sec": 0.5,  # これより短い VAD 出力はスキップ (閾値ゲート [thresholds].min_seg_sec と揃える)
    },
    "runtime": {
        "device": "auto",  # "auto" / "cpu" / "cuda"。Qwen3-ASR・SQUIM 等の torch 推論が従う (docs/GPU.md)
    },
    "asr": {
        "backend": "faster-whisper",  # filters.toml の既定は "qwen3-asr"。こちらはコード側の安全側フォールバック (CPU 軽量)
        "model": "small",
        "compute_type": "int8",  # faster-whisper のみ。cuda では float16 へ自動切替 (int8 は CPU 量子化)
        "device": "auto",  # "auto" なら runtime.device に従う
        "language": "ja",
        "beam_size": 1,
        "batch_size": 8,  # F5: デコード窓のバッチ (faster-whisper の並列デコード)
    },
    # E2: thresholds に hidden default は置かない。filters.toml が唯一の閾値ソース。
    "thresholds": {},
    "dedup": {
        "text_jaccard": 0.92,  # テキストファジィ (shingle Jaccard)
        "phoneme_jaccard": 0.90,  # 音素三量文字列 Jaccard
        "audio_jaccard": 0.5,  # D8: 音声内容指紋 (fp_sig MinHash 一致率) によるズレ切り出し重複検出
    },
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class Config:
    """filters.toml + デフォルト の差分設定。点記号で挙動パラメータを参照できる。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data: dict[str, Any] = _merge(DEFAULTS, data)

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        data: dict[str, Any] = {}
        if path is not None and Path(path).exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
        return cls(data)

    def get(self, dotted: str) -> Any:
        """点記号参照。TOML 由来の値は型が一律に決まらないので Any (閾値は float、パスは str 等)."""
        cur = self.data
        for key in dotted.split("."):
            cur = cur[key]
        return cur

    @property
    def thresholds(self) -> dict[str, Any]:
        """閾値テーブル (キーが存在しない = そのチェックは無効)。"""
        return self.data["thresholds"]


def resolve_device(cfg: Config) -> str:
    """runtime.device ("auto"/"cpu"/"cuda") を実デバイス名へ解決する。auto は CUDA があれば cuda。"""
    dev = cfg.get("runtime.device")
    if dev != "auto":
        return dev
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
