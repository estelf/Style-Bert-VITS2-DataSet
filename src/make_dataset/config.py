"""filters.toml 読み込み. 閾値とパイプライン挙動設定を一元管理する (設計原則: フィルタロジックと閾値設定の分離)."""

from __future__ import annotations

import tomllib
from pathlib import Path

DEFAULTS = {
    "output": {"sample_rate": 44100, "channels": 1, "target_lufs": -23.0},
    "stage1": {
        "speech_pad_ms": 70,      # VAD パディング
        "refine_window_ms": 70,   # 強制アライメント代替の境界補正窓 (語尾切れ/先端ノイズ対策)
        "min_seg_sec": 0.6,
        "max_seg_sec": 30.0,
    },
    "asr": {
        "backend": "faster-whisper",  # または "qwen3-asr" (model には HF リポジトリID を入れる)
        "model": "small",
        "compute_type": "int8",
        "device": "cpu",
        "language": "ja",
        "beam_size": 1,
    },
    "thresholds": {
        # すべて「記録済みスコアに対する閾値」。欠損キーなら該当チェックはスキップ。
        # 既定値は test/asset/haruka_ran_nobgm の実測分布で校正済み (p5 をやや下回るあたり)。
        "min_snr_db": 5.0,
        "min_asr_conf": 0.30,
        "min_align_conf": 0.30,
        "min_mora_rate": 2.0,
        "max_mora_rate": 14.0,
        "min_hf_energy_ratio": 0.0005,  # 7kHz 以上エネルギー比。16k/24k アップコン見せかけ品質の検出 (実測 p5≈0.001)
        "min_rolloff_hz": None,         # ロールオフ単体は音量の低い文で誤爆するためゲート既定値なし
        "max_reverb_score": 0.5,    # ドラフトでは簡易プロキシ。無効化したいならキーを削除
        "min_quality_pesq": 1.0,    # SQUIM PESQ (1-4.5)。ドラフトでは緩めに通し、レポートで分布確認
        "min_chars": 4,
        "max_filler_ratio": 0.8,
        "max_jitter_local": 0.5,    # F0 摂動プロキシ。praat 本来の jitter へ差し替え時に 0.02 程度へ締める
        "max_shimmer_local": None,  # プロキシ未整備のためデフォルト無効
    },
    "dedup": {
        "text_jaccard": 0.92,       # テキストファジィ (shingle Jaccard)
        "phoneme_jaccard": 0.90,    # 音素三量文字列 Jaccard
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class Config:
    """filters.toml + デフォルト の差分設定。点記号でスコア閾値を参照できる。"""

    def __init__(self, data: dict):
        self.data = _merge(DEFAULTS, data)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        data = {}
        if path is not None and Path(path).exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
        return cls(data)

    def get(self, dotted: str):
        cur = self.data
        for key in dotted.split("."):
            cur = cur[key]
        return cur
