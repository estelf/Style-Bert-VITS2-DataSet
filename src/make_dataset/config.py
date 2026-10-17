"""filters.toml 読み込み. 閾値とパイプライン挙動設定を一元管理する (設計原則: フィルタロジックと閾値設定の分離).

対応の設計: 「キーを消せば無効化される」を本当に成立させるため、閾値 (thresholds) に
コード側デフォルトを持たせない。全閾値は filters.toml に明示され、キーが無ければ該当チェックは走らない。
TOML には None を書けないので、「キー自体を消す = 無効」が唯一の無効化手段であることが保証される。
パイプライン挙動 (stage1/asr/dedup の一部) は DEFAULTS に持ってよい (閾値ではなく動作パラメータ)。
"""

from __future__ import annotations

import copy
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
        # silero-vad の主要パラメータを露出する。「1行=1文」契約を左右するのは speech_pad_ms よりこちら側。
        "vad_threshold": 0.5,
        "min_silence_duration_ms": 200,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 15.0,  # 長い塊を VAD 側で分割し、seg_dur で捨てない (DESIGN「2文混入の自動分割」)
        "speech_pad_ms": 0,  # 既定は無音パディングなし (VAD 側の切り出しを前後へ広げるだけ。output_pad_ms とは別物)
        "refine_window_ms": 70,  # 境界補正窓 [ms]。語尾切れ/先端ノイズ対策で side を見て寄せる
        # 納品音声の前後に付ける無音マージン [ms]。境界補正が話声の実境界まで詰めるため、
        # speech_pad_ms は補正窓の内側でトリムされて出力に残らない。語頭/語尾のメロスぺクトル窓切れ・
        # クリック対策に「補正後」の境界へマージンを足す専用ノブが必要。0 で無効 (従来動作)。
        #   分割地点の実無音 (>= min_silence_duration_ms) より十分に小さく置くこと (隣の発話の巻き込み防止)。
        "output_pad_ms": 0.0,
        # キャッシュ保持の「物理下限」。これより短い VAD 出力は音声として成立しないので中間キャッシュせず
        #       manifest 上のスキップ記録だけ残す。採用/除外の発話長ゲートは [thresholds].min_seg_sec が担う
        #       (下限を閾値より低く置くことで、rescore で min_seg_sec を下げる再採用が可能になる)。
        "cache_min_sec": 0.2,
        # 1 ソースをこの長さ (分) のチャンクに分割してデコード→VAD する (重なり込みで境界の文を失わない)。
        #       長尺動画 1 本をフル尺デコードして常時保持し続ける OOM を防ぐ。None で無効 (旧来=一括デコード)。
        "decode_chunk_min": 10.0,
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
        "batch_size": 8,  # デコード窓のバッチ (faster-whisper の並列デコード)
    },
    # thresholds に hidden default は置かない。filters.toml が唯一の閾値ソース。
    "thresholds": {},
    "dedup": {
        "text_jaccard": 0.92,  # テキストファジィ (shingle Jaccard)
        "phoneme_jaccard": 0.90,  # 音素三量文字列 Jaccard
        "audio_jaccard": 0.5,  # 音声内容指紋 (fp_sig MinHash 一致率) によるズレ切り出し重複検出
    },
    # 多様性選別 (Stage 5b): 重複排除の後に残り、貪欲選択して「既採用と類似しすぎる発話」を落とす。
    #   採用集合の総対類似度を sim_threshold 未満に抑えることで、データセットの多様性を上げる。
    #   metric: phoneme(音素三量文字列 Jaccard, 既定) / text(shingle) / audio(内容指紋 fp_sig 一致率) — いずれも「内容」の近さ。話者/音響の emb は単一話者で別内容まで間引くため不採用
    #   protect_coverage: true の間、ある音素の唯一の供給源は多様性名目で落とさない (カバレッジ保護)。
    #   以下は追加ノブ (既定 0 = 無効 = 従来の「品質順+類似間引き」と同動作):
    #     gain_lambda      : 選択スコアに足す「新規音素 three-gram 利得」の重み (多様性の能動的最大化)
    #     max_bucket_share : 韻律バケット (F0 三分位 × 話速三分位) ありの最大採用シェア (偏りの整流化)
    #     max_source_share : 投入ソース 1 本の最大採用シェア (長尺 1 本のスタイル引きずりの防止)
    #     target_count     : 目標本数に達したら選択を打ち切る (K 本の多様性最大化選択。λ>0 と併用)
    "diversity": {
        "enabled": True,
        "metric": "phoneme",
        "sim_threshold": 0.6,
        "protect_coverage": True,
        "gain_lambda": 0.0,
        "max_bucket_share": 0.0,
        "max_source_share": 0.0,
        "target_count": 0,
    },
    # 第 2 ASR クロス検証。メインと別系統を走らせ、2 系統間 CER を**読みレベル**で x_cer に記録する
    #   (表層文字値は x_cer_surface。読み化の根拠は stage3_asr.reading_key の実測コメント参照)。
    #   既定オフ (第 2 モデルのダウンロード/実行コスト)。on のとき「明らかなゴミ」発話だけスキップ
    #   (asr2_skipped=obvious_garbage。カットオフは固定値でユーザー閾値から独立 — rescore で緩めても
    #    スコア欠損の罠を引かない)。ゲート max_x_cer は [thresholds] 側で有効化。
    "asr2": {
        "enabled": False,
        "backend": "faster-whisper",  # メイン (既定 qwen3-asr) とは違う系統にすること
        "model": "small",
        "compute_type": "int8",
        "device": "auto",
        "language": "ja",
        "beam_size": 1,
        "batch_size": 8,
    },
    # SQUIM_SUBJECTIVE を mos として記録 (SNR 最良発話をアンカーにした相対 MOS)。初回 DL 約 360MB。
    "stage4": {"subjective": False},
    # AudioSet タグ付け (AST)。音楽/笑い/咳などを音声から直接検出して tag_hit に記録。初回 DL 約 1.3GB。
    #   reject は AudioSet クラス名 (filters.toml で増減可)。min_prob 以上でヒット。
    "tags": {
        "enabled": False,
        "model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "min_prob": 0.25,
        "reject": [
            "Music",
            "Musical instrument",
            "Singing",
            "Laughter",
            "Applause",
            "Cheering",
            "Cough",
            "Sneeze",
            "Crying, sobbing",
            "Baby cry, infant cry",
            "Alarm",
            "Siren",
            "Telephone",
            "Beep, bleep",
        ],
    },
    # pyannote segmentation-3.0 による 2 人同時発話検出 (任意依存 + HF_TOKEN が必要)。
    "overlap": {
        "enabled": False,
        "model": "pyannote/segmentation-3.0",
        "token": None,  # 省略時は環境変数 HF_TOKEN
        "min_overlap_sec": 0.2,
    },
    # CTC 強制アライメント (日本語 CTC ASR)。ctc_align_score / unaligned_edge_ratio を記録。
    "alignment": {
        "enabled": False,
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    },
    # MLM 擬似パープレキシティ。lm_ppl を記録 (max_lm_ppl / review_lm_ppl で判定)。
    #   ku-nlp DeBERTa-v2 char-wwm (SentencePiece 文字トークナイザ)。bert-japanese 系は char 版でも
    #   MeCab (fugashi) を要求するため、追加パッケージなしで動くこちらを既定にする。
    "lm": {
        "enabled": False,
        "model": "ku-nlp/deberta-v2-large-japanese-char-wwm",
    },
    # 記号の読み上げ展開 (% → パーセント 等) と残留記号カウント。既定 on。
    "text": {"symbol_readings": True},
    # DeepFilterNet による除去修復 (任意依存)。ノイズ系ゲート単独落ちを修復し、
    #   再検証を通過しても pending_review 行き (自動採用しない)。元音声は .cache_orig/ に保持。
    "repair": {
        "enabled": False,
        "backend": "deepfilternet",
        "targets": ["min_snr_db", "min_quality_pesq", "min_stoi", "min_si_sdr"],
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
        # DEFAULTS を深いコピーしてからマージ: さもないと _merge が入れ子 dict を参照共有し、
        # cfg.data の一部を書き換えたとき DEFAULTS (＝後続の全 Config) が汚染される。
        self.data: dict[str, Any] = _merge(copy.deepcopy(DEFAULTS), data)

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
