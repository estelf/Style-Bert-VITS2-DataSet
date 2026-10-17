"""Stage 6: 出力. 閾値で adopted/excluded を確定し、esd.list + raw.zip + manifest.jsonl + report.html を生成する.

- 失格理由は最初の 1 個ではなく全部を dropped_by に残す (感度分析と説明可能性の基盤)。
- g2p / unknown_tokens の扱い (除外 or ユーザー辞書候補) を filters.toml の on_g2p_fail /
  on_unknown_tokens で決める。ハードコードゲートをなくす。
- 非有限値や欠損スコアは「検証不能=不合格」。x * -inf で nan を納品しない。
- 納品用 raw/ は毎回作り直す。中間音声は .cache/ (rescore 用) と分離。
- 閾値感度分析表 (この閾値を緩めれば +N 発話) をレポートに出す。
- レポート (report.html) は「何を見る数字か」がその場で判断できるよう、失格理由キーの凡例と見方を
  同じページに置く。判定は機械が行うので、人手レビュー用の別ページは作らない
  (人が動かすのは filters.toml の閾値と text_overrides.jsonl だけ)。
"""

from __future__ import annotations

import contextlib
import html
import io
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config
from .manifest import Record

# 納品契約 (学習側 preprocess と同じ): 44.1kHz・モノラル。[output] がこれ以外なら書き出し前に弾く。
CONTRACT_SR, CONTRACT_CH = 44100, 1

# (閾値キー, スコアキー, 方式)。方式: min=スコアが閾値未満で不合格 / max=超過で不合格 /
# max_excl=閾値以上で不合格 (「12秒未満」のような排他的上限) / flag=True で不合格。
# 閾値キーが filters.toml に無ければチェック自体が走る。スコアが None は「検証不能=不合格」。
CHECKS: list[tuple[str, str, str]] = [
    ("min_seg_sec", "seg_dur", "min"),  # 発話長ゲート (既定 0.5 秒以上)
    ("max_seg_sec", "seg_dur", "max_excl"),  # 発話長ゲート (既定 12 秒未満。12.0 ちょうどは除外)
    ("min_snr_db", "snr", "min"),
    ("min_hf_energy_ratio", "hf_energy_ratio", "min"),
    ("min_rolloff_hz", "rolloff_hz", "min"),
    ("max_reverb_score", "reverb_score", "max"),
    ("max_jitter_local", "jitter_local", "max"),
    ("max_shimmer_local", "shimmer_local", "max"),
    ("min_speaker_sim", "speaker_sim", "min"),
    ("min_quality_pesq", "pesq", "min"),
    ("min_asr_conf", "asr_conf", "min"),
    ("min_align_conf", "align_conf", "min"),
    ("min_chars", "char_count", "min"),
    ("max_filler_ratio", "filler_ratio", "max"),
    ("max_mora_rate_dev", "mora_rate_dev", "max"),
    ("min_kana_ratio", "kana_ratio", "min"),
    ("max_clip_ratio", "clip_ratio", "max"),
    ("max_dc_offset", "dc_offset_abs", "max"),
    ("max_no_speech_prob", "no_speech_prob", "max"),
    ("max_compression_ratio", "compression_ratio", "max"),
    ("reject_repetition", "repeat_block", "flag"),
    ("reject_alpha_only", "alpha_only", "flag"),  # filler_ratio とは別理由で記録する
    ("reject_music", "music_flag", "flag"),
    # --- 計算済みなのに未配線だったスコアをゲート化 (B) ---
    (
        "min_f0_hz",
        "f0_median_hz",
        "min",
    ),  # 異常に低いピッチ (filters.toml に記載があったのに未実装だった)
    ("max_f0_hz", "f0_median_hz", "max"),  # 異常に高いピッチ (叫び・子供声混入)
    (
        "max_digital_silence",
        "digital_silence_ratio",
        "max",
    ),  # B: 無音スライス混入 (追加コストゼロの高価値ゲート)
    ("min_voiced_ratio", "voiced_ratio", "min"),  # B: 有声率极低 (ノイズ/換気/咳のみの発話)
    ("min_stoi", "stoi", "min"),  # B: 明瞭度 (pesq とは別軸の SQUIM 指標)
    ("min_si_sdr", "si_sdr", "min"),  # B: 分離度
    (
        "min_input_lufs",
        "lufs",
        "min",
    ),  # E: 極端に小さい入力 (LUFS 正規化でノイズ床も持ち上げるのを防ぐ)
    (
        "reject_overlap",
        "overlap_flag",
        "flag",
    ),  # 2人同時発話 (pyannote 接続時に有効化。未接続は None で素通り)
    # --- P0-P2 で追加した検証軸 (プロバイダは [asr2]/[tags]/[stage4]/[alignment]/[lm]/[overlap]/[text]。既定は全無効) ---
    ("max_x_cer", "x_cer", "max"),  # 2 系統 ASR 間 CER (読みレベル。不一致 = どちらかが誤認識)
    (
        "min_token_prob",
        "token_prob_min",
        "min",
    ),  # トークン確率の最小値 (平均が高くても 1 語だけ怪しい)
    ("max_token_nll", "token_nll", "max"),  # 平均負対数尤度 (分布全体の信頼度)
    ("reject_audio_tags", "tag_hit", "flag"),  # AudioSet タグ (音楽/笑い/咳を音声から直接検出)
    ("min_mos", "mos", "min"),  # 非侵入 MOS (SQUIM_SUBJECTIVE。pesq とは別軸の自然さ)
    ("min_ctc_align", "ctc_align_score", "min"),  # CTC 強制アライメントの経路平均後部確率
    (
        "max_unaligned_edge",
        "unaligned_edge_ratio",
        "max",
    ),  # 先頭/末尾の「テキストに無い発話」比
    ("max_lm_ppl", "lm_ppl", "max"),  # MLM 擬似 PPL (文脈的に不自然な文字起こし)
    ("max_residual_symbols", "residual_symbols", "max"),  # 読み上げ展開後も残る記号の数
    (
        "max_text_tail_open",
        "text_tail_open",
        "max",
    ),  # 文末の開き (1=言い淀み止めを許すなら 1、語・発話が途中で切れている 2 は常に落とす)
]


def _check_failed(mode: str, value, thresh) -> bool:
    if mode == "flag":
        return bool(value)
    if value is None:
        return True  # 閾値が有効なのにスコアが無い/非有限 → 検証不能なので不合格 (異常値ほど通るを禁止)
    if mode == "max_excl":
        return value >= thresh
    return value < thresh if mode == "min" else value > thresh


_ALLOWED_ACTIONS = {
    "on_g2p_fail": {"exclude", "dict_data"},
    # "adopt" は削除: unknown = 学習側で読めない語であり、採用すると学習側 preprocess
    # (yomi_error=raise 既定) で必ず落ちる行を納品する。読めない語を含むデータは不採用とする
    "on_unknown_tokens": {"exclude", "dict_data"},
}


def _validate_actions(t: dict) -> None:
    """on_g2p_fail / on_unknown_tokens の値を検証する (誤字が黙って「除外→見送り降格」するのを防ぐ)."""
    for key, allowed in _ALLOWED_ACTIONS.items():
        v = t.get(key)
        if v is not None and v not in allowed:
            raise ValueError(f"[thresholds] {key} = {v!r} が不正 (使用可: {sorted(allowed)})")


def evaluate_checks(records: list[Record], cfg: Config) -> None:
    """全スコアはここまで「記録」止まり。ここで初めて閾値で各チェックの合否を判定する (全理由)."""
    t = cfg.thresholds
    _validate_actions(t)
    for r in records:
        s = r.scores
        r.review_reason = None
        fails: list[str] = []
        if s.get("errors"):
            fails.append("pipeline_error")
        if s.get(
            "stage1_skipped"
        ):  # 短すぎ/ソース失敗は閾値と無関係に常に不合格 (音声キャッシュが無い)
            fails.append(str(s["stage1_skipped"]).split(":")[0])
        for tkey, skey, mode in CHECKS:  # 発話長 (seg_dur) も含め全て閾値テーブル駆動
            if tkey not in t or t[tkey] is None:
                continue  # キーが無い = チェック自体が走らない
            if mode == "flag" and not t[tkey]:
                continue
            if _check_failed(mode, s.get(skey), t[tkey]) and tkey not in fails:
                fails.append(
                    tkey
                )  # 閾値キー単位で記録 (min/max が同じスコアを見ても感度表が混線しない)
        # 話速レンジ (モーラ数が取れない発話はスキップ)。キーがあっても None なら片側開放
        lo_t, hi_t = t.get("min_mora_rate"), t.get("max_mora_rate")
        if (lo_t is not None or hi_t is not None) and s.get("mora_count", 0):
            rate = s.get("mora_rate") or 0
            lo = lo_t if lo_t is not None else -1e9
            hi = hi_t if hi_t is not None else 1e9
            if not (lo <= rate <= hi):
                fails.append("mora_rate")
        # 閾値と独立の不変条件 — 空セリフ / モーラ0 は学習不能なので常に不合格 (min_chars を消しても守る)
        if not s.get("stage1_skipped") and (
            not (r.text or "").strip() or (s.get("mora_count") or 0) <= 0
        ):
            fails.append("empty_text")
        # G2P 失敗・unknown トークンの扱いを filters.toml で決める (値は検証済み)
        reviews: list[str] = []
        act = t.get("on_g2p_fail")
        if act is not None and not s.get("g2p_ok", True):
            if act == "exclude":
                fails.append("g2p")
            else:
                reviews.append("g2p:dict_data")
        act = t.get("on_unknown_tokens")
        if act is not None and s.get("unknown_tokens", 0) > 0:
            if act == "exclude":
                fails.append("unknown_tokens")
            else:
                reviews.append("dict_data")  # DESIGN「ユーザー辞書候補への振り分け」の受け皿
        # LM 不自然度の見送り閾値 (max_lm_ppl=除外より一段柔らかい受け皿。超過は pending_review)
        lm_rev = t.get("review_lm_ppl")
        if lm_rev is not None and s.get("lm_ppl") is not None and s["lm_ppl"] > lm_rev:
            reviews.append("lm_ppl")
        # 2 系統不一致の見送り閾値 (max_x_cer=除外より一段柔らかい受け皿)。
        # 食い違っても「主 ASR が正しく whisper 側が間違えただけ」が実測 4 分 (CTC 勝敗判定) なので、
        # 機械的に落とさず pending_review に出し、text_overrides.jsonl で本文を確定できる枠にする
        x_rev = t.get("review_x_cer")
        if x_rev is not None and s.get("x_cer") is not None and s["x_cer"] > x_rev:
            reviews.append("x_cer")
        r.review_reason = ",".join(dict.fromkeys(reviews)) or None
        s["failed_checks"] = fails


def finalize_statuses(records: list[Record]) -> None:
    """dedup / 多様性選別 / チェック結果を合わせて status / dropped_by (全理由) を確定する."""
    for r in records:
        reasons = list(r.scores.get("failed_checks") or [])
        if r.dup_reason:
            reasons.append(f"dedup:{r.dup_reason}")
        if r.diversity_reason:
            reasons.append(f"diversity:{r.diversity_reason}")
        if reasons:
            r.status, r.dropped_by = "excluded", reasons
        elif r.review_reason:
            r.status = "pending_review"
        else:
            r.status, r.dropped_by = "adopted", None
        # 修復音声は再検証を通過しても自動採用しない (除去アーティファクトを機械の自己申告で通さない)
        if r.scores.get("repaired") and r.status == "adopted":
            r.status = "pending_review"
            r.review_reason = "repaired"
        # text_overrides.jsonl の人手判定は自動判定より常に優先 (判定を確定する最終受け皿)
        ov = r.scores.get("status_override")
        if ov == "adopted":
            r.status, r.dropped_by = "adopted", None
            r.scores["manual_override"] = True
        elif ov == "excluded":
            r.status, r.dropped_by = "excluded", ["manual"]
            r.scores["manual_override"] = True
        elif ov == "pending_review":
            r.status = "pending_review"
            r.scores["manual_override"] = True


def write_outputs(
    adopted: list[Record], cache_dir: Path, outdir: Path, speaker: str, cfg: Config
) -> Path:
    """esd.list + raw.zip (44.1k/mono・LUFS正規化済み flac) を書き、契約検証まで通す."""
    import pyloudnorm as pyln

    from .stage0_ingest import load_cached

    sr = cfg.get("output.sample_rate")
    ch = cfg.get("output.channels")
    if (sr, ch) != (
        CONTRACT_SR,
        CONTRACT_CH,
    ):  # 契約は 44.1k/mono 固定。全作業後に遅れて落ちず先に告げる
        raise ValueError(
            f"出力契約は {CONTRACT_SR}Hz/mono 固定 (現在 [output] sr={sr} ch={ch})。"
            "学習側契約を変えるなら filters.toml の [output] を合わせてください"
        )
    meter = pyln.Meter(sr)
    target = cfg.get("output.target_lufs")
    raw = outdir / "raw"
    if raw.exists():  # 前回の除外済みファイルが混ざらないよう納品用 raw/ は毎回作り直す
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    esd_lines = []
    for r in adopted:
        x = np.asarray(load_cached(r, cache_dir, sr), dtype=np.float32)
        if not np.all(np.isfinite(x)):  # nan/inf を納品しない (errors 側でも除外されるが二重防御)
            x = np.nan_to_num(x)
        if len(x) >= meter.block_size * sr:  # 1秒ゲートではなくメータのブロック長基準
            lufs = meter.integrated_loudness(x)
            if np.isfinite(lufs):  # -inf を掛けて nan を撒かない
                with contextlib.suppress(
                    ValueError
                ):  # 正規化不可のクリップはそのまま納品 (直後の peak で守る)
                    x = pyln.normalize.loudness(x, lufs, target)  # (data, 実測LUFS, 目標LUFS)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if np.isfinite(peak) and peak > 10 ** (-1 / 20):  # nan > 0.891 == False で素通りしない
            x = x * (10 ** (-1 / 20) / peak)
            r.scores["limited"] = True  # リミッタが LUFS ターゲットを崩したことを黙らせない
        if len(x) >= meter.block_size * sr:
            lo = meter.integrated_loudness(x)
            r.scores["lufs_out"] = round(float(lo), 1) if np.isfinite(lo) else None
        fname = f"{r.utt_id}.flac"
        sf.write(raw / fname, x.astype(np.float32), sr, format="FLAC")
        esd_lines.append(f"{fname}|{speaker}|{r.lang}|{r.text}")

    (outdir / "esd.list").write_text(
        "\n".join(esd_lines) + ("\n" if esd_lines else ""),
        encoding="utf-8",  # 0 件なら空ファイル
    )
    with zipfile.ZipFile(outdir / "raw.zip", "w", zipfile.ZIP_STORED) as z:
        for p in sorted(raw.rglob("*.flac")):
            z.write(p, p.relative_to(raw))
    _contract_check(outdir / "raw.zip", sr, ch)
    return outdir


def _contract_check(zip_path: Path, sr: int, channels: int) -> None:
    """納品前に学習側 preprocess と同じ検証 (指定 sr / モノラル等) を全ファイルへ通す."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            info = sf.info(io.BytesIO(z.read(name)))
            if info.samplerate != sr:
                raise ValueError(f"contract violation: {name} sr={info.samplerate}")
            if info.channels != channels:
                raise ValueError(f"contract violation: {name} not {channels}ch ({info.channels}ch)")


# ---------------------------------------------------------------------------
# report.html 生成 (単一ファイル / 外部 CDN 依存なし / ダークテーマ)
#
# 「数字を並べても判断材料にはならない」ので、各ゲートが何を見るのかの **凡例** と
# 表の読み方・動かし方の **見方** を同じページに置く。人手レビュー用の別ページは作らない
# (判定は機械が行い、人が動かすのは filters.toml の閾値と text_overrides.jsonl だけ、という運用)。
# ---------------------------------------------------------------------------

# 失格理由キー → (分類, 何を見るのか)。dropped_by のキーは閾値キーそのものなので、
# ここで日本語を引けないと「min_snr_db: 41」だけの意味不明な表になる。
# 分類は chip の色に対応 (CATEGORY_CLASS)。
_REASON_LEGEND: dict[str, tuple[str, str]] = {
    # --- 発話そのもの ---
    "min_seg_sec": ("発話", "発話長が下限未満 (短すぎて学習に使えない)"),
    "max_seg_sec": ("発話", "発話長が上限以上 (2 文混入・長尺の地話しの疑い)"),
    "min_input_lufs": ("発話", "入力のラウドネスが下限未満 (LUFS 正規化でノイズ床まで持ち上げる)"),
    # --- 音響品質 ---
    "min_snr_db": ("音質", "SNR が下限未満 (ノイズに埋もれている)"),
    "min_hf_energy_ratio": (
        "音質",
        "7kHz 以上のエネルギー比が下限未満 (16/24k をアップコンした見せかけ高音質)",
    ),
    "min_rolloff_hz": (
        "音質",
        "ロールオフ周波数が下限未満 (高域が切れている。音量の低い文で誤爆しやすい)",
    ),
    "max_reverb_score": ("音質", "残響スコアが上限超過 (部屋鳴り・マイク距離)"),
    "min_quality_pesq": ("音質", "PESQ (SQUIM 推定) が下限未満"),
    "min_stoi": ("音質", "STOI 明瞭度が下限未満 (何を言っているか判別しにくい)"),
    "min_si_sdr": ("音質", "SI-SDR 分離度が下限未満 (ノイズ成分の大きさ)"),
    "min_mos": ("音質", "非侵入 MOS が下限未満 (同一話者の最クリーン発話に対する相対値)"),
    # --- 声の質・ピッチ ---
    "max_jitter_local": ("声質", "声帯ゆらぎ jitter local が上限超過 (嗄れ・不安定な発声)"),
    "max_shimmer_local": ("声質", "振幅ゆらぎ shimmer local が上限超過"),
    "min_f0_hz": ("ピッチ", "F0 中央値が帯域より低い (うなり・床に伏せた声)"),
    "max_f0_hz": ("ピッチ", "F0 中央値が帯域より高い (叫び・子供声混入)"),
    # --- 話者 ---
    "min_speaker_sim": ("話者", "主話者埋め込みとの cosine が下限未満 (別人混入の疑い)"),
    "reject_overlap": ("話者", "2 人同時発話 (pyannote。未接続は None で素通り)"),
    # --- 文字起こしの信頼度 ---
    "min_asr_conf": ("文字起こし", "ASR の平均トークン確率が下限未満 (聞き取れない発話)"),
    "min_align_conf": ("文字起こし", "アライメント確信度が下限未満"),
    "min_token_prob": (
        "文字起こし",
        "トークン確率の最小値が下限未満 (平均は良くても 1 語だけ怪しい)",
    ),
    "max_token_nll": ("文字起こし", "平均負対数尤度が上限超過 (確率分布全体の信頼度が低い)"),
    "min_ctc_align": (
        "文字起こし",
        "CTC 強制アライメントの経路平均後部確率が下限未満 (音声↔テキストの実一致)",
    ),
    "max_no_speech_prob": (
        "文字起こし",
        "「無声区間」と ASR が申告した比率が上限超過 (ハルシネーション。faster-whisper 専用)",
    ),
    "max_compression_ratio": (
        "文字起こし",
        "圧縮率が上限超過 (繰り返しガチャ。faster-whisper 専用)",
    ),
    "max_x_cer": ("照合", "2 系統 ASR 間の読みレベル CER が上限超過 (どちらかが誤認識)"),
    "max_unaligned_edge": (
        "境界",
        "先頭/末尾の「テキストに無い発話」比が上限超過 (2 文混入・切り出しズレ)",
    ),
    # --- テキスト ---
    "min_chars": ("テキスト", "文字数が下限未満 (情報として短すぎる)"),
    "max_filler_ratio": ("テキスト", "フィラー (あの・えーと) 比率が上限超過"),
    "min_kana_ratio": ("テキスト", "かな比率が下限未満 (日本語らしさ。英語・記号化テキストの検出)"),
    "max_residual_symbols": ("テキスト", "記号読み上げ展開後も残る記号の数 (G2P で読めないリスク)"),
    "max_lm_ppl": ("テキスト", "MLM 擬似 PPL が上限超過 (文脈的に不自然 = 誤変換の疑い)"),
    "max_text_tail_open": (
        "テキスト",
        "文末の開きが許容度超過 (0=閉じる / 1=言い淀み止め / 2=途中で切れ)",
    ),
    "reject_repetition": ("テキスト", "同一文言 3 連呼以上のブロック"),
    "reject_alpha_only": ("テキスト", "アルファベットのみ (日本語として学習不能)"),
    "empty_text": ("テキスト", "空セリフまたはモーラ 0 (閾値と無関係に学習不能)"),
    # --- 読み (学習側 preprocess と同じ打診) ---
    "g2p": ("読み", "G2P が通らない (納品しても学習側 preprocess で落ちる)"),
    "unknown_tokens": ("読み", "学習側で読めない語を含む (pron=「、」。希少漢字・古いカタカナ等)"),
    # --- 話速 ---
    "mora_rate": ("話速", "話速 (モーラ/秒) が min_mora_rate〜max_mora_rate の範囲外"),
    "max_mora_rate_dev": (
        "話速",
        "話者中央値からの話速乖離が上限超過 (2 文混入・ASR 欠落・ガチャ)",
    ),
    # --- 音声欠陥 ---
    "max_clip_ratio": ("欠陥", "クリッピング率が上限超過 (学習後に修復不可能)"),
    "max_dc_offset": ("欠陥", "DC オフセット絶対値が上限超過"),
    "max_digital_silence": ("欠陥", "デジタル無音サンプル率が上限超過 (無音スライス混入)"),
    "min_voiced_ratio": ("欠陥", "有声フレーム率が下限未満 (ノイズ・換気・咳のみ)"),
    # --- 音響タグ ---
    "reject_music": ("音響タグ", "♪ を含む = 音楽と判定 (ASR 出力ベース)"),
    "reject_audio_tags": (
        "音響タグ",
        "AudioSet タグヒット (音楽・笑い・咳・拍手を音声から直接検出)",
    ),
    # --- 実行・前段・人手 ---
    "pipeline_error": ("実行", "パイプライン実行中にエラー (scores.errors)"),
    "too_short": ("前段", "Stage 1 の VAD 出力が短すぎて音声キャッシュを作っていない (常に不合格)"),
    "source_error": ("前段", "ソースのデコードに失敗 (常に不合格)"),
    "manual": ("手動", "text_overrides.jsonl で status=excluded を指定"),
}

# プレフィックスで寄せる理由 (dedup:text / diversity:similar:<utt> のように個体 ID を含むもの)
_REASON_PREFIXES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("dedup:", ("重複", "同一内容のグループから代表 1 本以外を間引いた (内容は代表と同じ)")),
    (
        "diversity:similar",
        ("多様性", "既採用の内容と似すぎ (音素三量 Jaccard)。末尾に代表 utt_id が残る"),
    ),
    ("diversity:cap:prosody", ("多様性", "韻律バケット (F0×話速) の採用シェア上限に達した")),
    ("diversity:cap:source", ("多様性", "投入ソース 1 本の採用シェア上限に達した")),
    ("diversity:target_size", ("多様性", "target_count に達したため以降は選択を打ち切り")),
    ("diversity:", ("多様性", "多様性選別 (Stage 5b) による間引き")),
)

# 分類 → chip のクラス (色で「どの軸のゲートが効いているか」を一覧で見せる)
_CATEGORY_CLASS = {
    "発話": "c-audio",
    "音質": "c-audio",
    "欠陥": "c-audio",
    "声質": "c-voice",
    "ピッチ": "c-voice",
    "話者": "c-spk",
    "文字起こし": "c-asr",
    "照合": "c-asr",
    "境界": "c-asr",
    "テキスト": "c-text",
    "読み": "c-text",
    "話速": "c-text",
    "音響タグ": "c-tag",
    "重複": "c-dedup",
    "多様性": "c-dedup",
    "前段": "c-sys",
    "実行": "c-sys",
    "手動": "c-sys",
    "その他": "c-sys",
}

_CSS = """
:root{
  --bg:#0e1116; --bg-soft:#151a21; --card:#171d26; --card-2:#1d242f; --line:#2a333f;
  --fg:#e6edf3; --muted:#93a1b1; --dim:#6b7787;
  --ok:#3fb950; --bad:#f85149; --warn:#d29922; --info:#58a6ff; --accent:#a371f7;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans JP","Hiragino Sans",Meiryo,sans-serif}
code{font-family:var(--mono);background:var(--card-2);border:1px solid var(--line);border-radius:4px;
  padding:.5px 5px;font-size:12.5px}
a{color:var(--info);text-decoration:none}
a:hover{text-decoration:underline}
header.top{position:sticky;top:0;z-index:20;background:rgba(14,17,22,.94);backdrop-filter:blur(6px);
  border-bottom:1px solid var(--line);padding:14px 22px 10px}
header.top h1{margin:0;font-size:17px;letter-spacing:.02em}
header.top .sub{color:var(--muted);font-size:12.5px;margin-top:4px}
nav.toc{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
nav.toc a{font-size:12.5px;color:var(--muted);border:1px solid var(--line);border-radius:999px;
  padding:2px 10px;background:var(--card)}
nav.toc a:hover{color:var(--fg);border-color:var(--info);text-decoration:none}
main{max-width:1180px;margin:0 auto;padding:22px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;
  margin:0 0 18px;scroll-margin-top:126px}
section>h2{margin:0 0 4px;font-size:15.5px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
section>h2 .tag{font:11px var(--mono);color:var(--dim);font-weight:400}
.lead{color:var(--muted);font-size:13px;margin:0 0 14px}
h3{font-size:13.5px;margin:18px 0 6px}
p{margin:8px 0}
.muted{color:var(--muted)}.dim{color:var(--dim)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--bg-soft);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.kpi .v{font:600 20px/1.25 var(--mono);letter-spacing:-.01em}
.kpi .k{color:var(--muted);font-size:11.5px;margin-top:3px}
.kpi .s{color:var(--dim);font-size:11px;margin-top:2px}
.kpi.ok .v{color:var(--ok)}.kpi.bad .v{color:var(--bad)}.kpi.warn .v{color:var(--warn)}
.stack{display:flex;height:14px;border-radius:7px;overflow:hidden;background:var(--bg-soft);
  border:1px solid var(--line);margin:16px 0 6px}
.stack i{display:block;height:100%}
.stack-key{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--muted)}
.stack-key b{font-family:var(--mono);font-weight:500;color:var(--fg)}
.dot{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px;vertical-align:1px}
.dot.ok{background:var(--ok)}.dot.bad{background:var(--bad)}.dot.warn{background:var(--warn)}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0}
.toolbar input{background:var(--bg-soft);border:1px solid var(--line);border-radius:8px;color:var(--fg);
  padding:6px 10px;font-size:12.5px;min-width:230px}
.toolbar input:focus{outline:none;border-color:var(--info)}
.toolbar .count,.toolbar .dim{color:var(--dim);font-size:12px}
.tscroll{overflow:auto;max-height:520px;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:12.8px}
caption{caption-side:top;text-align:left;color:var(--dim);font-size:12px;padding:8px 10px;
  background:var(--bg-soft)}
th,td{padding:7px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
thead th{position:sticky;top:0;z-index:2;background:var(--card-2);color:var(--muted);font-weight:600;
  font-size:12px;white-space:nowrap}
table.sortable thead th{cursor:pointer}
table.sortable thead th:hover{color:var(--fg)}
table.sortable thead th::after{content:"↕";margin-left:5px;color:var(--dim);font-size:10px}
table.sortable thead th.asc::after{content:"↑";color:var(--info)}
table.sortable thead th.desc::after{content:"↓";color:var(--info)}
tbody tr:hover{background:var(--bg-soft)}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.key{font-family:var(--mono);white-space:nowrap}
td.txt{max-width:460px;word-break:break-word}
td .alt{color:var(--muted);font-size:12px;margin-top:2px}
.bar{display:block;height:9px;border-radius:5px;background:var(--bg-soft);min-width:110px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:5px;background:var(--info)}
.bar i.bad{background:var(--bad)}.bar i.warn{background:var(--warn)}.bar i.ok{background:var(--ok)}
.chip{display:inline-block;font:11px/1.5 var(--mono);border-radius:999px;padding:0 8px;
  border:1px solid var(--line);color:var(--muted);white-space:nowrap}
.chip.c-audio{color:#7ee787;border-color:#2a5b3a}
.chip.c-voice{color:#d2a8ff;border-color:#4d3363}
.chip.c-spk{color:#ffa657;border-color:#5c4326}
.chip.c-asr{color:#79c0ff;border-color:#2b4a68}
.chip.c-text{color:#ffd58a;border-color:#5b4a24}
.chip.c-tag{color:#ff7b72;border-color:#5c2f2c}
.chip.c-dedup{color:#a5d6ff;border-color:#33506b}
.chip.c-sys{color:#8b949e;border-color:#39424e}
.pill{display:inline-block;font:11px/1.5 var(--mono);border-radius:5px;padding:0 7px;white-space:nowrap}
.pill.adopted{color:var(--ok);background:rgba(63,185,80,.12);border:1px solid rgba(63,185,80,.35)}
.pill.excluded{color:var(--bad);background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.32)}
.pill.pending_review{color:var(--warn);background:rgba(210,153,34,.12);
  border:1px solid rgba(210,153,34,.32)}
.pill.other{color:var(--muted);background:var(--bg-soft);border:1px solid var(--line)}
.howto{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.note{background:var(--bg-soft);border:1px solid var(--line);border-left:3px solid var(--info);
  border-radius:10px;padding:12px 14px}
.note h4{margin:0 0 6px;font-size:13px}
.note ul{margin:0;padding-left:18px}
.note li{margin:5px 0;color:var(--muted);font-size:12.6px}
.note li b{color:var(--fg)}
.legend-inline{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:14px}
.hists{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;margin-top:6px}
.hist{display:grid;grid-template-columns:auto 1fr auto;gap:6px;align-items:center;font-size:11.5px;
  font-family:var(--mono)}
.hist .hlab{color:var(--dim);white-space:nowrap}
.hist .htrack{height:10px;background:var(--bg-soft);border-radius:5px;overflow:hidden}
.hist .htrack i{display:block;height:100%;background:var(--info);border-radius:5px}
.hist .hnum{color:var(--muted);text-align:right;min-width:3.5em}
.hcap{color:var(--dim);font-size:11.5px;margin:0 0 6px;font-family:var(--mono)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.chips .b{background:var(--bg-soft);border:1px solid var(--line);border-radius:8px;padding:5px 10px;
  font-size:12px;color:var(--muted)}
.chips .b b{font-family:var(--mono);color:var(--fg);font-weight:600}
details{border:1px solid var(--line);border-radius:10px;padding:0 14px;margin:12px 0;background:var(--bg-soft)}
details>summary{cursor:pointer;padding:9px 0;color:var(--muted);font-size:12.8px;user-select:none}
details[open]>summary{border-bottom:1px solid var(--line);margin-bottom:10px;color:var(--fg)}
footer{max-width:1180px;margin:0 auto;padding:0 22px 40px;color:var(--dim);font-size:12px}
@media (max-width:720px){main{padding:14px}section{padding:14px}td.txt{max-width:none}}
@media print{
  :root{--bg:#fff;--bg-soft:#f5f6f8;--card:#fff;--card-2:#eef0f3;--fg:#111;--muted:#444;--dim:#666;
    --line:#ccd2da}
  header.top{position:static;backdrop-filter:none}
  .tscroll{max-height:none;overflow:visible}
}
"""

_JS = """
(function () {
  // 数値として読めるセルは数値で、それ以外は日本語列順でソートする (ヘッダクリックで切替)
  function cellValue(td) {
    var raw = (td.textContent || '').trim();
    var n = Number(raw.replace(/[,，\\s%＋]/g, ''));
    return isNaN(n) || raw === '' ? raw : n;
  }
  document.querySelectorAll('table.sortable').forEach(function (table) {
    var ths = table.querySelectorAll('thead th');
    ths.forEach(function (th, idx) {
      th.addEventListener('click', function () {
        var body = table.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows);
        var dir = th.classList.contains('asc') ? -1 : 1;
        ths.forEach(function (o) { o.classList.remove('asc', 'desc'); });
        th.classList.add(dir === 1 ? 'asc' : 'desc');
        rows.sort(function (a, b) {
          var x = cellValue(a.cells[idx]), y = cellValue(b.cells[idx]);
          if (typeof x === 'number' && typeof y === 'number') return (x - y) * dir;
          return String(x).localeCompare(String(y), 'ja') * dir;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
  });
  // 長い一覧 (見送り・x_cer・除外内訳) はその場で絞り込めるようにする (判定の順番付け用)
  document.querySelectorAll('input[data-filter-target]').forEach(function (input) {
    var table = document.getElementById(input.getAttribute('data-filter-target'));
    if (!table) return;
    var rows = table.tBodies[0].rows;
    var count = document.querySelector('[data-count-for="' + table.id + '"]');
    function apply() {
      var q = input.value.trim().toLowerCase(), n = 0;
      Array.prototype.forEach.call(rows, function (r) {
        var hit = !q || (r.textContent || '').toLowerCase().indexOf(q) >= 0;
        r.style.display = hit ? '' : 'none';
        if (hit) n++;
      });
      if (count) count.textContent = n + ' 行表示中';
    }
    input.addEventListener('input', apply);
    apply();
  });
})();
"""


def _e(v) -> str:
    """全テキストセルで使う HTML エスケープ (セリフ・パス名に < > & が混ざっても壊れないようにする)。"""
    return html.escape("" if v is None else str(v), quote=True)


def _reason_key(key: str) -> str:
    """dropped_by の生キーを凡例引きしやすい形に丸める (diversity:similar:<utt> → diversity:similar)。"""
    if key.startswith("diversity:cap:"):
        return ":".join(key.split(":")[:3])
    if key.startswith("diversity:similar:"):
        return "diversity:similar"
    return key


def _reason_info(key: str) -> tuple[str, str]:
    """(分類, 説明)。未登録のキーでも「その他」として落ちずに、キー名を説明に落とす。"""
    k = _reason_key(key)
    if k in _REASON_LEGEND:
        return _REASON_LEGEND[k]
    for prefix, info in _REASON_PREFIXES:
        if k.startswith(prefix):
            return info
    return ("その他", f"このビルドで凡例未登録の理由キー (コード/設定の {k} を参照)")


def _chip(key: str) -> str:
    cat = _reason_info(key)[0]
    return f'<span class="chip {_CATEGORY_CLASS.get(cat, "c-sys")}">{_e(cat)}</span>'


def _pill(status: str) -> str:
    cls = status if status in ("adopted", "excluded", "pending_review") else "other"
    return f'<span class="pill {_e(cls)}">{_e(status)}</span>'


def _bar(pct: float, cls: str = "") -> str:
    return f'<span class="bar"><i class="{cls}" style="width:{max(0.0, min(100.0, pct)):.1f}%"></i></span>'


def _fmt(v) -> str:
    """閾値の表示 (None は em-dash、bool は true/false、float は 0.45 / 0.0005 のように短く)。"""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _num(v, nd: int = 2) -> str:
    """スコアの表示 (桁を揃えて表を縦に読みやすくする。None は em-dash)。"""
    if v is None:
        return "—"
    if isinstance(v, (bool, int)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 1e6 else f"{v:g}"  # 0 は "0.00" と桁を見せて出す
    return str(v)


# ヒストグラムの目盛り書式 (f-string の中では {{ }} が使えないのでここで定義する)
FMT_INT = "{:.0f}"
FMT_1 = "{:.1f}"


def _hist_html(vals, *, fmt: str = FMT_1, unit: str = "", bins: int = 10) -> str:
    """分布を CSS 横棒ヒストグラムで出す (█ 文字は等幅フォント前提で崩れるため使わない)。"""
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return '<p class="muted">分布を出せるスコアが 1 件もありません (対応するプロバイダ未有効か未記録)</p>'
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        hi = lo + 1e-9
    counts = [0] * bins
    for v in xs:
        counts[min(int((v - lo) / (hi - lo) * bins), bins - 1)] += 1
    m = max(counts) or 1
    rows = "".join(
        f'<span class="hlab">{fmt.format(lo + (hi - lo) * i / bins)}{unit}</span>'
        f'<span class="htrack"><i style="width:{100 * c / m:.1f}%"></i></span>'
        f'<span class="hnum">{c}</span>'
        for i, c in enumerate(counts)
    )
    cap = (
        f"n={len(xs)} / min {fmt.format(lo)}{unit} / 中央値 {fmt.format(float(np.median(xs)))}{unit}"
        f" / max {fmt.format(hi)}{unit}"
    )
    return f'<p class="hcap">{_e(cap)}</p><div class="hist">{rows}</div>'


def _sensitivity_stats(records: list[Record], cfg: Config) -> list[dict]:
    """「その閾値 1 つだけ緩めた/締めたときにひっくり返る本数」+ 凡例用のメタ情報。

    単独失格 (他の理由で落ちていない) レコードだけを数える近似なので、実際の rescore での
    増減はこれより少なくなる。flag ゲートは緩める/締めるが定義できないので感度列は空。
    """
    t = cfg.thresholds
    excluded = [r for r in records if r.status == "excluded"]
    adopted = [r for r in records if r.status == "adopted"]
    out: list[dict] = []
    for tkey, skey, mode in CHECKS:
        if tkey not in t:
            continue  # キーが無い = チェック自体が走っていない
        v = t[tkey]
        if v is None or (mode == "flag" and not v):
            continue
        fails = sum(1 for r in excluded if tkey in (r.scores.get("failed_checks") or []))
        solo = [r for r in excluded if (r.scores.get("failed_checks") or []) == [tkey]]
        gain = loss = relaxed = tightened = None
        if mode != "flag":
            relaxed = v * (0.8 if mode == "min" else 1.25)
            tightened = v * (1.2 if mode == "min" else 0.8)
            gain = sum(
                1
                for r in solo
                if r.scores.get(skey) is not None
                and (r.scores[skey] >= relaxed if mode == "min" else r.scores[skey] <= relaxed)
            )
            loss = sum(
                1
                for r in adopted
                if r.scores.get(skey) is not None
                and (r.scores[skey] < tightened if mode == "min" else r.scores[skey] > tightened)
            )
        out.append(
            {
                "key": tkey,
                "score": skey,
                "mode": mode,
                "value": v,
                "relaxed": relaxed,
                "gain": gain,
                "tightened": tightened,
                "loss": loss,
                "fails": fails,
            }
        )
    # 話速レンジは CHECKS テーブル外 (モーラ数が取れない発話はスキップ) なのでここで足す
    lo_t, hi_t = t.get("min_mora_rate"), t.get("max_mora_rate")
    if lo_t is not None or hi_t is not None:
        out.append(
            {
                "key": "mora_rate",
                "score": "mora_rate",
                "mode": "range",
                "value": f"{_fmt(lo_t)} 〜 {_fmt(hi_t)}",
                "relaxed": None,
                "gain": None,
                "tightened": None,
                "loss": None,
                "fails": sum(
                    1 for r in excluded if "mora_rate" in (r.scores.get("failed_checks") or [])
                ),
            }
        )
    return out


def _gate_table_html(records: list[Record], cfg: Config) -> str:
    """有効なゲート = 凡例 + 現在値 + 感度を 1 枚に寄せる。「除外内訳」と見比べるだけで
    どのゲートが何本落とし、何を緩めれば何本戻るか が分かるようにする。"""
    stats = _sensitivity_stats(records, cfg)
    max_fails = max((s["fails"] for s in stats), default=0) or 1
    rows = []
    for s in sorted(stats, key=lambda s: -s["fails"]):
        cat, desc = _reason_info(s["key"])
        # 0 を "+0"/"-0" と出すとノイズなので、増減が無いときは 0 を素で出す
        gain = "—" if s["gain"] is None else (f"+{s['gain']}" if s["gain"] else "0")
        loss = "—" if s["loss"] is None else (f"-{s['loss']}" if s["loss"] else "0")
        rows.append(
            "<tr>"
            f'<td class="key">{_e(s["key"])}</td>'
            f'<td><span class="chip {_CATEGORY_CLASS.get(cat, "c-sys")}">{_e(cat)}</span> {_e(desc)}'
            f'<div class="alt">スコア <code>{_e(s["score"])}</code> / 判定 {_e(s["mode"])}</div></td>'
            f'<td class="num">{_e(_fmt(s["value"]))}</td>'
            f'<td class="num">{s["fails"]}</td>'
            f'<td style="min-width:110px">{_bar(100 * s["fails"] / max_fails, "bad")}</td>'
            f'<td class="num dim">{_e(_fmt(s["relaxed"]))}</td>'
            f'<td class="num">{_e(gain)}</td>'
            f'<td class="num dim">{_e(_fmt(s["tightened"]))}</td>'
            f'<td class="num">{_e(loss)}</td>'
            "</tr>"
        )
    return (
        '<div class="toolbar"><input type="search" placeholder="ゲート名・説明で絞り込む" '
        'data-filter-target="gate-table"><span class="count" data-count-for="gate-table"></span>'
        '<span class="dim">ヘッダをクリックするとその列でソート</span></div>'
        '<div class="tscroll"><table class="sortable" id="gate-table">'
        "<caption>該当 = このゲートで落ちた本数 (他の理由と重複して落ちた分も含む)。緩和後/厳格化後の"
        "採用+/- は「そのゲート単独で落ちた発話だけ」を数える近似なので、実際の rescore での増減は少ない</caption>"
        "<thead><tr><th>ゲート (閾値キー)</th><th>何を見るのか</th><th class=num>現在値</th>"
        "<th class=num>該当</th><th></th><th class=num>緩和後</th><th class=num>採用+</th>"
        "<th class=num>厳格化後</th><th class=num>採用-</th></tr></thead>"
        f"<tbody>{''.join(rows) or '<tr><td colspan=9>(有効なゲートがありません)</td></tr>'}</tbody></table></div>"
    )


def _exclusion_table_html(records: list[Record], dropped: Counter) -> str:
    """除外内訳を「分類 chip + 見方 + 本数バー」で出す (キー生のままだと読めない前提の表)。"""
    excluded_total = sum(1 for r in records if r.status == "excluded") or 1
    rows = "".join(
        "<tr>"
        f'<td class="key">{_e(_reason_key(key))}</td>'
        f"<td>{_chip(key)}</td>"
        f'<td class="txt">{_e(_reason_info(key)[1])}</td>'
        f'<td class="num">{n}</td>'
        f'<td style="min-width:140px">{_bar(100 * n / excluded_total, "bad")}</td>'
        f'<td class="num dim">{100 * n / excluded_total:.1f}%</td>'
        "</tr>"
        for key, n in dropped.most_common()
    )
    return (
        '<div class="toolbar"><input type="search" placeholder="失格理由キー・説明で絞り込む" '
        'data-filter-target="drop-table"><span class="count" data-count-for="drop-table"></span>'
        '<span class="dim">1 レコードが複数理由を持つので、件数の合計は除外本数より増える</span></div>'
        '<div class="tscroll"><table class="sortable" id="drop-table">'
        "<thead><tr><th>失格理由</th><th>分類</th><th>見方</th><th class=num>件数</th><th></th>"
        "<th class=num>除外に占める</th></tr></thead>"
        f"<tbody>{rows or '<tr><td colspan=6>(除外されたレコードはありません)</td></tr>'}</tbody>"
        "</table></div>"
    )


def _diversity_metrics_html(records: list[Record]) -> str:
    """多様性メトリクス (平均ペアwise類似度・音素 trigram 被覆率・話速/F0/長さ分布) を選別前後で出す。

    「pool」= Stage 5b に入る候補集合 (閾値と重複排除を通過 = 採用 + 多様性で落ちた分)。
    5b の効き目 (間引いた後に類似度が下がり、被覆率が保たれているか) を測定できるようにする。
    """
    import random

    from .stage5_dedup import _jaccard, _phoneme_trigrams

    pool = [
        r
        for r in records
        if r.text
        and not (r.scores.get("failed_checks") or r.scores.get("errors"))
        and not r.dup_reason
    ]
    adopted = [r for r in records if r.status == "adopted" and r.text]

    def mean_sim(rows):
        sets = [_phoneme_trigrams(r.text) for r in rows]
        n = len(sets)
        if n < 2:
            return None
        rng = random.Random(0)  # 固定シードで再現可能に (レポート差分が実行間で揺れない)
        tot = cnt = 0
        for _ in range(min(n * (n - 1) * 4, 20000)):  # 総対ペアはサンプル近似 (大コーパスでも有界)
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            tot += _jaccard(sets[i], sets[j])
            cnt += 1
        return tot / cnt if cnt else None

    def trigram_union(rows):
        out: set = set()
        for r in rows:
            out |= _phoneme_trigrams(r.text)
        return out

    tri_pool = trigram_union(pool)
    cov = len(trigram_union(adopted) & tri_pool) / len(tri_pool) if tri_pool else None
    sim_pool, sim_adopted = mean_sim(pool), mean_sim(adopted)

    def fmt4(v):
        return f"{v:.4f}" if v is not None else "—"

    cov_txt = f"{100 * cov:.1f}%" if cov is not None else "—"
    kpis = (
        '<div class="kpis">'
        f'<div class="kpi"><div class="v">{len(pool)}</div><div class="k">候補 pool 本数</div>'
        '<div class="s">閾値と重複排除を通過した集合</div></div>'
        f'<div class="kpi ok"><div class="v">{len(adopted)}</div><div class="k">採用 (5b 後)</div></div>'
        f'<div class="kpi"><div class="v">{fmt4(sim_pool)} → {fmt4(sim_adopted)}</div>'
        '<div class="k">平均ペアwise類似度 (選別前 → 後)</div>'
        '<div class="s">音素三量 Jaccard。低いほど多様</div></div>'
        f'<div class="kpi"><div class="v">{cov_txt}</div><div class="k">音素三量被覆率 (採用/候補)</div>'
        '<div class="s">高いほど間引きによる音素の欠けが少ない</div></div>'
        "</div>"
    )
    hists = (
        '<div class="hists">'
        f"<div><h3>話速 mora_rate (adopted)</h3>{_hist_html([r.scores.get('mora_rate') for r in adopted])}</div>"
        f"<div><h3>F0 中央値 f0_median_hz (adopted)</h3>{_hist_html([r.scores.get('f0_median_hz') for r in adopted], fmt=FMT_INT, unit=' Hz')}</div>"
        f"<div><h3>発話長 seg_dur (adopted)</h3>{_hist_html([r.scores.get('seg_dur') for r in adopted], fmt=FMT_1, unit='s')}</div>"
        "</div>"
    )
    return kpis + hists


def write_report(records: list[Record], outdir: Path, cfg: Config) -> None:
    """除外内訳・閾値感度・カバレッジ (音素/モーラ/F0) を、見方と凡例つき HTML で出す。"""
    from .stage5_dedup import phoneme_tokens

    dv = cfg.data.get("diversity", {})
    # 個体 ID を含むキー (diversity:similar:<utt> 等) を丸めて集計するので、生キーのまま数えない
    dropped = Counter(
        _reason_key(k) for r in records if r.status == "excluded" for k in (r.dropped_by or [])
    )
    adopted = [r for r in records if r.status == "adopted"]
    excluded = [r for r in records if r.status == "excluded"]
    pending = [r for r in records if r.status == "pending_review"]
    dup_groups = {r.dup_group_id for r in records if r.dup_group_id}
    diversity_dropped = sum(1 for r in records if r.diversity_reason)
    skipped = Counter(
        str(r.scores.get("stage1_skipped")).split(":")[0]
        for r in records
        if r.scores.get("stage1_skipped")
    )
    # F0 が計測できない発話 (None) はレンジ計算から除外する (f0_median_hz は 0 にならないので is not None で十分)
    f0s = [v for r in adopted if (v := r.scores.get("f0_median_hz")) is not None]
    limited = sum(1 for r in adopted if r.scores.get("limited"))
    total = len(records)
    total_sec = sum(r.scores.get("seg_dur") or 0 for r in adopted)
    rate = 100 * len(adopted) / total if total else 0.0

    src_rows = "".join(
        f'<tr><td class="txt">{_e(k)}</td><td class="num">{v}</td>'
        f'<td style="min-width:120px">{_bar(100 * v / (total or 1))}</td></tr>'
        for k, v in Counter(r.src for r in records).most_common()
    )
    skip_rows = "".join(
        f'<tr><td class="key">{_e(k)}</td><td>{_chip(k)}</td><td class="num">{v}</td></tr>'
        for k, v in skipped.most_common()
    )

    # 音素被覆率 (adopted 全文に出現した音素トークン) とモーラ数分布
    tokens: Counter = Counter()
    for r in adopted:
        tokens.update(set(phoneme_tokens(r.text)))
    rare = ", ".join(f"{k}({v})" for k, v in sorted(tokens.items(), key=lambda kv: kv[1])[:8])
    hist_mora = _hist_html([r.scores.get("mora_count") or 0 for r in adopted], fmt=FMT_INT)
    hist_rate = _hist_html([r.scores.get("mora_rate") for r in adopted])
    hist_dur = _hist_html([r.scores.get("seg_dur") for r in adopted], fmt=FMT_1, unit="s")

    # 見送りは「機械が自動採用を避けた」もの。第 2 系統の出力が無いと x_cer 理由を判断できないので併記する
    pend_rows = "".join(
        "<tr>"
        f'<td class="key">{_e(r.utt_id)}</td>'
        f'<td class="key">{_e(r.review_reason or "")}</td>'
        f'<td class="txt">{_e(r.text)}'
        + (
            f'<div class="alt">第 2 系統: {_e(r.scores.get("asr2_text"))}</div>'
            if r.scores.get("asr2_text") and str(r.scores.get("asr2_text")) != (r.text or "")
            else ""
        )
        + "</td>"
        f'<td class="num">{_e(_num(r.scores.get("asr_conf")))}</td>'
        f'<td class="num">{_e(_num(r.scores.get("x_cer")))}</td>'
        f'<td class="num">{_e(_num(r.scores.get("lm_ppl"), 1))}</td>'
        f'<td class="txt dim">{_e(r.scores.get("yomi_bad_words") or "")}</td>'
        "</tr>"
        for r in sorted(pending, key=lambda r: r.scores.get("asr_conf") or 0)
    )
    # 読みチェック (yomi.py) で出た読めない語。除外・見送りどちらにも残るので全レコードから数える
    # (on_g2p_fail="dict_data" のとき辞書へ登録する語の決定材料になる)
    yomi_words = Counter(
        w for r in records for w in str(r.scores.get("yomi_bad_words") or "").split(",") if w
    )
    yomi_max = yomi_words.most_common(1)[0][1] if yomi_words else 1
    yomi_rows = "".join(
        f'<tr><td class="key">{_e(k)}</td><td class="num">{v}</td>'
        f'<td style="min-width:120px">{_bar(100 * v / yomi_max)}</td></tr>'
        for k, v in yomi_words.most_common(50)
    )
    # 文末の開き分布。2 (語・発話が途中で切れている) は常に落とす。1 (言い淀み止め) は既定で採用なので、
    # 中身を一覧化して `max_text_tail_open = 0` で rescore し直せるようにする (再 run 不要)
    gate_tail = cfg.thresholds.get("max_text_tail_open")
    tail_dist = Counter(int(r.scores.get("text_tail_open") or 0) for r in records if r.text)

    def tail_rows(sev: int, limit: int) -> str:
        rows = [r for r in records if r.text and int(r.scores.get("text_tail_open") or 0) == sev]
        return (
            "".join(
                f'<tr><td class="key">{_e(r.utt_id)}</td>'
                f'<td class="key">{_e(r.scores.get("text_tail_open_reason"))}</td>'
                f'<td class="txt">{_e(r.text)}</td><td>{_pill(r.status)}</td></tr>'
                for r in rows[:limit]
            )
            or "<tr><td colspan=4>(なし)</td></tr>"
        )

    tail_html = (
        '<div class="chips">'
        f'<span class="b">0 閉じている <b>{tail_dist.get(0, 0)}</b></span>'
        f'<span class="b">1 言い淀み止め <b>{tail_dist.get(1, 0)}</b></span>'
        f'<span class="b">2 途中で切れている <b>{tail_dist.get(2, 0)}</b></span>'
        f'<span class="b">実ゲート <b>max_text_tail_open = {_e(_fmt(gate_tail))}</b></span>'
        "</div>"
        "<h3>severity 2 — 語・発話が途中で切れている (常に除外)</h3>"
        '<div class="tscroll"><table><thead><tr><th>utt_id</th><th>理由</th><th>セリフ</th>'
        f"<th>status</th></tr></thead><tbody>{tail_rows(2, 50)}</tbody></table></div>"
        f"<details><summary>severity 1 — 条件節・連用形中止・連体止め ({tail_dist.get(1, 0)} 件。"
        "max_text_tail_open を 0 にすれば落ちる)</summary>"
        '<div class="tscroll"><table><thead><tr><th>utt_id</th><th>理由</th><th>セリフ</th>'
        f"<th>status</th></tr></thead><tbody>{tail_rows(1, 200)}</tbody></table></div></details>"
    )

    # 2 系統 ASR の不一致分布。max_x_cer で切った結果と境界帯を一覧化し、
    # rescore (再 run 不要) で締め直せる判断材料にする。x_cer は読みレベル、表層値も併記する
    xs = [
        (float(r.scores["x_cer"]), r)
        for r in records
        if r.text and r.scores.get("x_cer") is not None
    ]
    xcer_html = (
        '<p class="muted">x_cer 未記録。filters.toml の <code>[asr2]</code> を有効にすると、別系統 ASR との'
        "不一致が読みレベルで出ます (メイン ASR が独りでに正しいとは限らないので、2 系統の食い違いは"
        "「どちらかが壊れている」印として見る)</p>"
    )
    if xs:
        buckets = (
            (0.0, 0.001, "完全一致"),
            (0.001, 0.15, "<0.15"),
            (0.15, 0.3, "0.15-0.3"),
            (0.3, 0.45, "0.3-0.45"),
            (0.45, 0.7, "0.45-0.7"),
            (0.7, 1.01, "0.7-"),
        )
        cnt: Counter = Counter()
        confs: dict = {}
        for v, r in xs:
            for lo, hi, lbl in buckets:
                if lo <= v < hi:
                    cnt[lbl] += 1
                    c = r.scores.get("asr2_conf")
                    if c is not None:
                        confs.setdefault(lbl, []).append(float(c))
                    break

        def _dist(lbl: str) -> str:
            # 第 2 系統自身の確信度をそえる: 食い違い帯で確信度が低ければ
            # 「第 2 系統も確信なく違うことを言っている」= 除外の証拠として弱い印になる
            cs = confs.get(lbl)
            med = f" (第2確信度 {np.median(cs):.2f})" if cs else ""
            return f"{lbl} = {cnt.get(lbl, 0)}{med}"

        # x_cer = 0 (2 系統が読みレベルで完全一致) を 60 行並べてもノイズなので、食い違ったものだけを並べる
        diff = [t for t in xs if t[0] > 0]
        worst = sorted(diff, key=lambda t: -t[0])[:60]
        xrows = "".join(
            f'<tr><td class="key">{_e(r.utt_id)}</td><td class="num">{v:.2f}</td>'
            f'<td class="num dim">{float(r.scores.get("x_cer_surface") or 0):.2f}</td>'
            f'<td class="num">{_e(_num(r.scores.get("asr2_conf")))}</td>'
            f'<td class="txt">{_e(r.text)}</td>'
            f'<td class="txt dim">{_e(r.scores.get("asr2_text"))}</td>'
            f"<td>{_pill(r.status)}</td></tr>"
            for v, r in worst
        )
        xcer_html = (
            '<div class="chips">'
            + "".join(f'<span class="b">{_e(_dist(lbl))}</span>' for _, _, lbl in buckets)
            + f'<span class="b">実ゲート <b>max_x_cer = {_e(_fmt(cfg.thresholds.get("max_x_cer")))}</b></span>'
            f'<span class="b">review_x_cer <b>{_e(_fmt(cfg.thresholds.get("review_x_cer")))}</b></span></div>'
            "<p>分布の <code>第2確信度</code> はその帯にある第 2 系統 ASR 確信度の中央値です。食い違い帯で "
            "0.4 前後まで落ちるなら第 2 系統自身も確信なく違うことを言っている証拠で、除外の証拠は弱くなります"
            " (実測で食い違いの約 4 割は主 ASR の方が正しい)。落としすぎを感じたら <code>max_x_cer</code> を"
            "上げて rescore — x_cer は manifest にキャッシュ済みなので第 2 ASR を引き直さず数秒で終わります。"
            "境界帯を救いたいときだけ <code>review_x_cer</code> を有効化すると、除外ではなく "
            "<code>pending_review</code> 行き (text_overrides.jsonl で本文を確定できる) になります</p>"
            '<div class="toolbar"><input type="search" placeholder="utt_id・本文で絞り込む" '
            'data-filter-target="xcer-table"><span class="count" data-count-for="xcer-table"></span>'
            '<span class="dim">2 系統が食い違った '
            f"{len(diff)} 件のうち不一致が大きい順に 60 件 (x_cer = 0 の完全一致は並べない)</span></div>"
            '<div class="tscroll"><table class="sortable" id="xcer-table"><thead><tr><th>utt_id</th>'
            "<th class=num>x_cer</th><th class=num>表層</th><th class=num>第2確信度</th>"
            "<th>メイン ASR</th><th>第 2 ASR</th><th>status</th></tr></thead>"
            f"<tbody>{xrows or '<tr><td colspan=7>(なし)</td></tr>'}</tbody></table></div>"
        )

    # --- サマリー KPI + 内訳の積み上げバー ---
    avg_sec = f"{total_sec / len(adopted):.2f}s" if adopted else "—"
    kpis = (
        '<div class="kpis">'
        f'<div class="kpi ok"><div class="v">{len(adopted)}</div><div class="k">採用 (納品)</div>'
        f'<div class="s">{total_sec / 60:.1f} 分 / 平均 {avg_sec}</div></div>'
        f'<div class="kpi bad"><div class="v">{len(excluded)}</div><div class="k">除外</div>'
        f'<div class="s">失格理由 {sum(dropped.values())} 件分 (複数理由合算)</div></div>'
        f'<div class="kpi warn"><div class="v">{len(pending)}</div><div class="k">見送り pending_review</div>'
        '<div class="s">納品には入らない</div></div>'
        f'<div class="kpi"><div class="v">{rate:.1f}%</div><div class="k">採用率</div>'
        f'<div class="s">全 {total} レコード中</div></div>'
        f'<div class="kpi"><div class="v">{len(dup_groups)}</div><div class="k">重複グループ</div>'
        '<div class="s">代表 1 本のみ採用</div></div>'
        f'<div class="kpi"><div class="v">{diversity_dropped}</div><div class="k">多様性で間引き</div>'
        f'<div class="s">metric={_e(dv.get("metric"))} / sim={_e(_fmt(dv.get("sim_threshold")))}</div></div>'
        f'<div class="kpi"><div class="v">{sum(skipped.values())}</div><div class="k">Stage 1 スキップ</div>'
        '<div class="s">音声キャッシュ無し=常に除外</div></div>'
        f'<div class="kpi"><div class="v">{limited}</div><div class="k">ピークリミッタ作動</div>'
        '<div class="s">納品音声の LUFS が目標から外れた本数</div></div>'
        "</div>"
    )
    stack_total = (len(adopted) + len(excluded) + len(pending)) or 1
    stack = (
        '<div class="stack">'
        f'<i style="width:{100 * len(adopted) / stack_total:.2f}%;background:var(--ok)"></i>'
        f'<i style="width:{100 * len(excluded) / stack_total:.2f}%;background:var(--bad)"></i>'
        f'<i style="width:{100 * len(pending) / stack_total:.2f}%;background:var(--warn)"></i>'
        "</div>"
        '<div class="stack-key">'
        f'<span><i class="dot ok"></i>adopted <b>{len(adopted)}</b> ({100 * len(adopted) / stack_total:.1f}%)</span>'
        f'<span><i class="dot bad"></i>excluded <b>{len(excluded)}</b> ({100 * len(excluded) / stack_total:.1f}%)</span>'
        f'<span><i class="dot warn"></i>pending_review <b>{len(pending)}</b> ({100 * len(pending) / stack_total:.1f}%)</span>'
        "</div>"
    )

    rescore_cmd = f"uv run make_dataset rescore Data/{outdir.name} --config filters.toml"
    howto = f"""
<div class="howto">
  <div class="note"><h4>1. status の見方</h4><ul>
    <li>{_pill("adopted")} 全ゲート通過 + 重複/多様性で落ちなかったもの。<b>納品物 (<code>esd.list</code> /
      <code>raw.zip</code>) に入るのはこれだけ</b></li>
    <li>{_pill("excluded")} 失格理由が 1 つ以上。<b>理由は最初の 1 個ではなく全部</b> <code>dropped_by</code> に
      残すので、除外内訳の件数を合計すると除外本数を超える</li>
    <li>{_pill("pending_review")} 機械が自動採用を避けた見送り (ユーザー辞書候補・修復音声など)。
      <b>納品には入らない</b>。下の「見送り」に中身を出す</li>
  </ul></div>

  <div class="note"><h4>2. 失格理由キーの読み方</h4><ul>
    <li><code>min_*</code> = スコアが閾値<b>未満</b>で不合格 / <code>max_*</code> = <b>超過</b>で不合格
      (<code>max_seg_sec</code> だけ「以上」) / <code>reject_*</code> = フラグが立ったら不合格</li>
    <li>キー名は <code>filters.toml [thresholds]</code> そのもの。キーを消す (コメント化) と
      <b>チェック自体が走らない</b> — コード側デフォルトは無い</li>
    <li><b>検証不能=不合格</b>: 閾値が有効なのにスコアが取れない発話は不合格。新しいゲートを有効化して
      rescore すると、そのスコアを持たない manifest は<b>全件落ちる</b> (先に run し直す)</li>
    <li>接頭辞 <code>dedup:</code> = 重複排除、<code>diversity:</code> = 多様性選別 (Stage 5b)、
      <code>too_short</code> / <code>source_error</code> = Stage 1 で音声キャッシュすら作っていない分</li>
  </ul></div>

  <div class="note"><h4>3. 閾値感度表の見方</h4><ul>
    <li>「そのゲート<b>1 つだけ</b>緩めた/締めたときにひっくり返る本数」。<b>他の理由でも落ちている発話は
      数えない</b>近似なので、実際の rescore での増減はこれより少なくなる</li>
    <li>緩和 = <code>min_*</code> は 20% 下げる / <code>max_*</code> は 25% 上げる。厳格化 =
      <code>min_*</code> は 20% 上げる / <code>max_*</code> は 20% 下げる</li>
    <li><b>採用+ が大きく採用- が小さい行</b>が「緩める価値がある」候補。採用- だけ大きい行は締めすぎの警告</li>
    <li><code>該当</code> は他の理由と重複して落ちた本数も含むので、<code>採用+</code> より大きくて普通</li>
  </ul></div>

  <div class="note"><h4>4. 多様性メトリクスの見方</h4><ul>
    <li><b>平均ペアwise類似度</b> (音素三量 Jaccard、低いほど多様)。実測の目安: 別発話 ≤0.05 /
      似た内容 ~0.67 / 完全一致 ~1.0。<b>選別前後で下がれば Stage 5b が効いている</b></li>
    <li><b>音素三量被覆率</b> = 候補集合が出した音素三量文字のうち、採用集合がカバーする割合。
      下がっているほど「間引きで音素が欠けた」 (= <code>protect_coverage</code> の効き目を確かめる)</li>
    <li>話速・F0・発話長のヒストグラムは採用集合のみ。1 バケットに集中していればスタイル片寄りのサイン
      (<code>max_bucket_share</code> で整流化できる)</li>
  </ul></div>

  <div class="note"><h4>5. x_cer と文末の開き</h4><ul>
    <li><code>x_cer</code> は<b>読みレベル</b>の 2 系統 ASR 間 CER (同音異字・句読点・文末の伸ばしは一致扱い)。
      表層値 <code>x_cer_surface</code> は中央値 0.11 になるのでゲートに使わない — 参考列として並べる</li>
    <li>食い違い帯の<b>第 2 確信度が低い</b>なら、第 2 系統も確信なく違うことを言っている = 除外の証拠は弱い</li>
    <li><code>text_tail_open</code>: 0=閉じた文 / 1=言い淀み止め (既定は採用) / 2=途中で切れ (常に除外)。
      1 も落とすなら <code>max_text_tail_open = 0</code> で rescore</li>
  </ul></div>

  <div class="note"><h4>6. 数字を動かすには</h4><ul>
    <li>閾値を変えて再判定 (音声・ASR は再実行しない): <code>{_e(rescore_cmd)}</code></li>
    <li>本文を直す / 判定を確定する: <code>{_e(outdir.name)}/text_overrides.jsonl</code> に 1 行 1 レコード
      (<code>{{"utt_id": "...", "text": "修正後"}}</code> や
      <code>{{"utt_id": "...", "status": "adopted"}}</code>) を置いて rescore。<b>dedup・話速中央値・多様性選別より
      前</b>に効き、status 上書きは自動判定より常に優先される</li>
    <li>生成日時・使用閾値スナップショット・ライブラリ版・commit は <code>run.json</code>、
      1 レコード分の全スコアは <code>manifest.jsonl</code> にある</li>
    <li>音声はこのページには埋め込まない。聴くときは <code>.cache/&lt;utt_id&gt;.flac</code> を直接開く
      (納品済みの音声は <code>raw/</code>)</li>
  </ul></div>
</div>
<div class="legend-inline"><span class="dim">分類カラー凡例:</span>
  <span class="chip c-audio">音声・音質・欠陥</span>
  <span class="chip c-voice">声質・ピッチ</span>
  <span class="chip c-spk">話者</span>
  <span class="chip c-asr">文字起こし・照合・境界</span>
  <span class="chip c-text">テキスト・読み・話速</span>
  <span class="chip c-tag">音響タグ</span>
  <span class="chip c-dedup">重複・多様性</span>
  <span class="chip c-sys">前段・実行・手動</span>
</div>"""

    doc = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset Report — {_e(outdir.name)}</title>
<style>{_CSS}</style></head>
<body>
<header class="top">
  <h1>Dataset Report <span class="dim" style="font-family:var(--mono)">/ {_e(outdir.name)}</span></h1>
  <div class="sub">全 {total} レコード → 採用 {len(adopted)} 本 ({total_sec / 60:.1f} 分) / 除外
    {len(excluded)} / 見送り {len(pending)} ／ 採用率 {rate:.1f}% ／ 納品物は <code>esd.list</code> +
    <code>raw.zip</code>。この HTML はデータセットディレクトリに置いたままブラウザで開く
    (外部 CDN 依存なし)。表はヘッダでソート、検索ボックスで絞り込みできます</div>
  <nav class="toc">
    <a href="#summary">サマリー</a><a href="#howto">見方・凡例</a><a href="#excluded">除外内訳</a>
    <a href="#gates">ゲート別感度</a><a href="#diversity">多様性</a><a href="#tail">文末の開き</a>
    <a href="#xcer">2 系統 ASR</a><a href="#coverage">カバレッジ</a><a href="#pending">見送り</a>
    <a href="#sources">ソース・スキップ</a><a href="#yomi">読めない語</a>
  </nav>
</header>
<main>
<section id="summary"><h2>サマリー</h2>
  <p class="lead">機械が自動判定した結果の内訳と、閾値をどう動かすかの判断材料を同じページに置く。
  数字の横に必ず「何を見る数字か」を添えている。</p>
  {kpis}{stack}
</section>

<section id="howto"><h2>見方・凡例<span class="tag">この後の表を読むための前提</span></h2>
  <p class="lead">失格理由のキーは <code>filters.toml</code> の閾値キーそのもの。下の凡例を引けば
  「どのゲートが、何を根拠に、何本落としたか」がそのまま分かる。</p>
  {howto}
</section>

<section id="excluded"><h2>除外内訳<span class="tag">全失格理由 (1 レコードに複数理由あり)</span></h2>
  {_exclusion_table_html(records, dropped)}
</section>

<section id="gates"><h2>ゲート別の該当本数と閾値感度</h2>
  <p class="lead">有効なゲートを 1 枚に並べる。「このゲートは何本落とし、何を緩めれば何本戻り、
  何を締めると何本失うか」を横に並べて見せるのが狙い。</p>
  {_gate_table_html(records, cfg)}
</section>

<section id="diversity"><h2>重複排除と多様性選別 (Stage 5a / 5b)<span class="tag">選別前後</span></h2>
  <p class="lead">重複排除で {len(dup_groups)} グループ (代表 1 本のみ採用)、多様性選別で {diversity_dropped} 本を
  間引き。設定: protect_coverage={_e(_fmt(dv.get("protect_coverage")))} /
  gain_lambda={_e(_fmt(dv.get("gain_lambda", 0)))} / max_bucket_share={_e(_fmt(dv.get("max_bucket_share", 0)))} /
  max_source_share={_e(_fmt(dv.get("max_source_share", 0)))} / target_count={_e(_fmt(dv.get("target_count", 0)))}</p>
  <h3>多様性メトリクス (候補 pool → 採用)</h3>
  {_diversity_metrics_html(records)}
</section>

<section id="tail"><h2>文末の開き<span class="tag">0=閉じる / 1=言い淀み止め / 2=途中で切れ</span></h2>
  {tail_html}
</section>

<section id="xcer"><h2>2 系統 ASR の不一致<span class="tag">x_cer = 読みレベル CER</span></h2>
  {xcer_html}
</section>

<section id="coverage"><h2>カバレッジ (音素・モーラ・F0)</h2>
  <p class="lead">採用集合が「どんな音・どんな長さ・どんな高さの声」をカバーしているか。学習側の音素欠け
  (未知音素で落ちる/特徴が偏る) を先に拾うためのセクション。</p>
  <div class="kpis">
    <div class="kpi"><div class="v">{len(tokens)}</div><div class="k">出現音素 (adopted)</div>
      <div class="s">希少音素: {_e(rare or "—")}</div></div>
    <div class="kpi"><div class="v">{_e(_num(min(f0s) if f0s else None, 1))} – {_e(_num(max(f0s) if f0s else None, 1))} Hz</div>
      <div class="k">F0 中央値レンジ (adopted)</div>
      <div class="s">中央値 {_e(_num(float(np.median(f0s)) if f0s else None, 1))} Hz / 計測 {len(f0s)} 本</div></div>
  </div>
  <div class="hists">
    <div><h3>モーラ数分布 (adopted)</h3>{hist_mora}</div>
    <div><h3>話速 mora_rate (adopted)</h3>{hist_rate}</div>
    <div><h3>発話長 seg_dur (adopted)</h3>{hist_dur}</div>
  </div>
</section>

<section id="pending"><h2>見送り (pending_review)<span class="tag">納品には入らない</span></h2>
  <p class="lead">機械の自己申告で通したくないもの (<code>[repair]</code> による修復音声、
  <code>on_g2p_fail</code>/<code>on_unknown_tokens</code> を <code>dict_data</code> にした場合のユーザー辞書候補、
  <code>review_lm_ppl</code>/<code>review_x_cer</code> の受け皿) を自動採用しなかった一覧。確定させるなら
  <code>text_overrides.jsonl</code> に <code>status</code> を書いて rescore する (<a href="#howto">見方 6</a>)。
  ASR 信頼度の低い順に並べてある。</p>
  <div class="toolbar"><input type="search" placeholder="utt_id・理由・本文で絞り込む"
    data-filter-target="pend-table"><span class="count" data-count-for="pend-table"></span></div>
  <div class="tscroll"><table class="sortable" id="pend-table"><thead><tr><th>utt_id</th><th>理由</th>
    <th>セリフ</th><th class=num>asr_conf</th><th class=num>x_cer</th><th class=num>lm_ppl</th>
    <th>読めない語</th></tr></thead>
    <tbody>{pend_rows or '<tr><td colspan=7>(なし)</td></tr>'}</tbody></table></div>
</section>

<section id="sources"><h2>投入ソース別件数 / Stage 1 スキップ / ピークリミッタ</h2>
  <div class="tscroll"><table><caption>投入ファイル 1 本 (=1 ソース) あたりのセグメント数。長尺 1 本に
    引きずられていないかを見る (採用シェアの上限は <code>diversity.max_source_share</code>)</caption>
    <thead><tr><th>ソース</th><th class=num>レコード</th><th></th></tr></thead>
    <tbody>{src_rows or '<tr><td colspan=3>(なし)</td></tr>'}</tbody></table></div>
  <h3>Stage 1 スキップ (短すぎ/ソース失敗。音声キャッシュ無し=常に除外)</h3>
  <div class="tscroll"><table><thead><tr><th>理由</th><th>分類</th><th class=num>件数</th></tr></thead>
    <tbody>{skip_rows or '<tr><td colspan=3>(なし)</td></tr>'}</tbody></table></div>
  <p class="muted">ピークリミッタ作動: <b>{limited}</b> 本。作動した本数は納品音声のピークを -1dBFS に
  下げるため、LUFS ターゲットが崩れている (<code>limited</code> / <code>lufs_out</code> は manifest に残る)</p>
</section>

<section id="yomi"><h2>読めない語 (学習側読みチェック YomiError 事前打診)</h2>
  <p class="lead">学習側 preprocess と同じ条件で「読めない」語を先に叩いた一覧。
  <code>on_unknown_tokens = "exclude"</code> の間はこれを含む発話を落とすので、<b>ユーザー辞書に登録して
  救うかそのまま捨てるかの判断材料</b>にする (登録して直したら rescore で戻る)。</p>
  <div class="tscroll"><table><thead><tr><th>語</th><th class=num>出現</th><th></th></tr></thead>
    <tbody>{yomi_rows or '<tr><td colspan=3>(なし)</td></tr>'}</tbody></table></div>
</section>
</main>
<footer>report.html は Stage 6 が機械生成。実行メタ情報 (生成日時・使用閾値・ライブラリ版・commit) は
<code>run.json</code>、レコード単位の全スコアは <code>manifest.jsonl</code>。</footer>
<script>{_JS}</script>
</body></html>
"""
    (outdir / "report.html").write_text(doc, encoding="utf-8")


def write_run_meta(outdir: Path, cfg: Config, mode: str) -> None:
    """再現性メタデータ (使用した閾値・ASR モデル・ライブラリ版・commit hash) を run.json に残す."""
    import subprocess

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or "unknown"
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ):  # git 不在 PATH / timeout 等は commit 不明のまま続行
        commit = "unknown"
    libs = {}
    from importlib.metadata import PackageNotFoundError, version

    for name in [
        "numpy",
        "scipy",
        "soundfile",
        "torch",
        "torchaudio",
        "faster-whisper",
        "silero-vad",
        "pyopenjtalk-plus",
        "praat-parselmouth",
        "pyloudnorm",
        "transformers",
        "pyannote.audio",  # 任意依存。未導入でも run.json は書く (unknown)
        "deepfilternet",  # 任意依存
    ]:
        try:
            libs[name] = version(name)
        except PackageNotFoundError:  # 任意依存が未導入でも run.json は書く
            libs[name] = "unknown"
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,  # run / rescore
        "commit": commit,
        "python": sys.version.split()[0],
        "libs": libs,
        "config": cfg.data,  # 使用した閾値・ASR モデル・挙動設定のスナップショット
    }
    (outdir / "run.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
