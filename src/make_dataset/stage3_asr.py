"""Stage 3: 文字起こしと言語系チェック (ASR + G2P 整合 3段構え).

- ASR バックエンドは `filters.toml [asr] backend` で差し替え可能:
    backend="qwen3-asr" (既定, Qwen/Qwen3-ASR-1.7B-hf を transformers native で実行) /
    backend="faster-whisper" (CPU で軽い代替, model は Whisper サイズ名)
  model にはそれぞれ HF リポジトリID / Whisper サイズ名 を入れる。
- G2P 整合①: pyopenjtalk-plus (学習側と同一エンジン) が正しく処理できるか・unknown トークン検出
- G2P 整合②: モーラ数 ÷ 実音声長 = 話速が話者中央値から乖離ならフラグ (2文混入・ASR欠落・ガチャ検出)
- G2P 整合③: アライメント尤度 → align_conf として記録
- D1: ハルシネーション検出 — no_speech_prob / compression_ratio / 同一文言の繰り返し (ブロック) を記録し、
  Stage 6 の閾値で弾く (不良データ混入量の最大要因)。
- D5: ♪ 等の音楽記号は「音楽だと判定した」強いシグナルなので除去せず music_flag として残す。
- A4: NFKC は全角｜を半角 | に変換して esd.list の行フォーマットを壊すため、| を ・ に置換する。
- C7: mora_rate の分母は実音声長 (Stage 1 が無音パディングなしで切るため end-start がそのまま音声長)。
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from tqdm import tqdm

from .config import Config

FILLERS = ["あの", "まあ", "えーと", "えっと", "なんか", "やっぱり", "まぁ"]
# 装飾記号等はセリフから除去 (学習側 dict_data 連携での ITN 正規化は今後の課題)
_STRIP_CHARS = "♪♡♥★☆※―…"
_MUSIC_CHARS = "♪♫"  # D5: 除去する前に検出して music_flag に残す
# D1: 同一文字列の 3 回以上の連続 = whisper ハルシネーションの典型 (「ありがとう」無限ループ等)
_REPEAT_RE = re.compile(r"(.{3,})\1{2,}")


def normalize_text(text: str) -> str:
    """全角/半角正規化: NFKC + esd.list 区切り保護 + 装飾記号除去 + 空白圧縮."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(
        "|", "・"
    )  # A4: NFKC が全角｜を半角に変換するので、区切り文字は安全な記号へ
    text = "".join(c for c in text if c not in _STRIP_CHARS)
    return re.sub(r"\s+", " ", text).strip()


def g2p_check(text: str) -> dict:
    """pyopenjtalk-plus が通るか・モーラ数を返す (学習側で raise しないことの保証)."""
    import pyopenjtalk as pj  # pyopenjtalk-plus (学習側と同一エンジン)

    out = {"g2p_ok": True, "unknown_tokens": 0, "mora_count": 0}
    try:
        phrases = pj.run_frontend(text)
        out["mora_count"] = sum(f["mora_size"] for f in phrases)
        out["unknown_tokens"] = sum(
            "*" in str(f.get("pron", "")) or "?" in str(f.get("pron", "")) for f in phrases
        )
        if out["mora_count"] == 0:
            out["g2p_ok"] = False
    except Exception:  # noqa: BLE001
        # pyopenjtalk-plus は入力により多种の例外を投げうるので、G2P 失敗は全て g2p_ok=False に寄せる
        out["g2p_ok"] = False
    return out


# ---------------------------------------------------------------- ASR バックエンド

_WHISPER = {}


def _asr_whisper(audio16k: np.ndarray, cfg: Config) -> tuple[str, float, dict]:
    """faster-whisper で文字起こし → (テキスト, 信頼度, 追加スコア). word probability は align_conf 用."""
    from faster_whisper import WhisperModel

    from .config import resolve_device

    device = cfg.get("asr.device")
    if device == "auto":
        device = resolve_device(cfg)
    compute_type = cfg.get("asr.compute_type")
    if device.startswith("cuda") and compute_type == "int8":
        compute_type = "float16"  # int8 は CPU 量子化。GPU では float16 が速い (docs/GPU.md)
    key = (cfg.get("asr.model"), device, compute_type)
    if key not in _WHISPER:
        _WHISPER[key] = WhisperModel(key[0], device=key[1], compute_type=key[2])
    model = _WHISPER[key]
    import inspect

    kwargs = {}  # F5: batch_size 対応版の faster-whisper なら渡す (古い版では内部バッチに任せる)
    if "batch_size" in inspect.signature(model.transcribe).parameters:
        kwargs["batch_size"] = cfg.get("asr.batch_size")
    segs, _info = model.transcribe(
        audio16k,
        language=cfg.get("asr.language"),
        beam_size=cfg.get("asr.beam_size"),
        word_timestamps=True,
        condition_on_previous_text=False,
        **kwargs,
    )
    texts, confs, word_probs = [], [], []
    nsp, cr = 0.0, 0.0
    for seg in segs:
        texts.append(seg.text.strip())
        confs.append(float((1.0 - seg.no_speech_prob) * np.exp(min(0.0, seg.avg_logprob))))
        nsp = max(nsp, float(seg.no_speech_prob))  # D1: 「無声だと言う確率」
        cr = max(cr, float(seg.compression_ratio))  # D1: 圧縮率 = 繰り返しガチャの代表指標
        word_probs.extend(
            float(w.probability) for w in (seg.words or [])
        )  # B3: words が None でも落ちない
    text = "".join(texts)
    extra = {
        "align_conf": round(float(np.mean(word_probs)), 3) if word_probs else 0.0,
        "no_speech_prob": round(nsp, 3),
        "compression_ratio": round(cr, 2),
    }
    return text, (float(np.mean(confs)) if confs else 0.0), extra


_QWEN = {}


def _asr_qwen(audio16k: np.ndarray, cfg: Config) -> tuple[str, float, dict]:
    """Qwen3-ASR (既定バックエンド, transformers native). greedy generate のトークン平均確率を信頼度にする.

    出力は "language <LANG><asr_text>..." 形式なので decode(return_format="transcription_only") で剥がす。
    no_speech_prob / compression_ratio は whisper 固有のハルシネーション指標なのでここでは記録しない
    (filters.toml の該当閾値は qwen3-asr ではコメントアウト済み。有効にすると欠損スコアで全件落ちる)。
    """
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from .config import resolve_device

    repo, device = cfg.get("asr.model"), resolve_device(cfg)
    if (repo, device) not in _QWEN:
        model = AutoModelForMultimodalLM.from_pretrained(repo, dtype=torch.bfloat16).to(device)
        _QWEN[(repo, device)] = (model.eval(), AutoProcessor.from_pretrained(repo))
    model, processor = _QWEN[(repo, device)]

    inputs = processor.apply_transcription_request(
        audio=audio16k, language=cfg.get("asr.language") or None  # numpy 16k を直接渡せる
    ).to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max(64, int(len(audio16k) / 16000 * 40) + 32),
        )
    gen = out.sequences[:, inputs["input_ids"].shape[1] :]
    probs = []
    for step, logits in enumerate(out.scores):
        lg = logits[:, -1] if logits.ndim > 2 else logits  # (B, V) を得る
        lp = torch.log_softmax(lg.float(), dim=-1)
        tok = gen[:, step].clamp(max=lp.shape[-1] - 1)
        probs.append(float(lp.gather(-1, tok[:, None]).exp().mean().item()))
    text = processor.decode(gen[0], return_format="transcription_only")
    conf = float(np.mean(probs)) if probs else 0.0
    return text, conf, {"align_conf": round(conf, 3)}  # トークン確率をアライメント信頼度に流用


def transcribe_segments(records, cfg: Config, cache_dir) -> None:
    """各 Record に ASR テキスト・信頼度・テキストフィルタ用スコアを書き込む."""
    from .stage0_ingest import load_cached

    backend = cfg.get("asr.backend")
    if backend not in {"faster-whisper", "qwen3-asr"}:
        raise ValueError(
            f"unknown asr backend: {backend!r} (使用可: 'qwen3-asr' / 'faster-whisper')"
        )
    fn = {"faster-whisper": _asr_whisper, "qwen3-asr": _asr_qwen}[backend]

    for r in tqdm(records, desc="stage3 asr", unit="utt"):
        try:  # B2: 1 本の異常セグメントでラン全体を止めない
            audio16k = load_cached(r, cache_dir, 16000)
            raw_text, conf, extra = fn(audio16k, cfg)
            text = normalize_text(raw_text)
            r.text = text
            g = (
                g2p_check(text)
                if text
                else {"g2p_ok": False, "unknown_tokens": -1, "mora_count": 0}
            )
            dur = r.end - r.start  # C7: Stage 1 が無音パディングなしで切るため、これが実音声長
            sc = r.scores
            sc["asr_conf"] = round(conf, 3)
            sc.update(extra)
            sc.update(g)
            sc["mora_rate"] = (
                round(g["mora_count"] / dur, 2) if g["mora_count"] and dur > 0 else 0.0
            )
            # D1: 繰り返しブロック (whisper ハルシネーション) をブールで記録
            sc["repeat_block"] = bool(_REPEAT_RE.search(raw_text))
            # D5: 音楽記号は「音楽だと判定した」シグナルとして残す (除去だけして黙殺しない)
            sc["music_flag"] = any(c in raw_text for c in _MUSIC_CHARS)
            # テキストフィルタ用スコア (Stage 6 の閾値で棄却可否が決まる)
            kana = sum(1 for c in text if "\u3041" <= c <= "\u3096" or "\u30a1" <= c <= "\u30fa")
            alpha = sum(1 for c in text if c.isascii() and c.isalpha())
            filler_chars = sum(len(f) for f in FILLERS if f in text)
            sc["char_count"] = len(text)
            sc["kana_ratio"] = round(kana / max(len(text), 1), 2)
            sc["alpha_only"] = bool(alpha and kana == 0 and text)
            sc["filler_ratio"] = round(filler_chars / max(len(text), 1), 2)
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("errors", []).append(f"stage3: {e!r}")


def flag_mora_rate_outliers(records) -> None:
    """G2P 整合②: 話速 (モーラ数/秒) が dataset 話者中央値から乖離なら mora_rate_dev に記録."""
    rates = [r.scores["mora_rate"] for r in records if r.scores.get("mora_count", -1) > 0]
    med = float(np.median(rates)) if rates else 0.0
    for r in records:
        v = r.scores.get("mora_rate", 0.0)
        r.scores["mora_rate_median"] = round(med, 2)
        r.scores["mora_rate_dev"] = round(abs(v - med) / med, 2) if med and v else 0.0
