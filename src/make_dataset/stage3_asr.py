"""Stage 3: 文字起こしと言語系チェック (ASR + G2P 整合 3段構え).

- ASR バックエンドは `filters.toml [asr] backend` で差し替え可能:
    backend="faster-whisper" (既定, CPU で速い) / "qwen3-asr" (Qwen/Qwen3-ASR-*.hf を transformers native で実行)
  model にはそれぞれ Whisper サイズ名 / HF リポジトリID を入れる。
- G2P 整合①: pyopenjtalk-plus (学習側と同一エンジン) が正しく処理できるか・unknown トークン検出
- G2P 整合②: モーラ数 ÷ 実音声長 = 話速が話者中央値から乖離ならフラグ (2文混入・ASR欠落・ガチャ検出)
- G2P 整合③: アライメント尤度 → align_conf として記録
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from .config import Config

FILLERS = ["あの", "まあ", "えーと", "えっと", "なんか", "やっぱり", "まぁ"]
# 装飾記号等はセリフから除去 (学習側 dict_data 連携での ITN 正規化は今後の課題)
_STRIP_CHARS = "♪♡♥★☆※―…"


def normalize_text(text: str) -> str:
    """全角/半角正規化ドラフト: NFKC + 装飾記号除去 + 空白圧縮."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c not in _STRIP_CHARS)
    return re.sub(r"\s+", " ", text).strip()


def g2p_check(text: str) -> dict:
    """pyopenjtalk-plus が通るか・モーラ数を返す (学習側で raise しないことの保証)."""
    import pyopenjtalk as pj  # pyopenjtalk-plus (学習側と同一エンジン)

    out = {"g2p_ok": True, "unknown_tokens": 0, "mora_count": 0}
    try:
        phrases = pj.run_frontend(text)
        out["mora_count"] = sum(f["mora_size"] for f in phrases)
        out["unknown_tokens"] = sum("*" in str(f.get("pron", "")) or "?" in str(f.get("pron", "")) for f in phrases)
        if out["mora_count"] == 0:
            out["g2p_ok"] = False
    except Exception:
        out["g2p_ok"] = False
    return out


# ---------------------------------------------------------------- ASR バックエンド

_WHISPER = {}


def _asr_whisper(audio16k: np.ndarray, cfg: Config) -> tuple[str, float]:
    """faster-whisper で文字起こし → (テキスト, 信頼度). word probability は align_conf 用に残す."""
    from faster_whisper import WhisperModel

    key = (cfg.get("asr.model"), cfg.get("asr.device"), cfg.get("asr.compute_type"))
    if key not in _WHISPER:
        _WHISPER[key] = WhisperModel(key[0], device=key[1], compute_type=key[2])
    model = _WHISPER[key]
    segs, _info = model.transcribe(
        audio16k,
        language=cfg.get("asr.language"),
        beam_size=cfg.get("asr.beam_size"),
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    texts, confs = [], []
    for seg in segs:
        texts.append(seg.text.strip())
        confs.append(float((1.0 - seg.no_speech_prob) * np.exp(min(0.0, seg.avg_logprob))))
        _WHISPER.setdefault("_word_probs", []).extend(float(w.probability) for w in seg.words)
    text = "".join(texts)
    return text, (float(np.mean(confs)) if confs else 0.0)


_QWEN = {}


def _asr_qwen(audio16k: np.ndarray, cfg: Config) -> tuple[str, float]:
    """Qwen3-ASR (transformers native). greedy generate のトークン平均確率を信頼度として返す."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    repo = cfg.get("asr.model")
    if "model" not in _QWEN:
        _QWEN["model"] = AutoModelForMultimodalLM.from_pretrained(repo, dtype=torch.bfloat16)
        _QWEN["model"].eval()
        _QWEN["processor"] = AutoProcessor.from_pretrained(repo)
    model, processor = _QWEN["model"], _QWEN["processor"]

    import soundfile as sf

    tmp = __import__("tempfile").NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio16k, 16000)
    try:
        inputs = processor.apply_transcription_request(
            audio=tmp.name, language=cfg.get("asr.language")
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            out = model.generate(
                **inputs, do_sample=False, output_scores=True, return_dict_in_generate=True,
                max_new_tokens=max(64, int(len(audio16k) / 16000 * 40) + 32),
            )
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        probs = []
        for step, logits in enumerate(out.scores):
            lg = logits[:, -1] if logits.ndim > 2 else logits  # (B, V) を得る
            lp = torch.log_softmax(lg.float(), dim=-1)
            tok = gen[:, step].clamp(max=lp.shape[-1] - 1)
            probs.append(float(lp.gather(-1, tok[:, None]).exp().item()))
        text = processor.decode(gen, return_format="transcription_only")[0]
    finally:
        __import__("os").unlink(tmp.name)
    return text, float(np.mean(probs)) if probs else 0.0


def transcribe_segments(segments, cfg: Config) -> None:
    """各 Segment に ASR テキスト・信頼度・テキストフィルタ用スコアを書き込む."""
    backend = cfg.get("asr.backend")
    fn = {"faster-whisper": _asr_whisper, "qwen3-asr": _asr_qwen}[backend]

    for s in segments:
        raw_text, conf = fn(s.audio16k, cfg)
        if backend == "faster-whisper":  # word probability 版 align_conf (whisper のみ)
            word_probs = _WHISPER.get("_word_probs", [])
            sc0 = s.record.scores
            sc0["align_conf"] = round(float(np.mean(word_probs)), 3) if word_probs else 0.0
            _WHISPER["_word_probs"] = []
        text = normalize_text(raw_text)
        s.record.text = text
        g = g2p_check(text) if text else {"g2p_ok": False, "unknown_tokens": -1, "mora_count": 0}
        dur = s.record.end - s.record.start
        sc = s.record.scores
        sc["asr_conf"] = round(conf, 3)
        if backend != "faster-whisper":  # qwen3-asr はトークン確率をそのまま align_conf にも流用 (差し替え余地)
            sc["align_conf"] = round(conf, 3)
        sc.update(g)
        sc["mora_rate"] = round(g["mora_count"] / dur, 2) if g["mora_count"] else 0.0
        # テキストフィルタ用スコア (Stage 6 の閾値で棄却可否が決まる)
        kana = sum(1 for c in text if "\u3041" <= c <= "\u3096" or "\u30a1" <= c <= "\u30fa")
        alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        filler_chars = sum(len(f) for f in FILLERS if f in text)
        sc["char_count"] = len(text)
        sc["kana_ratio"] = round(kana / max(len(text), 1), 2)
        sc["alpha_only"] = bool(alpha and kana == 0 and text)
        sc["filler_ratio"] = round(filler_chars / max(len(text), 1), 2)


def flag_mora_rate_outliers(segments) -> None:
    """G2P 整合②: 話速 (モーラ数/秒) が dataset 話者中央値から乖離なら mora_rate_dev に記録."""
    rates = [s.record.scores["mora_rate"] for s in segments if s.record.scores.get("mora_count", -1) > 0]
    med = float(np.median(rates)) if rates else 0.0
    for s in segments:
        r = s.record.scores.get("mora_rate", 0.0)
        s.record.scores["mora_rate_median"] = round(med, 2)
        s.record.scores["mora_rate_dev"] = round(abs(r - med) / med, 2) if med and r else 0.0
