"""オーディオタグ付け (AudioSet)。音楽・笑い・咳などを「音声から直接」検出する。

既存の music_flag は ASR が ♪ を吐くかに依存して検出漏れが大きく、笑い声・咳・拍手などは
全く見ていなかった。AST (Audio Spectrogram Transformer) は transformers 既存依存で動かせるため
新規パッケージ追加なしに使える (モデルは初回に約 1.3GB をダウンロード)。
Stage 4 で記録し、Stage 6 の reject_audio_tags ゲートで弾く。
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

_MODEL: dict = {}


def _load(cfg):
    from .config import resolve_device

    repo, device = cfg.get("tags.model"), resolve_device(cfg)
    if (repo, device) not in _MODEL:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model = AutoModelForAudioClassification.from_pretrained(repo).eval().to(device)
        _MODEL[(repo, device)] = (model, AutoFeatureExtractor.from_pretrained(repo))
    return _MODEL[(repo, device)]


def _label(id2label, i: int) -> str:
    """id2label は int キー / str キーの両方あり (transformers の版差)。どちらでも引けるようにする。"""
    v = id2label.get(i, id2label.get(str(i)))
    return str(v) if v is not None else str(i)


def tag_segments(records, cfg, cache_dir) -> None:
    """各 Record に audio_tags (上位タグ) と tag_hit (reject リストに掛かった最有力タグ) を記録する。"""
    import torch

    from .stage0_ingest import load_cached

    min_prob = float(cfg.get("tags.min_prob"))
    reject = set(cfg.get("tags.reject") or [])
    model, fe = _load(cfg)
    id2label = model.config.id2label
    win = 10 * 16000  # AST の入力長 (10 秒超は窓ごとに max-pool)

    for r in tqdm(records, desc="stage4 tags", unit="utt"):
        try:  # 原則: 1 本の失敗でパイプラインを止めない (失敗は tag_errors に寄せる)
            x = np.asarray(load_cached(r, cache_dir, 16000), dtype=np.float32)
            wins = [x[i : i + win] for i in range(0, max(len(x), 1), win)]
            probs = []
            for w in wins:
                if len(w) < 1600:
                    continue
                inp = fe(w, sampling_rate=16000, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    logits = model(**inp).logits[0]
                probs.append(
                    torch.sigmoid(logits).float().cpu().numpy()
                )  # マルチラベルなので sigmoid
            if not probs:
                continue
            p = np.max(np.stack(probs), axis=0)  # 窓間の max-pool (一部分だけ音楽でも検出したい)
            ranked = sorted(
                ((float(p[i]), _label(id2label, i)) for i in range(len(p))), reverse=True
            )
            r.scores["audio_tags"] = [[lbl, round(v, 3)] for v, lbl in ranked[:5] if v >= min_prob]
            r.scores["tag_hit"] = next(
                (lbl for v, lbl in ranked if lbl in reject and v >= min_prob), None
            )
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("tag_errors", []).append(repr(e))
