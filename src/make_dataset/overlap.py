"""2 人同時発話の検出 (pyannote segmentation-3.0)。

Stage 2 は overlap_flag を常に None (未接続) で記録してきた。pyannote.audio が導入されていれば
powerset セグメンテーション (フレーム毎 [非話声, 話者1, 話者2, 重複] の 4 クラス) から
重複区間の合計秒を測って overlap_flag / overlap_sec を上書きする。

pyannote.audio は任意依存 (uv sync --extra extras) で、モデルダウンロードに HF_TOKEN が必要
(モデル側の利用規約同意も)。未導入のまま [overlap] enabled = true にした場合は
黙って素通りせず、明示的なエラーで導入手順を告げる。
"""

from __future__ import annotations

import os


def compute_overlap(records, cfg, cache_dir) -> None:
    """各 Record の overlap_flag / overlap_sec を計算して上書きする (有効化時のみ呼ばれる)。"""
    try:
        import numpy as np
        import torch
        from pyannote.audio import Model  # pyright: ignore[reportMissingImports]
        from pyannote.audio.inference import (  # pyright: ignore[reportMissingImports]
            SlidingWindowInferer,
        )
    except ImportError as e:  # 明示的な有効化に対する fail-fast (黙って None のままにしない)
        raise ValueError(
            "[overlap] enabled = true だが pyannote.audio が導入されていない。"
            " `uv sync --extra extras` で導入し、モデルダウンロード用の HF_TOKEN"
            " (環境変数または [overlap] token、pyannote モデルの利用規約同意付き) を設定してください"
        ) from e

    from .stage0_ingest import load_cached

    token = (
        cfg.get("overlap.token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    try:
        model = Model.from_pretrained(cfg.get("overlap.model"), token=token)  # pyannote.audio 4.x
    except TypeError:
        model = Model.from_pretrained(
            cfg.get("overlap.model"), use_auth_token=token
        )  # 3.x の引数名
    model = model.eval()
    inferer = SlidingWindowInferer(duration=5.0, hop_size=0.5, min_duration_on=0.0)
    min_sec = float(cfg.get("overlap.min_overlap_sec"))

    for r in records:
        try:  # 1 本の失敗で全体を止めない。失敗は overlap_errors に寄せる (overlap_flag は None のまま)
            x = np.asarray(load_cached(r, cache_dir, 16000), dtype=np.float32)
            data = {"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": 16000}
            with torch.inference_mode():
                frames = inferer(model, data)  # SlidingWindowFeature (T, 4) powerset
            data_arr = np.asarray(frames.data)
            if data_arr.ndim != 2 or data_arr.shape[1] < 4:
                r.scores.setdefault("overlap_errors", []).append("unexpected_powerset_shape")
                continue
            overlap_sec = float(np.sum(data_arr[:, 3] > 0.5)) * float(frames.sliding_window.step)
            r.scores["overlap_sec"] = round(overlap_sec, 2)
            r.scores["overlap_flag"] = bool(overlap_sec >= min_sec)
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("overlap_errors", []).append(repr(e))
