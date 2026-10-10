"""Stage 4: 音声品質スコア. SQUIM 系客观品質 → quality_score に集約 (UTMOS2 はオプション/今後の差し替え)."""

from __future__ import annotations

from tqdm import tqdm


def score_quality(records, cfg, cache_dir) -> None:
    import torch
    import torchaudio

    from .config import resolve_device
    from .stage0_ingest import load_cached

    device = resolve_device(cfg)  # GPU があれば SQUIM も CUDA で回す (docs/GPU.md)
    model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
    model.eval().to(device)
    for r in tqdm(records, desc="stage4 quality", unit="utt"):
        try:  # B2: 1 本の失敗でパイプラインを止めない
            x = load_cached(r, cache_dir, 16000)
            x = x[
                : len(x) // 400 * 400
            ]  # SQUIM の 25ms フレームに切り揃える (端数で落ちるのを防ぐ)
            if len(x) < 400:
                r.scores.setdefault("errors", []).append("stage4: segment too short for SQUIM")
                continue
            with torch.inference_mode():
                stoi, pesq, sdr = model.forward(
                    torch.from_numpy(x).unsqueeze(0).to(device)
                )  # (B,) を 3 つ返す
            r.scores.update(
                {
                    "stoi": round(float(stoi[0]), 3),
                    "pesq": round(float(pesq[0]), 2),
                    "si_sdr": round(float(sdr[0]), 1),
                    "quality_score": round(
                        float(pesq[0]), 3
                    ),  # UTMOS2 差し替え時に置換される集約先
                }
            )
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("errors", []).append(f"stage4: {e!r}")
