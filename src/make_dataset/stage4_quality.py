"""Stage 4: 音声品質スコア. SQUIM 系客观品質 → quality_score に集約 (UTMOS2 はオプション/今後の差し替え)."""

from __future__ import annotations


def score_quality(segments) -> None:
    import torch
    import torchaudio

    model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
    model.eval()
    for s in segments:
        x = torch.from_numpy(s.audio16k).unsqueeze(0)
        with torch.inference_mode():
            stoi, pesq, sdr = model.forward(x)  # (B,) を 3 つ返す
        s.record.scores.update(
            {
                "stoi": round(float(stoi[0]), 3),
                "pesq": round(float(pesq[0]), 2),
                "si_sdr": round(float(sdr[0]), 1),
                "quality_score": round(float(pesq[0]), 3),  # UTMOS2 差し替え時に置換される集約先
            }
        )
