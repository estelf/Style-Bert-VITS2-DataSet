"""Stage 4: 音声品質スコア. SQUIM 系客观品質 → quality_score に集約.

- [stage4] subjective = true で SQUIM_SUBJECTIVE を mos として追加記録。PESQ (音質軸) とは別軸の
  「総合 MOS」で、UTMOS2 相当の役割を既存 torchaudio 依存で果たす (DNSMOS/NISQA を差し込む際も同じ
  mos ゲート枠を流用できる)。SQUIM_SUBJECTIVE は「クリーンな参照音声」をアンカーに取るため、
  単一話者コーパスでは SNR 最良の発話をアンカーにする → mos は**その話者の最クリーン発話に対する
  相対品質** (絶対 MOS ではない)。モデルは初回に約 360MB をダウンロード。
- [tags] enabled で AudioSet タグ付け (tags.py)。音楽・笑い・咳などを音声から直接検出し
  tag_hit に記録する (music_flag は ASR が ♪ を吐くかに依存していた)。
- 追加モデルの失敗は errors (常に不合格) ではなく mos_errors / tag_errors に寄せる:
  ゲートが無効な間に追加検証が落ちただけでレコードを道連れにしない。
"""

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
        try:  # 1 本の失敗でパイプラインを止めない
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

    if cfg.get("stage4.subjective"):  # 非侵入 MOS (WavLM-large ベース。重いのでオプトイン)
        _subjective_mos(records, cfg, cache_dir, device)
    if cfg.get("tags.enabled"):  # AudioSet タグ付け (AST)
        from .tags import tag_segments

        tag_segments(records, cfg, cache_dir)


def _subjective_mos(records, cfg, cache_dir, device) -> None:
    import numpy as np
    import torch
    import torchaudio

    from .stage0_ingest import load_cached

    model = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
    model.eval().to(device)
    # SQUIM_SUBJECTIVE は「クリーンな参照音声」を第 2 引数に取る (同一発話の正解ではなく、
    # 校正アンカーとなる任意のクリーン発話でよい)。単一話者コーパスでは SNR 最良の発話を
    # アンカーに使う → mos は「その話者自身の最もクリーンな発話に対する相対品質」になる
    # (絶対 MOS ではない。min_mos を締める前に実測分布を見ること)。
    if not records:
        return
    ref_rec = max(records, key=lambda r: r.scores.get("snr") or -99.0)
    try:
        ref = (
            torch.from_numpy(np.asarray(load_cached(ref_rec, cache_dir, 16000), dtype=np.float32))
            .unsqueeze(0)
            .to(device)
        )
    except Exception as e:  # noqa: BLE001
        for r in records:
            r.scores.setdefault("mos_errors", []).append(f"anchor: {e!r}")
        return
    for r in tqdm(records, desc="stage4 mos", unit="utt"):
        try:
            x = load_cached(r, cache_dir, 16000)
            with torch.inference_mode():
                out = model.forward(torch.from_numpy(x).unsqueeze(0).to(device), ref)
            mos = out[0] if isinstance(out, tuple) else out  # 版差で tuple / Tensor の両方ある
            v = float(torch.as_tensor(mos).flatten()[0])
            r.scores["mos"] = round(v, 2) if np.isfinite(v) else None
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("mos_errors", []).append(repr(e))
