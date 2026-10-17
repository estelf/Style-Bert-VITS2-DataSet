"""除去修復ルート (DeepFilterNet)。ノイズ系のゲートだけで落ちた発話を除去→再検証する。

設計上の約束:
- 修復音声は**自動採用しない**。再検証を通過しても必ず pending_review 行き (除去アーティファクトは
  学習後に効いてくるため、機械の「直った」を鵜呑みにしない)。
- 元音声は `.cache_orig/` に保持する (.cache が修復音声で上書きされても rescore / 手動復元できるように)。
- 修復対象の選定だけ、現在の閾値での evaluate_checks を先に走らせる (冪等。Stage 6 で再実行される)。
- emb (話者ベクトル) はコーパス全体の CMVN スケーリング済み値。score_segments を部分集合で再実行すると
  単一レコードの std で割って壊れるため、再スコア前にバックアップし再スコア後に復元する
  (話者本人性は除去で大きく変わらない前提。fp_sig 等の内容系は再計算が正しいのでそのまま採用)。

DeepFilterNet は任意依存だが uv lock 外 (numpy<2 pins のため)。導入は
`uv pip install --no-deps deepfilternet deepfilterlib loguru appdirs requests sympy`。
"""

from __future__ import annotations

from tqdm import tqdm


def repair_pass(records, cfg, cache_dir) -> int:
    """ノイズ系ゲート単独の失格レコードを除去→再スコアリングする。修復本数を返す。"""
    try:
        from df.enhance import enhance, init_df  # pyright: ignore[reportMissingImports]
    except ImportError as e:  # 明示的な有効化には fail-fast
        raise ValueError(
            "[repair] enabled = true だが deepfilternet が導入されていない。deepfilternet は numpy<2 を"
            " pins するため lock には入れられない。"
            " `uv pip install --no-deps deepfilternet deepfilterlib loguru appdirs requests sympy`"
            " で導入すること (--no-deps なしだとプロジェクトの numpy 2.x を downgrade して壊す)"
        ) from e

    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    from .stage2_score import score_segments
    from .stage4_quality import score_quality
    from .stage6_output import evaluate_checks

    evaluate_checks(
        records, cfg
    )  # 現在の閾値で「ノイズ系ゲート単独落ち」を選ぶ (Stage 6 で再実行され冪等)
    targets = set(cfg.get("repair.targets") or [])
    cand = [
        r
        for r in records
        if r.scores.get("failed_checks")
        and not r.scores.get("errors")
        and not r.scores.get("stage1_skipped")
        and set(r.scores["failed_checks"]) <= targets  # 対象外の失格理由が混ざっていたら修復しない
    ]
    if not cand:
        return 0

    try:
        model, df_state, _ = init_df()
    except Exception as e:  # numpy 2 との互差 (2023 年製パッケージ) は初期化時点で表面化する
        raise ValueError(
            f"[repair] DeepFilterNet の初期化に失敗: {e!r}。"
            " deepfilternet は numpy<2 前提のため numpy 2.x 環境では動作しない可能性がある。"
            " その場合は [repair] を無効に戻すか、専用 venv での実行を検討すること"
        ) from e
    sr_df = int(df_state.sr())  # DeepFilterNet の動作サンプルレート (既定 48k)
    orig_dir = cache_dir.parent / ".cache_orig"
    orig_dir.mkdir(parents=True, exist_ok=True)

    emb_backup = {r.utt_id: r.scores.get("emb") for r in cand}
    n = 0
    for r in tqdm(cand, desc="repair", unit="utt"):
        try:  # 1 本の失敗で全体を止めない (修復失敗はそのまま元の除外判定が残る)
            p = cache_dir / f"{r.utt_id}.flac"
            x, sr = sf.read(p, dtype="float32")
            if not (orig_dir / p.name).exists():
                sf.write(orig_dir / p.name, x, sr, format="FLAC")
            xr = librosa.resample(np.asarray(x), orig_sr=sr, target_sr=sr_df)
            enh = enhance(model, df_state, torch.from_numpy(xr).unsqueeze(0))
            back = librosa.resample(
                enh.squeeze(0).numpy().astype(np.float32), orig_sr=sr_df, target_sr=sr
            )
            sf.write(p, back.astype(np.float32), sr, format="FLAC")
            r.scores["repaired"] = True
            n += 1
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("repair_errors", []).append(repr(e))

    if n:
        done = [r for r in cand if r.scores.get("repaired")]
        score_segments(done, cfg, cache_dir)  # 修復音声で音響スコアを引き直す (fp_sig も含む)
        for r in done:  # 単一レコード std で割って壊れた emb をコーパス版へ復元 (docstring 参照)
            if emb_backup.get(r.utt_id) is not None:
                r.scores["emb"] = emb_backup[r.utt_id]
        score_quality(done, cfg, cache_dir)  # 修復音声で SQUIM 客观品質も引き直す
    return n
