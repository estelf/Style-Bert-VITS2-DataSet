"""Stage 6: 出力. 閾値で adopted/excluded を確定し、esd.list + raw.zip + manifest.jsonl + report.html を生成する.

- E4: 失格理由は最初の 1 個ではなく全部を dropped_by に残す (感度分析と説明可能性の基盤)。
- E3: g2p / unknown_tokens の扱い (除外 or ユーザー辞書候補) を filters.toml の on_g2p_fail /
  on_unknown_tokens で決める。ハードコードゲートをなくす。
- A1/A2: 非有限値や欠損スコアは「検証不能=不合格」。x * -inf で nan を納品しない。
- A3: 納品用 raw/ は毎回作り直す。中間音声は .cache/ (E1 の rescore 用) と分離。
- G1: 閾値感度分析表 (この閾値を緩めれば +N 発話) をレポートに出す。
"""

from __future__ import annotations

import contextlib
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

# (閾値キー, スコアキー, 方式)。方式: min=スコアが閾値未満で不合格 / max=超過で不合格 /
# max_excl=閾値以上で不合格 (「12秒未満」のような排他的上限) / flag=True で不合格。
# 閾値キーが filters.toml に無ければチェック自体が走る (E2)。スコアが None は「検証不能=不合格」(A2)。
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
    ("max_clip_ratio", "clip_ratio", "max"),  # D2
    ("max_dc_offset", "dc_offset_abs", "max"),  # D2
    ("max_no_speech_prob", "no_speech_prob", "max"),  # D1
    ("max_compression_ratio", "compression_ratio", "max"),  # D1
    ("reject_repetition", "repeat_block", "flag"),  # D1
    ("reject_alpha_only", "alpha_only", "flag"),  # E4: filler_ratio とは別理由で記録する
    ("reject_music", "music_flag", "flag"),  # D5
]


def _check_failed(mode: str, value, thresh) -> bool:
    if mode == "flag":
        return bool(value)
    if value is None:
        return True  # A2: 閾値が有効なのにスコアが無い/非有限 → 検証不能なので不合格 (異常値ほど通るを禁止)
    if mode == "max_excl":
        return value >= thresh
    return value < thresh if mode == "min" else value > thresh


def evaluate_checks(records: list[Record], cfg: Config) -> None:
    """全スコアはここまで「記録」止まり。ここで初めて閾値で各チェックの合否を判定する (E4: 全理由)."""
    t = cfg.thresholds
    for r in records:
        s = r.scores
        r.review_reason = None
        fails: list[str] = []
        if s.get("errors"):
            fails.append("pipeline_error")
        for tkey, skey, mode in CHECKS:  # 発話長 (seg_dur) も含め全て閾値テーブル駆動 (E2)
            if tkey not in t or t[tkey] is None:
                continue  # E2: キーが無い = チェック自体が走らない
            if mode == "flag" and not t[tkey]:
                continue
            if _check_failed(mode, s.get(skey), t[tkey]) and skey not in fails:
                fails.append(skey)  # 下限・上限の両方に落ちても理由の重複は 1 件に寄せる
        # 話速レンジ (モーラ数が取れない発話はスキップ)。キーがあっても None なら片側開放
        lo_t, hi_t = t.get("min_mora_rate"), t.get("max_mora_rate")
        if (lo_t is not None or hi_t is not None) and s.get("mora_count", 0):
            rate = s.get("mora_rate") or 0
            lo = lo_t if lo_t is not None else -1e9
            hi = hi_t if hi_t is not None else 1e9
            if not (lo <= rate <= hi):
                fails.append("mora_rate")
        # E3: G2P 失敗・unknown トークンの扱いを filters.toml で決める
        act = t.get("on_g2p_fail")
        if act is not None and not s.get("g2p_ok", True):
            if act == "exclude":
                fails.append("g2p")
            else:
                r.review_reason = f"g2p:{act}"
        act = t.get("on_unknown_tokens")
        if act is not None and s.get("unknown_tokens", 0) > 0:
            if act == "exclude":
                fails.append("unknown_tokens")
            else:
                r.review_reason = "dict_data"  # DESIGN「ユーザー辞書候補への振り分け」(E6 の受け皿)
        s["failed_checks"] = fails


def finalize_statuses(records: list[Record]) -> None:
    """dedup 結果とチェック結果を合わせて status / dropped_by (全理由) を確定する."""
    for r in records:
        reasons = list(r.scores.get("failed_checks") or [])
        if r.dup_reason:
            reasons.append(f"dedup:{r.dup_reason}")
        if reasons:
            r.status, r.dropped_by = "excluded", reasons
        elif r.review_reason:
            r.status = "pending_review"
        else:
            r.status, r.dropped_by = "adopted", None


def write_outputs(
    adopted: list[Record], cache_dir: Path, outdir: Path, speaker: str, cfg: Config
) -> Path:
    """esd.list + raw.zip (44.1k/mono・LUFS正規化済み flac) を書き、契約検証まで通す."""
    import pyloudnorm as pyln

    from .stage0_ingest import load_cached

    sr = cfg.get("output.sample_rate")
    meter = pyln.Meter(sr)
    target = cfg.get("output.target_lufs")
    raw = outdir / "raw"
    if raw.exists():  # A3: 前回の除外済みファイルが混ざらないよう納品用 raw/ は毎回作り直す
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    esd_lines = []
    for r in adopted:
        x = np.asarray(load_cached(r, cache_dir, sr), dtype=np.float32)
        if not np.all(
            np.isfinite(x)
        ):  # A1: nan/inf を納品しない (errors 側でも除外されるが二重防御)
            x = np.nan_to_num(x)
        if len(x) >= meter.block_size * sr:  # C4: 1秒ゲートではなくメータのブロック長基準
            lufs = meter.integrated_loudness(x)
            if np.isfinite(lufs):  # A1: -inf を掛けて nan を撒かない
                with contextlib.suppress(
                    ValueError
                ):  # 正規化不可のクリップはそのまま納品 (直後の peak で守る)
                    x = pyln.normalize.loudness(x, lufs, target)  # (data, 実測LUFS, 目標LUFS)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if np.isfinite(peak) and peak > 10 ** (-1 / 20):  # A1: nan > 0.891 == False で素通りしない
            x = x * (10 ** (-1 / 20) / peak)
            r.scores["limited"] = True  # G3: リミッタが LUFS ターゲットを崩したことを黙らせない
        if len(x) >= meter.block_size * sr:
            lo = meter.integrated_loudness(x)
            r.scores["lufs_out"] = round(float(lo), 1) if np.isfinite(lo) else None
        fname = f"{r.utt_id}.flac"
        sf.write(raw / fname, x.astype(np.float32), sr, format="FLAC")
        esd_lines.append(f"{fname}|{speaker}|{r.lang}|{r.text}")

    (outdir / "esd.list").write_text(
        "\n".join(esd_lines) + ("\n" if esd_lines else ""),
        encoding="utf-8",  # A6: 0 件なら空ファイル
    )
    with zipfile.ZipFile(outdir / "raw.zip", "w", zipfile.ZIP_STORED) as z:
        for p in sorted(raw.rglob("*.flac")):
            z.write(p, p.relative_to(raw))
    _contract_check(outdir / "raw.zip")
    return outdir


def _contract_check(zip_path: Path) -> None:
    """納品前に学習側 preprocess と同じ検証 (44.1kHz/モノラル等) を全ファイルへ通す (A7)."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            info = sf.info(io.BytesIO(z.read(name)))
            if info.samplerate != 44100:
                raise ValueError(f"contract violation: {name} sr={info.samplerate}")
            if info.channels != 1:
                raise ValueError(f"contract violation: {name} not mono ({info.channels}ch)")


def _sensitivity_rows(records: list[Record], cfg: Config) -> str:
    """G1: 「この閾値だけ X% 緩めれば +N 発話」表。単独失格のレコードだけを数える近似."""
    t = cfg.thresholds
    excluded = [r for r in records if r.status == "excluded"]
    adopted = [r for r in records if r.status == "adopted"]
    rows = []
    for tkey, skey, mode in CHECKS:
        v = t.get(tkey)
        if mode == "flag" or v is None:
            continue
        relaxed = v * (0.8 if mode == "min" else 1.25)
        tightened = v * (1.2 if mode == "min" else 0.8)
        gain = sum(
            1
            for r in excluded
            if r.scores.get("failed_checks") == [skey]
            and r.scores.get(skey) is not None
            and (r.scores[skey] >= relaxed if mode == "min" else r.scores[skey] <= relaxed)
        )
        loss = sum(
            1
            for r in adopted
            if r.scores.get(skey) is not None
            and (r.scores[skey] < tightened if mode == "min" else r.scores[skey] > tightened)
        )
        rows.append(
            f"<tr><td>{tkey}</td><td>{v}</td><td>{relaxed:.4g}</td><td>+{gain}</td><td>{tightened:.4g}</td><td>-{loss}</td></tr>"
        )
    return "".join(rows)


def write_report(records: list[Record], outdir: Path, cfg: Config) -> None:
    """除外内訳・閾値感度・カバレッジ (音素/モーラ/F0) の可視化."""
    from .stage5_dedup import phoneme_tokens

    dropped = Counter(k for r in records if r.status == "excluded" for k in (r.dropped_by or []))
    adopted = [r for r in records if r.status == "adopted"]
    pending = [r for r in records if r.status == "pending_review"]
    dup_groups = {r.dup_group_id for r in records if r.dup_group_id}
    # F0 が計測できない発話 (None) はレンジ計算から除外する (f0_median_hz は 0 にならないので is not None で十分)
    f0s = [v for r in adopted if (v := r.scores.get("f0_median_hz")) is not None]
    limited = sum(1 for r in adopted if r.scores.get("limited"))
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in dropped.most_common())
    srcs = Counter(r.src for r in records)
    srclist = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in srcs.most_common())

    # G2: 音素被覆率 (adopted 全文に出現した音素トークン) とモーラ数分布ヒストグラム
    tokens: Counter = Counter()
    for r in adopted:
        tokens.update(set(phoneme_tokens(r.text)))
    rare = ", ".join(f"{k}({v})" for k, v in sorted(tokens.items(), key=lambda kv: kv[1])[:8])
    moras = [r.scores.get("mora_count") or 0 for r in adopted]
    bins = [0] * 10
    for m in moras:
        bins[min(int(m // 5), 9)] += 1
    hist = "".join(
        f"{i*5:2d}-{i*5+4:2d}モーラ {'█' * int(c / max(bins) * 40) if max(bins) else ''} {c}\n"
        for i, c in enumerate(bins)
    )
    pend_rows = "".join(
        f"<tr><td>{r.utt_id}</td><td>{r.review_reason}</td><td>{r.text}</td></tr>" for r in pending
    )

    html = f"""<!doctype html><meta charset="utf-8"><title>dataset report</title>
<h1>Dataset Report</h1><p>adopted={len(adopted)} / pending_review={len(pending)} / total={len(records)}</p>
<h2>除外内訳 (全失格理由, E4)</h2><table border=1>{rows or '<tr><td>(なし)</td></tr>'}</table>
<h2>閾値感度分析 (G1: その閾値単独の失格がひっくり返る件数)</h2>
<table border=1><tr><th>閾値</th><th>現在</th><th>緩和後</th><th>採用+</th><th>厳格化後</th><th>採用-</th></tr>{_sensitivity_rows(records, cfg)}</table>
<h2>投入ソース別件数</h2><table border=1>{srclist}</table>
<h2>重複排除</h2><p>{len(dup_groups)} グループ (代表1本採用)</p>
<h2>ピークリミッタ作動 (G3: LUFS ターゲットから外れた本数)</h2><p>{limited} 本</p>
<h2>人手レビュー待ち (pending_review: ユーザー辞書候補等)</h2><table border=1>{pend_rows or '<tr><td>(なし)</td></tr>'}</table>
<h2>F0中央値レンジ (adopted)</h2><p>{min(f0s) if f0s else '-'} – {max(f0s) if f0s else '-'} Hz / 中央値 {np.median(f0s) if f0s else '-'} Hz</p>
<h2>音素被覆 (G2)</h2><p>出現音素 {len(tokens)} 種 / 希少音素: {rare or '-'}</p>
<h2>モーラ数分布 (G2)</h2><pre>{hist}</pre>"""
    (outdir / "report.html").write_text(html, encoding="utf-8")


def write_run_meta(outdir: Path, cfg: Config, mode: str) -> None:
    """E5: 再現性メタデータ (使用した閾値・ASR モデル・ライブラリ版・commit hash) を run.json に残す."""
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
