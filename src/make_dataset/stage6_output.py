"""Stage 6: 出力. 閾値で adopted/excluded を確定し、esd.list + raw.zip + manifest.jsonl + report.html を生成する."""

from __future__ import annotations

import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config
from .manifest import Record


def apply_thresholds(records: list[Record], cfg: Config) -> None:
    """全スコアはここまで「記録」止まり。ここで初めて閾値で採用/除外を決める (dropped_by 付き)."""
    t = cfg.data["thresholds"]

    def checks(r: Record):
        s = r.scores
        dur = r.end - r.start
        yield "seg_dur", dur < cfg.get("stage1.min_seg_sec") or dur > cfg.get("stage1.max_seg_sec")
        if t.get("min_snr_db") is not None:
            yield "snr", s.get("snr", 99) < t["min_snr_db"]
        if t.get("min_hf_energy_ratio") is not None:
            yield "hf_energy_ratio", s.get("hf_energy_ratio", 1) < t["min_hf_energy_ratio"]
        if t.get("min_rolloff_hz") is not None:
            yield "rolloff_hz", s.get("rolloff_hz", 1e9) < t["min_rolloff_hz"]
        if t.get("max_reverb_score") is not None:
            yield "reverb_score", s.get("reverb_score", 0) > t["max_reverb_score"]
        if t.get("max_jitter_local") is not None and s.get("jitter_local", -1) >= 0:
            yield "jitter_local", s["jitter_local"] > t["max_jitter_local"]
        if t.get("max_shimmer_local") is not None and s.get("shimmer_local", -1) >= 0:
            yield "shimmer_local", s["shimmer_local"] > t["max_shimmer_local"]
        if not s.get("g2p_ok", True):
            yield "g2p", True
        if s.get("unknown_tokens", 0) > 0:
            yield "unknown_tokens", True
        if t.get("min_mora_rate") is not None and s.get("mora_count", 0) and not (
            t["min_mora_rate"] <= s.get("mora_rate", 0) <= t["max_mora_rate"]
        ):
            yield "mora_rate", True
        if t.get("min_asr_conf") is not None:
            yield "asr_conf", s.get("asr_conf", 0) < t["min_asr_conf"]
        if t.get("min_align_conf") is not None:
            yield "align_conf", s.get("align_conf", 0) < t["min_align_conf"]
        if t.get("min_quality_pesq") is not None and s.get("pesq", 99) < t["min_quality_pesq"]:
            yield "quality_pesq", True
        if t.get("min_chars") is not None:
            yield "char_count", s.get("char_count", 999) < t["min_chars"]
        if t.get("max_filler_ratio") is not None:
            yield "filler_ratio", s.get("filler_ratio", 0) > t["max_filler_ratio"] or s.get("alpha_only")

    for r in records:
        if r.status != "pending":
            continue  # dedup 等で既に除外確定
        for name, failed in checks(r):
            if failed:
                r.status, r.dropped_by = "excluded", name
                break
        else:
            r.status = "adopted"


def write_outputs(segments_by_id: dict, adopted: list[Record], outdir: Path, speaker: str, cfg: Config) -> Path:
    """esd.list + raw.zip (44.1k/mono・LUFS正規化済み flac) を書き、契約検証まで通す."""
    import pyloudnorm as pyln

    sr = cfg.get("output.sample_rate")
    meter = pyln.Meter(sr)
    target = cfg.get("output.target_lufs")
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    esd_lines = []
    for r in adopted:
        x = segments_by_id[r.utt_id].audio441.astype(np.float32)
        if len(x) > sr:
            x = pyln.normalize.loudness(x, meter.integrated_loudness(x), target)  # (data, 実測LUFS, 目標LUFS)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if peak > 10 ** (-1 / 20):
            x = x * (10 ** (-1 / 20) / peak)
        fname = f"{r.utt_id}.flac"
        sf.write(raw / fname, x, sr, format="FLAC")
        esd_lines.append(f"{fname}|{speaker}|{r.lang}|{r.text}")

    (outdir / "esd.list").write_text("\n".join(esd_lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(outdir / "raw.zip", "w", zipfile.ZIP_STORED) as z:
        for p in raw.rglob("*.flac"):
            z.write(p, p.relative_to(raw))
    _contract_check(outdir / "raw.zip")
    return outdir


def _contract_check(zip_path: Path) -> None:
    """納品前に学習側 preprocess と同じ検証 (44.1kHz/モノラル等) を自走で通す."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist()[:1]:
            data, sr = sf.read(io.BytesIO(z.read(name)))
            assert sr == 44100, f"contract violation: {name} sr={sr}"
            assert data.ndim == 1 or data.shape[1] == 1, f"contract violation: {name} not mono"


def write_report(records: list[Record], outdir: Path) -> None:
    """除外内訳・カバレッジ (音素/モーラ/F0/クラスタ分布) の可視化ドラフト."""
    dropped = Counter(r.dropped_by for r in records if r.status == "excluded")
    adopted = [r for r in records if r.status == "adopted"]
    dup_groups = {r.dup_group_id for r in records if r.dup_group_id}
    f0s = [r.scores.get("f0_median_hz") for r in adopted if r.scores.get("f0_median_hz")]
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in dropped.most_common())
    srcs = Counter(r.src for r in records)
    srclist = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in srcs.most_common())
    html = f"""<!doctype html><meta charset="utf-8"><title>dataset report</title>
<h1>Dataset Report</h1><p>adopted={len(adopted)} / total={len(records)}</p>
<h2>除外内訳</h2><table border=1>{rows or '<tr><td>(なし)</td></tr>'}</table>
<h2>投入ソース別件数</h2><table border=1>{srclist}</table>
<h2>重複排除</h2><p>{len(dup_groups)} グループ (代表1本採用)</p>
<h2>F0中央値レンジ (adopted)</h2><p>{min(f0s) if f0s else '-'} – {max(f0s) if f0s else '-'} Hz / 中央値 {np.median(f0s) if f0s else '-'} Hz</p>
<p>※ 音素被覆率・モーラ数分布ヒストグラムは今後の拡張</p>"""
    (outdir / "report.html").write_text(html, encoding="utf-8")
