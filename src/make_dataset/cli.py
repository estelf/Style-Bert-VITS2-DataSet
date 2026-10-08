"""CLI 本体: uv run make_dataset <input...> [--model-name 名前] --config filters.toml

話者名・データセットディレクトリ名は省略時、音声が入っているディレクトリ名から自動採用する
(例: test/asset/haruka_ran_nobgm/*.mp3 → 話者名も Data/haruka_ran_nobgm/ も同じ名前)。
"""

from __future__ import annotations

import typer

from .config import Config
from .manifest import write_manifest
from .stage0_ingest import ingest
from .stage1_segment import segment_sources
from .stage2_score import score_segments
from .stage3_asr import flag_mora_rate_outliers, transcribe_segments
from .stage4_quality import score_quality
from .stage5_dedup import dedup
from .stage6_output import apply_thresholds, write_outputs, write_report

app = typer.Typer(add_completion=False)


@app.command()
def run(
    inputs: list[str] = typer.Argument(..., help="音声・動画ファイル (全部で1話者分のデータセット)"),
    model_name: str = typer.Option(None, "-m", "--model-name", help="出力先 Data/<モデル名>/ (省略時は入力ディレクトリ名から自動)"),
    speaker: str = typer.Option(None, "--speaker", help="話者名 (省略時はモデル名と同じ)"),
    config: str = typer.Option("filters.toml", "--config", help="閾値・挙動設定 filters.toml"),
    outdir: str = typer.Option("Data", "--outdir", help="出力ルート (Data/<モデル名>/ に書く)"),
):
    from pathlib import Path

    cfg = Config.load(Path(config))
    paths = [Path(p) for p in inputs]
    model_name = model_name or paths[0].parent.name  # 音声が入っているディレクトリ名をそのまま使う
    speaker = speaker or model_name

    sources, skipped = ingest(paths)
    typer.echo(f"[stage0] sources={len(sources)} byte-hash除外={len(skipped)}")
    for path, why in skipped:
        typer.echo(f"  skip {path.name}: {why}")

    segments = segment_sources(sources, cfg)
    typer.echo(f"[stage1] segments={len(segments)}")
    score_segments(segments)
    transcribe_segments(segments, cfg)
    flag_mora_rate_outliers(segments)
    score_quality(segments)
    records = [s.record for s in segments]
    dedup(records, cfg)
    apply_thresholds(records, cfg)

    out = Path(outdir) / model_name
    write_manifest(records, out / "manifest.jsonl")
    segments_by_id = {s.record.utt_id: s for s in segments}
    adopted = [r for r in records if r.status == "adopted"]
    write_outputs(segments_by_id, adopted, out, speaker, cfg)
    write_report(records, out)
    typer.echo(f"[stage6] adopted={len(adopted)}/{len(records)} -> {out}/ (esd.list, raw.zip, manifest.jsonl, report.html)")
    from collections import Counter

    for k, v in Counter(r.dropped_by for r in records if r.status == "excluded").most_common():
        typer.echo(f"  excluded by {k}: {v}")


def main():
    app()
