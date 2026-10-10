"""CLI 本体: uv run make_dataset run <話者名> [--model-name 名前] --config filters.toml

音声は `voice/<話者名>/` に置く。`run <話者名>` は `[paths] voice` (既定 "voice") 直下の
ディレクトリとして解決し、中のメディアファイルを展開する。話者名・データセットディレクトリ名は
そのディレクトリ名から自動採用する (例: `run speeker2` → voice/speeker2/ を読み Data/speeker2/ へ)。
話者名とパス指定は「見た目」で分岐する: 区切り文字を含む / `.` `~` で始まる / メディア拡張子を
持つトークンだけパスとして扱い、それ以外は常に話者名 (voice 配下) とする。そのため
`voice/<トークン>` がカレントの同名ファイル・ディレクトリと衝突することはない
(1 実行 = 1 話者分 = 同じディレクトリのファイル)。

E1: `rescore` サブコマンドは manifest.jsonl と中間キャッシュ (.cache/) だけから、閾値変更で
採用/除外と納品物 (esd.list / raw.zip / report.html) を再生成する (DESIGN.md「閾値変更だけで再判定」)。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from .config import Config
from .manifest import read_manifest, write_manifest
from .stage0_ingest import ingest
from .stage1_segment import segment_sources
from .stage2_score import score_segments
from .stage3_asr import flag_mora_rate_outliers, transcribe_segments
from .stage4_quality import score_quality
from .stage5_dedup import dedup
from .stage6_output import (
    evaluate_checks,
    finalize_statuses,
    write_outputs,
    write_report,
    write_run_meta,
)

app = typer.Typer(add_completion=False)

# 投入として受け付ける拡張子 (Stage 1 が ffmpeg で拡張子非依存にデコードする)
MEDIA_EXTS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".mpg",
    ".mpeg",
    ".ts",
}


def _expand(directory: Path) -> list[Path]:
    """話者ディレクトリ直下 (フラット配置) のメディアファイルを名前順で展開する."""
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTS)


def _is_path_form(token: str) -> bool:
    """このトークンを「パス指定」として扱うか (話者名との判別).

    区切り文字を含む / `.` `~` で始まる / メディア拡張子を持つ ならパス指定。
    裸のトークンは常に話者名とし、`voice/<トークン>` がカレントの同名ファイル・ディレクトリと
    静かに衝突するのを防ぐ (話者名 `test` がカレントの `test/` を読みにいく事故をなくす)。
    """
    return (
        "/" in token
        or "\\" in token
        or token.startswith((".", "~"))
        or Path(token).suffix.lower() in MEDIA_EXTS
    )


def _bad_input(msg: str) -> typer.Exit:
    """入力指定の誤りを報告する。呼び出し側で `raise _bad_input(...)` して終了コード 2 で抜ける."""
    typer.echo(f"error: {msg}", err=True)
    return typer.Exit(2)


def resolve_inputs(tokens: list[str], voice_root: Path) -> list[Path]:
    """CLI 引数を音声・動画ファイル列へ展開する (話者名とパス指定は「見た目」で判別する).

    - 話者名 (裸のトークン): `voice_root/<話者名>/` のみを見る。カレントの同名とは衝突させない
    - パス指定 (区切り文字を含む等): カレント基準 (または絶対パス) のファイル・ディレクトリ
    1 実行 = 1 話者契約なので、複数ディレクトリにまたがる入力は話者名が定まらないため拒否する。
    """
    files: list[Path] = []
    for token in tokens:
        if _is_path_form(token):
            target = Path(token)
            if not target.exists():
                hint = (
                    f" (voice 配なら {voice_root / token})" if (voice_root / token).exists() else ""
                )
                raise _bad_input(
                    f"{token} が見付からない (ファイル・ディレクトリいずれでもありません){hint}"
                )
            if target.is_file():
                files.append(target)
                continue
        else:  # 話者名: voice ルート配下だけを解決する (カレントの同名とは衝突させない)
            target = voice_root / token
            if not target.is_dir():
                hint = (
                    f" カレントの {token}/ を読むなら './{token}' のようにパスで渡す"
                    if Path(token).is_dir()
                    else ""
                )
                raise _bad_input(
                    f"話者 {token} が {voice_root}/ 配下に見付からない ({target} が存在しない)。{hint}"
                )
        found = _expand(target)
        if not found:
            raise _bad_input(
                f"{target} に対応拡張子のメディアファイルがない ({' '.join(sorted(MEDIA_EXTS))})"
            )
        files.extend(found)
    parents = {f.resolve().parent for f in files}
    if len(parents) > 1:
        raise _bad_input(
            "1 実行に渡せるのは 1 話者分 (同じディレクトリのファイル) だけ。"
            f" 混在: {', '.join(sorted(str(x) for x in parents)[:3])}"
        )
    return files


def _judge_and_write(records, cfg: Config, out: Path, speaker: str, cache: Path, mode: str):
    """閾値評価 → 重複排除 → 確定 → 納品物書き出し (run と rescore で共通)."""
    evaluate_checks(records, cfg)  # E4: 全失格理由を記録 (B5: dedup より先に)
    dedup(records, cfg)  # B5: 閾値通過者から代表を選ぶ
    finalize_statuses(records)

    adopted = [r for r in records if r.status == "adopted"]
    write_outputs(adopted, cache, out, speaker, cfg)  # lufs_out / limited を scores に追記する
    write_manifest(records, out / "manifest.jsonl")  # なので manifest 書き出しはその後
    write_report(records, out, cfg)
    write_run_meta(out, cfg, mode)
    typer.echo(
        f"[stage6] adopted={len(adopted)}/{len(records)} -> {out}/ (esd.list, raw.zip, manifest.jsonl, report.html, run.json)"
    )
    for k, v in Counter(
        k for r in records if r.status == "excluded" for k in (r.dropped_by or [])
    ).most_common():
        typer.echo(f"  excluded by {k}: {v}")
    n_pending = sum(1 for r in records if r.status == "pending_review")
    if n_pending:
        typer.echo(f"  pending_review (ユーザー辞書候補等): {n_pending}")


@app.command()
def run(
    inputs: list[str] = typer.Argument(
        ...,
        help="話者名 (voice/<話者名>/ を展開) / パス指定 (./dir, voice/x, *.mp3)。1実行=1話者分",
    ),
    model_name: str = typer.Option(
        None,
        "-m",
        "--model-name",
        help="出力先 Data/<モデル名>/ (省略時は入力ディレクトリ名から自動)",
    ),
    speaker: str = typer.Option(None, "--speaker", help="話者名 (省略時はモデル名と同じ)"),
    config: str = typer.Option("filters.toml", "--config", help="閾値・挙動設定 filters.toml"),
    outdir: str = typer.Option("Data", "--outdir", help="出力ルート (Data/<モデル名>/ に書く)"),
):
    cfg = Config.load(Path(config))
    paths = resolve_inputs(inputs, Path(cfg.get("paths.voice")))
    typer.echo(f"[input] {paths[0].resolve().parent} から {len(paths)} ファイル")
    # A5: カレントの素文件名 (Path("clip.mp3").parent.name == "") でも話者名が空にならない
    model_name = model_name or paths[0].resolve().parent.name
    speaker = speaker or model_name

    sources, skipped = ingest(paths)
    typer.echo(f"[stage0] sources={len(sources)} byte-hash除外={len(skipped)}")
    for path, why in skipped:
        typer.echo(f"  skip {path.name}: {why}")

    out = Path(outdir) / model_name
    cache = out / ".cache"  # A3/E1: 中間キャッシュと納品用 raw/ を分離
    records = segment_sources(sources, cfg, cache)
    typer.echo(f"[stage1] segments={len(records)}")
    score_segments(records, cfg, cache)
    transcribe_segments(records, cfg, cache)
    flag_mora_rate_outliers(records)
    score_quality(records, cfg, cache)
    _judge_and_write(records, cfg, out, speaker, cache, mode="run")


@app.command()
def rescore(
    data_dir: str = typer.Argument(
        ..., help="既存の Data/<モデル名>/ (manifest.jsonl と .cache/ がある)"
    ),
    speaker: str = typer.Option(
        None, "--speaker", help="話者名 (省略時はデータセットディレクトリ名)"
    ),
    config: str = typer.Option("filters.toml", "--config", help="新しい閾値 filters.toml"),
):
    """E1: 音声処理を再実行せず、manifest.jsonl + 中間キャッシュから閾値再判定だけで納品物を再生成する."""
    out = Path(data_dir)
    cache = out / ".cache"
    if not (out / "manifest.jsonl").exists():
        typer.echo(f"error: {out}/manifest.jsonl がない (先に run してください)", err=True)
        raise typer.Exit(1)
    if not cache.exists():
        typer.echo(
            f"error: 中間キャッシュ {cache} がない (このデータセットは rescore 不可)",
            err=True,
        )
        raise typer.Exit(1)

    cfg = Config.load(Path(config))
    records = read_manifest(out / "manifest.jsonl")
    typer.echo(f"[rescore] records={len(records)} config={config}")
    _judge_and_write(records, cfg, out, speaker or out.name, cache, mode="rescore")


def main():
    app()
