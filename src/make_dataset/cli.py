"""CLI 本体: uv run make_dataset run <話者名> [--model-name 名前] --config filters.toml

音声は `voice/<話者名>/` に置く。`run <話者名>` は `[paths] voice` (既定 "voice") 直下の
ディレクトリとして解決し、中のメディアファイルを展開する。話者名・データセットディレクトリ名は
そのディレクトリ名から自動採用する (例: `run speeker2` → voice/speeker2/ を読み Data/speeker2/ へ)。
話者名とパス指定は「見た目」で分岐する: 区切り文字を含む / `.` `~` で始まる / メディア拡張子を
持つトークンだけパスとして扱い、それ以外は常に話者名 (voice 配下) とする。そのため
`voice/<トークン>` がカレントの同名ファイル・ディレクトリと衝突することはない
(1 実行 = 1 話者分 = 同じディレクトリのファイル)。

`rescore` サブコマンドは manifest.jsonl と中間キャッシュ (.cache/) だけから、閾値変更で
採用/除外と納品物 (esd.list / raw.zip / report.html) を再生成する (DESIGN.md「閾値変更だけで再判定」)。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from .config import Config
from .manifest import read_manifest, write_manifest
from .overrides import apply_text_overrides
from .stage0_ingest import ingest
from .stage1_segment import segment_sources
from .stage2_score import recompute_speaker_sim, score_segments
from .stage3_asr import (
    flag_mora_rate_outliers,
    refresh_asr2_scores,
    refresh_text_tail_scores,
    refresh_yomi_scores,
    transcribe_segments,
)
from .stage4_quality import score_quality
from .stage5_dedup import dedup, diversity_select
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


# 有効なゲートに対してプロバイダ (スコアを記録する任意機能) がオフだと「検証不能=全件不合格」になる
# もの。run 時にだけ警告する (rescore は過去ランの manifest スコアで正当に再判定できるため)。
_GATE_PROVIDERS = {
    "max_x_cer": "asr2.enabled",
    "min_mos": "stage4.subjective",
    "reject_audio_tags": "tags.enabled",
    "min_ctc_align": "alignment.enabled",
    "max_unaligned_edge": "alignment.enabled",
    "max_lm_ppl": "lm.enabled",
    "reject_overlap": "overlap.enabled",
}


def _warn_config(cfg: Config, mode: str = "run") -> None:
    """設定の相互矛盾を起動時に警告する (黙って効かないゲートを作らない)."""
    t = cfg.thresholds
    cache_min = cfg.get("stage1.cache_min_sec")
    lo = t.get("min_seg_sec")
    if lo is not None and cache_min is not None and lo < cache_min:
        typer.echo(
            f"warning: [thresholds] min_seg_sec={lo} < [stage1] cache_min_sec={cache_min}。"
            f" {cache_min}秒未満は Stage 1 でスキップ (音声キャッシュを作らない) ためこのゲートの下側は効かない"
            f" (min_seg_sec を {cache_min} 以上に上げるか cache_min_sec を下げる)",
            err=True,
        )
    if mode != "run":
        return
    for tkey, provider in _GATE_PROVIDERS.items():
        if tkey in t and t[tkey] is not None and not cfg.get(provider):
            sec, key = provider.split(".", 1)
            typer.echo(
                f"warning: [thresholds] {tkey} が有効だが [{sec}] {key} = false。"
                f" このランではスコアが記録されず全件「検証不能=不合格」になる"
                f" (既にスコアを持つ manifest への再判定なら rescore を使うこと)",
                err=True,
            )


def _judge_and_write(records, cfg: Config, out: Path, speaker: str, cache: Path, mode: str):
    """人手上書きの再適用 → 相関集計の再計算 → 閾値評価 → 重複排除 → 多様性選別 → 確定 → 書き出し."""
    n_ov = apply_text_overrides(records, out, cfg)  # dedup・話速中央値より前に適用するのが重要
    if n_ov:
        typer.echo(f"[review] text_overrides.jsonl から {n_ov} 件の上書きを再適用")
    recompute_speaker_sim(
        records
    )  # C: 主話者 centroid を頑健サブセットから (emb があるので rescore でも最新化)
    flag_mora_rate_outliers(records)  # C: 話速中央値もクリーンな発話だけで再計算
    evaluate_checks(records, cfg)  # 全失格理由を記録 (dedup より先に)
    dedup(records, cfg)  # 閾値通過者から代表を選ぶ
    diversity_select(records, cfg)  # 多様性選別: 重複排除後の候補から「既採用と類似すぎ」を落とす
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
        typer.echo(
            f"  pending_review (納品しない見送り: ユーザー辞書候補・修復音声等): {n_pending}"
            f" → report.html の「見送り」一覧。確定するなら text_overrides.jsonl に status を書いて rescore"
        )


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
    _warn_config(cfg, "run")
    paths = resolve_inputs(inputs, Path(cfg.get("paths.voice")))
    typer.echo(f"[input] {paths[0].resolve().parent} から {len(paths)} ファイル")
    # カレントの素文件名 (Path("clip.mp3").parent.name == "") でも話者名が空にならない
    model_name = model_name or paths[0].resolve().parent.name
    speaker = speaker or model_name

    sources, skipped = ingest(paths)
    typer.echo(f"[stage0] sources={len(sources)} byte-hash除外={len(skipped)}")
    for path, why in skipped:
        typer.echo(f"  skip {path.name}: {why}")

    out = Path(outdir) / model_name
    cache = out / ".cache"  # 中間キャッシュと納品用 raw/ を分離
    records = segment_sources(sources, cfg, cache)
    typer.echo(f"[stage1] segments={len(records)}")
    score_segments(records, cfg, cache)
    transcribe_segments(records, cfg, cache)
    score_quality(records, cfg, cache)
    if cfg.get(
        "repair.enabled"
    ):  # ノイズ系ゲート単独落ちを除去→再検証 (通過しても pending_review 行き)
        from .repair import repair_pass

        n_rep = repair_pass(records, cfg, cache)
        typer.echo(f"[repair] {n_rep} 本を修復 (元音声は .cache_orig/ に保持。再検証済み)")
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
    """音声処理を再実行せず、manifest.jsonl + 中間キャッシュから閾値再判定だけで納品物を再生成する."""
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
    _warn_config(cfg, "rescore")
    records = read_manifest(out / "manifest.jsonl")
    typer.echo(f"[rescore] records={len(records)} config={config}")
    # 旧 manifest の unknown_tokens は死んでいた旧チェックの値。テキストから読みチェックを引き直す
    n_yomi = refresh_yomi_scores(records)
    if n_yomi:
        typer.echo(
            f"[yomi] 学習側読みチェック (YomiError) 失敗を {n_yomi} 件検出 "
            "(on_g2p_fail/on_unknown_tokens で判定)"
        )
    # 旧 manifest には text_tail_open が無い。テキストから引き直して、再 run なしで
    # max_text_tail_open を再調整できるようにする (無補正で rescore すると全件「検証不能」で落ちる)
    n_tail = refresh_text_tail_scores(records)
    if n_tail:
        typer.echo(f"[tail] 文末の開き を {n_tail} 件引き直し")
    # [asr2] を後から有効にしたケース。第 2 ASR は音声を引くしかないが、run だとメイン ASR・
    # AST・CTC・LM まで全部やり直しになるため、キャッシュから x_cer だけ追加計算する
    n_x = refresh_asr2_scores(records, cfg, cache)
    if n_x:
        typer.echo(f"[asr2] 第 2 ASR クロス検証 (x_cer・読みレベル) を {n_x} 件引き直し")
    if cfg.thresholds.get("max_x_cer") and not cfg.get("asr2.enabled"):
        missing = sum(
            1 for r in records if (r.text or "").strip() and "x_cer" not in (r.scores or {})
        )
        if missing:  # 黙って全件「検証不能」で落とさない (run 側の _warn_config と同じ方針)
            typer.echo(
                f"warning: max_x_cer 有効なのに [asr2] 無効で {missing} 件に x_cer が無い →"
                " 該当レコードは検証不能として不合格になります。asr2.enabled=true で自動計算するか"
                " max_x_cer をコメント-out してください",
                err=True,
            )
    _judge_and_write(records, cfg, out, speaker or out.name, cache, mode="rescore")


def main():
    app()
