"""ゴールデン方式スモークテスト: 数秒分の音声でパイプライン1周 → esd.list/raw.zip が契約検証を通ることを確認."""

import subprocess
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner


def test_resolve_inputs(tmp_path, monkeypatch):
    """話者名とパス指定の判別 (カレント同名と衝突しない) と 1 話者契約のガード."""
    from make_dataset.cli import resolve_inputs

    voice = tmp_path / "voice"
    (voice / "speeker1").mkdir(parents=True)
    for name in ("02.mp3", "01.mp4", "notes.txt"):  # 01.mp4 / 02.mp3 が名前順、txt は対象外
        (voice / "speeker1" / name).write_bytes(b"")
    (voice / "speeker2").mkdir()
    (voice / "speeker2" / "01.mp3").write_bytes(b"")
    (tmp_path / "speeker1").mkdir()  # voice/ と同名のカレントディレクトリ (衝突の確認)
    (tmp_path / "speeker1" / "99.mp3").write_bytes(b"")
    (tmp_path / "solo.mp3").write_bytes(b"")

    monkeypatch.chdir(tmp_path)  # [paths] voice の既定 "voice" はカレント基準
    # 話者名は voice 配下だけを見る。カレントの同名 speeker1/ は解決に割り込まない
    assert resolve_inputs(["speeker1"], Path("voice")) == [
        Path("voice/speeker1/01.mp4"),
        Path("voice/speeker1/02.mp3"),
    ]
    # パス指定 (区切り文字を含む) はカレント基準。voice 配と混ざらない
    assert resolve_inputs(["./speeker1"], Path("voice")) == [Path("speeker1/99.mp3")]
    assert resolve_inputs([str(voice / "speeker2")], Path("voice")) == [
        voice / "speeker2" / "01.mp3"
    ]
    # メディア拡張子を持つ裸のトークンはファイル指定として扱う
    assert resolve_inputs(["solo.mp3"], Path("voice")) == [Path("solo.mp3")]
    with pytest.raises(typer.Exit):  # 話者名が voice 配に無い (カレントに同名があっても救済しない)
        resolve_inputs(["nobody"], Path("voice"))
    with pytest.raises(typer.Exit):  # 2 話者を 1 実行に混ぜると話者名が定まらない
        resolve_inputs(["speeker1", "speeker2"], Path("voice"))


def test_smoke_pipeline_one_pass(tmp_path, monkeypatch):
    asset = Path(__file__).parents[0] / "asset/test"
    voice = tmp_path / "voice/test"  # voice/<話者名>/ 配置。ディレクトリ名が話者名になる
    voice.mkdir(parents=True)
    clip = voice / "clip.mp3"
    src = sorted(asset.glob("01.*"))[0]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-t", "20", "-i", str(src), str(clip)],
        check=True,
    )

    cfg = tmp_path / "filters.toml"
    cfg.write_text('[asr]\nmodel = "tiny"\n')  # スモークテストは tiny で速く回す

    from make_dataset.cli import app

    monkeypatch.chdir(tmp_path)  # voice ルート (既定 "voice") はカレント基準
    res = CliRunner().invoke(
        app,
        ["run", "test", "--config", str(cfg), "--outdir", str(tmp_path / "Data")],
    )
    assert res.exit_code == 0, res.output

    out = tmp_path / "Data/test"
    lines = (out / "esd.list").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    for line in lines:  # 契約: <音声ファイル名>|<話者名>|JP|<セリフ> (スタイルフォルダなしの直配置)
        name, spk, lang, text = line.split("|")
        assert name.endswith(".flac") and "/" not in name
        assert spk == "test" and lang == "JP" and text

    import zipfile

    with zipfile.ZipFile(out / "raw.zip") as z:
        assert len(z.namelist()) == len(
            lines
        )  # esd.list と実ファイルが一致 (write_outputs 内で sr/mono も検証済み)
        import io

        import soundfile as sf

        data, _sr = sf.read(io.BytesIO(z.read(z.namelist()[0])))
        assert (
            float(abs(data).max()) > 0.01
        )  # LUFS 正規化が効いていて実際に鳴る音量であること (全ゼロ回帰の検知)
        import numpy as np

        assert np.isfinite(data).all()  # nan/inf が混じっていない

    # 再現性メタデータ
    import json

    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "run" and "thresholds" in meta["config"]

    # manifest.jsonl は厳密な JSON として読み直せる (NaN/-Infinity の混入なし)
    for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)

    # 2回目の実行でも前回除外済みファイルが raw.zip に混入しない (件数が esd.list と一致し続ける)
    res = CliRunner().invoke(
        app,
        ["run", "test", "--config", str(cfg), "--outdir", str(tmp_path / "Data")],
    )
    assert res.exit_code == 0, res.output
    lines2 = (out / "esd.list").read_text(encoding="utf-8").splitlines()
    with zipfile.ZipFile(out / "raw.zip") as z:
        assert len(z.namelist()) == len(lines2)

    # rescore — 閾値を変えて manifest 再生成だけで再判定できる (DESIGN テスト方針)
    strict = tmp_path / "strict.toml"
    strict.write_text('[asr]\nmodel = "tiny"\n[thresholds]\nmin_chars = 999\n')
    res = CliRunner().invoke(app, ["rescore", str(out), "--config", str(strict)])
    assert res.exit_code == 0, res.output
    assert (out / "esd.list").read_text(encoding="utf-8") == ""  # 全件 min_chars で落ちる
    with zipfile.ZipFile(out / "raw.zip") as z:
        assert z.namelist() == []
