"""ゴールデン方式スモークテスト: 数秒分の音声でパイプライン1周 → esd.list/raw.zip が契約検証を通ることを確認."""

import subprocess
from pathlib import Path

from typer.testing import CliRunner


def test_smoke_pipeline_one_pass(tmp_path):
    asset = Path(__file__).parents[0] / "asset/haruka_ran_nobgm"
    clip_dir = tmp_path / "haruka_ran_nobgm"  # 入力ディレクトリ名 = 話者名 = データセットディレクトリ名
    clip_dir.mkdir()
    clip = clip_dir / "clip.mp3"
    src = sorted(asset.glob("01.*"))[0]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-t", "20", "-i", str(src), str(clip)], check=True
    )

    cfg = tmp_path / "filters.toml"
    cfg.write_text('[asr]\nmodel = "tiny"\n')  # スモークテストは tiny で速く回す

    from make_dataset.cli import app

    res = CliRunner().invoke(
        app, [str(clip), "--config", str(cfg), "--outdir", str(tmp_path / "Data")]
    )
    assert res.exit_code == 0, res.output

    out = tmp_path / "Data/haruka_ran_nobgm"
    lines = (out / "esd.list").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    for line in lines:  # 契約: <音声ファイル名>|<話者名>|JP|<セリフ> (スタイルフォルダなしの直配置)
        name, spk, lang, text = line.split("|")
        assert name.endswith(".flac") and "/" not in name
        assert spk == "haruka_ran_nobgm" and lang == "JP" and text

    import zipfile

    with zipfile.ZipFile(out / "raw.zip") as z:
        assert len(z.namelist()) == len(lines)  # esd.list と実ファイルが一致 (write_outputs 内で sr/mono も検証済み)
        import io

        import soundfile as sf

        data, _sr = sf.read(io.BytesIO(z.read(z.namelist()[0])))
        assert float(abs(data).max()) > 0.01  # LUFS 正規化が効いていて実際に鳴る音量であること (全ゼロ回帰の検知)
