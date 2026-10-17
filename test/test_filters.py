"""フィルタ・選別・契約の回帰テスト (合成レコード中心。モデル DL 不要の高速テスト + qwen の任意実行)。

カバー対象:
  感度分析の閾値キー分離 / 空テキスト不変条件 / F0 ゲート配線 / 出力契約ガード
  計算済みスコアのゲート化 (digital_silence / voiced_ratio / stoi / si_sdr / lufs / overlap)
  HTML エスケープ / on_* 値の検証 / adopt 昇格 / stage1 スキップ / チャンク分割
  多様性選別 (Stage 5b)
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from make_dataset import stage5_dedup as s5
from make_dataset import stage6_output as s6
from make_dataset.config import Config
from make_dataset.manifest import Record


def mk(utt, **sc):
    r = Record(utt_id=utt, src="x.mp3", start=0.0, end=1.0)
    r.scores.update(sc)
    return r


def cfg_with(thresholds=None, **over):
    cfg = Config.load(None)
    cfg.data["thresholds"] = thresholds or {}
    for dotted, val in over.items():
        cur = cfg.data
        keys = dotted.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = val
    return cfg


# ------------------------------------------------------------------ 感度分析
def test_sensitivity_counts_by_threshold_key():
    short = mk("s", seg_dur=0.3, mora_count=5)
    short.text = "あいうえお"
    long_ = mk("l", seg_dur=13.0, mora_count=5)
    long_.text = "あいうえお"
    cfg = cfg_with({"min_seg_sec": 0.5, "max_seg_sec": 12.0})
    s6.evaluate_checks([short, long_], cfg)
    s6.finalize_statuses([short, long_])
    assert short.scores["failed_checks"] == ["min_seg_sec"]
    assert long_.scores["failed_checks"] == ["max_seg_sec"]  # 共有スコア seg_dur でも混線しない
    stats = {s["key"]: s for s in s6._sensitivity_stats([short, long_], cfg)}
    assert (
        stats["max_seg_sec"]["gain"] == 1
    )  # 緩和で救えるのは長い発話 1 本だけ (短い発話は混ざらない)
    assert stats["min_seg_sec"]["gain"] == 0  # 0.5→0.4 緩和では 0.3 秒発話は救われない
    assert stats["max_seg_sec"]["fails"] == stats["min_seg_sec"]["fails"] == 1
    assert stats["max_seg_sec"]["relaxed"] == pytest.approx(15.0)  # max は 1.25 倍に緩和
    assert stats["min_seg_sec"]["relaxed"] == pytest.approx(0.4)  # min は 0.8 倍に緩和


# ------------------------------------------------------------------ 空テキスト不変条件
def test_empty_text_never_adopted_even_with_gates_off():
    r = mk("e", mora_count=0, asr_conf=0.9, align_conf=0.9, seg_dur=1.0)
    r.text = ""
    cfg = cfg_with({"min_asr_conf": 0.3, "min_align_conf": 0.3})  # min_chars / on_g2p_fail は無し
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "excluded"
    assert "empty_text" in (r.dropped_by or [])


# ------------------------------------------------------------------ / B ゲート配線
@pytest.mark.parametrize(
    "score_key",
    [
        "f0_median_hz",
        "digital_silence_ratio",
        "voiced_ratio",
        "stoi",
        "si_sdr",
        "lufs",
        "overlap_flag",
    ],
)
def test_computed_scores_are_gated(score_key):
    assert score_key in {s for _, s, _ in s6.CHECKS}


def test_digital_silence_gate_excludes_near_silence():
    r = mk("q", seg_dur=1.0, digital_silence_ratio=0.95, mora_count=8, asr_conf=0.9, align_conf=0.9)
    r.text = "あいうえおかきく"
    cfg = cfg_with({"max_digital_silence": 0.5})
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "excluded" and "max_digital_silence" in (r.dropped_by or [])


def test_f0_gate_excludes_out_of_band():
    r = mk("f", f0_median_hz=90.0, mora_count=8, asr_conf=0.9, align_conf=0.9)
    r.text = "あいうえお"
    cfg = cfg_with({"min_f0_hz": 120.0})
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert "min_f0_hz" in (r.dropped_by or [])


# ------------------------------------------------------------------ on_* の扱い (adopt 削除済み)
def test_on_action_typo_raises():
    r = mk("b", g2p_ok=True, mora_count=5)
    r.text = "あ"
    cfg = cfg_with({"on_g2p_fail": "excldue"})  # 誤字
    with pytest.raises(ValueError):
        s6.evaluate_checks([r], cfg)


def test_on_unknown_adopt_option_removed():
    # adopt は削除: unknown (学習側で読めない語) を含む行の納品は学習側 preprocess で必ず落ちるため
    r = mk(
        "u", g2p_ok=False, unknown_tokens=2, mora_count=5, asr_conf=0.9, align_conf=0.9, seg_dur=1.0
    )
    r.text = "あいうえお"
    cfg = cfg_with({"on_unknown_tokens": "adopt"})
    with pytest.raises(ValueError):
        s6.evaluate_checks([r], cfg)


# ------------------------------------------------------------------ 出力契約ガード
def test_contract_guard_rejects_non_contract_sample_rate():
    cfg = Config.load(None)
    cfg.data["output"]["sample_rate"] = 22050
    with tempfile.TemporaryDirectory() as d, pytest.raises(ValueError):
        s6.write_outputs([], Path(d), Path(d) / "out", "spk", cfg)


# ------------------------------------------------------------------ 多様性選別 (Stage 5b)
def _rec(utt, text, q):
    r = mk(utt, mora_count=5, quality_score=q, asr_conf=0.9, snr=20.0)
    r.text = text
    return r


def _judge_diversity(recs, cfg):
    s6.evaluate_checks(recs, cfg)
    s5.dedup(recs, cfg)
    s5.diversity_select(recs, cfg)
    s6.finalize_statuses(recs)


def _div_cfg(metric="phoneme", thr=0.6, enabled=True, protect=False):
    return cfg_with(
        {},
        **{
            "diversity": {
                "enabled": enabled,
                "metric": metric,
                "sim_threshold": thr,
                "protect_coverage": protect,
            }
        },
    )


def test_diversity_drops_similar_keeps_distinct():
    # 実測: 「似た内容」~0.67 / 「別発話」≤0.05。既定 0.6 は似た方だけ落とし別発話には触れない
    A = _rec("A", "今日はいい天気ですね", 4.0)
    B = _rec("B", "今日はいい天気でした", 3.0)  # A と似た内容 (音素三量 Jaccard ~0.67)
    C = _rec("C", "猫が寝転んだ", 2.0)  # 別内容
    _judge_diversity([A, B, C], _div_cfg())
    assert A.status == "adopted"
    assert C.status == "adopted"
    assert B.status == "excluded" and any(
        x.startswith("diversity:similar:A") for x in (B.dropped_by or [])
    )


def test_diversity_disabled_keeps_similar():
    A = _rec("A", "今日はいい天気ですね", 4.0)
    B = _rec("B", "今日はいい天気でした", 3.0)
    _judge_diversity([A, B], _div_cfg(enabled=False))
    assert all(r.status == "adopted" for r in (A, B))


def test_diversity_emb_metric_rejected():
    # 単一話者で emb(話者/音響ベクトル) は別内容まで間引くため不採用 → 明確なエラー
    recs = [_rec("A", "今日はいい天気ですね", 4.0), _rec("B", "猫が寝転んだ", 3.0)]
    with pytest.raises(ValueError):
        s5.diversity_select(recs, _div_cfg(metric="emb"))


def test_diversity_threshold_not_below_dedup_warns():
    # sim_threshold >= dedup.phoneme_jaccard なら dedup が先に間引くので多様性は効かない → 警告
    recs = [_rec("A", "今日はいい天気ですね", 4.0), _rec("B", "猫が寝転んだ", 3.0)]
    with pytest.warns(UserWarning):
        s5.diversity_select(recs, _div_cfg(metric="phoneme", thr=0.95))


# ------------------------------------------------------------------ チャンク分割
def test_chunk_windows_tile_without_gap_or_overlap():
    from make_dataset.stage1_segment import _chunk_windows

    total, cs = 100.0, 30.0
    wins = list(_chunk_windows(total, cs, 0.6, 13.0))
    owned = [(a, b) for _, _, a, b in wins]
    assert owned[0][0] == 0.0 and abs(owned[-1][1] - total) < 1e-9
    for i in range(len(owned) - 1):
        assert abs(owned[i][1] - owned[i + 1][0]) < 1e-9  # 隙間も重複もない
    for win_lo, win_hi, olo, ohi in wins:  # 窓は所有領域を、ファイルが許す限りの pad 付きで覆う
        assert win_lo <= olo + 1e-9 and win_hi >= ohi - 1e-9
        assert win_hi >= min(ohi + 13.0, total) - 1e-9


# ------------------------------------------------------------------ HTML エスケープ
def test_report_escapes_html():
    r = mk("p", g2p_ok=True, mora_count=5)
    r.text = "<script>alert(1)</script>"
    r.src = "a&b<x>.mp3"
    r.review_reason = "dict_data"
    r.status = "pending_review"
    with tempfile.TemporaryDirectory() as d:
        s6.write_report([r], Path(d), cfg_with({}))
        html = (Path(d) / "report.html").read_text(encoding="utf-8")
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "a&amp;b&lt;x&gt;.mp3" in html


# ------------------------------------------------------------------ Stage 3 バックエンド選択
def test_asr_backend_selection_and_unknown(monkeypatch):
    import make_dataset.stage0_ingest as s0
    import make_dataset.stage3_asr as s3

    called = {}
    monkeypatch.setattr(s0, "load_cached", lambda r, c, sr: np.zeros(1600, dtype=np.float32))
    for name in ("_asr_whisper", "_asr_qwen"):
        monkeypatch.setattr(
            s3,
            name,
            (
                lambda n: (
                    lambda a, cfg: (
                        called.setdefault(n, 0),
                        called.__setitem__(n, called[n] + 1),
                        ("テスト", 0.9, {}),
                    )[-1]
                )
            )(name),
        )

    r = mk("u0", seg_dur=1.0)
    s3.transcribe_segments([r], cfg_with({}, **{"asr.backend": "faster-whisper"}), Path("."))
    assert called.get("_asr_whisper") == 1 and r.text == "テスト"
    s3.transcribe_segments([r], cfg_with({}, **{"asr.backend": "qwen3-asr"}), Path("."))
    assert called.get("_asr_qwen") == 1

    with pytest.raises(ValueError):
        s3.transcribe_segments([r], cfg_with({}, **{"asr.backend": "bogus"}), Path("."))


# ------------------------------------------------------------------ qwen 実バックエンド (任意実行)
@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1 to exercise the real Qwen3-ASR backend"
)
def test_qwen_backend_real(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from make_dataset.cli import app

    asset = sorted((Path(__file__).parent / "asset/test").glob("01.*"))[0]
    voice = tmp_path / "voice/qtest"
    voice.mkdir(parents=True)
    # テストアセットの音声区間は ~7.9s 以降なので、その窓 (12s) を切り出して ASR に掛ける
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            "7",
            "-t",
            "12",
            "-i",
            str(asset),
            str(voice / "clip.mp3"),
        ],
        check=True,
    )
    cfg = tmp_path / "filters.toml"
    cfg.write_text(
        '[asr]\nbackend = "qwen3-asr"\nmodel = "Qwen/Qwen3-ASR-1.7B-hf"\n[thresholds]\nmin_chars = 1\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(
        app, ["run", "qtest", "--config", str(cfg), "--outdir", str(tmp_path / "Data")]
    )
    assert res.exit_code == 0, res.output
    lines = (tmp_path / "Data/qtest/esd.list").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    for line in lines:
        name, spk, lang, text = line.split("|")
        assert name.endswith(".flac") and spk == "qtest" and lang == "JP" and text
