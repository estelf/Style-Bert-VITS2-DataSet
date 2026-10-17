"""学習側読みチェック (YomiError) 事前排除の回帰テスト。

カバー対象:
  yomi.py: 学習側 text_to_sep_kata の失敗条件の鏡写し (読めない語の検出 / 誤爆禁止)
  g2p_check: unknown_tokens の実意味化 (pron=「、」) と g2p_ok=False・yomi_bad_words
  stage6: on_g2p_fail / on_unknown_tokens による除外・レビュー振り分け
  refresh_yomi_scores: 旧 manifest (unknown_tokens=0 固定だった死んだチェック) の rescore 引き直し
"""

from make_dataset import stage3_asr as s3
from make_dataset import stage6_output as s6
from make_dataset import yomi
from make_dataset.config import Config
from make_dataset.manifest import Record


def mk(utt, **sc):
    r = Record(utt_id=utt, src="x.mp3", start=0.0, end=1.0)
    r.scores.update(sc)
    return r


def cfg_with(thresholds=None):
    cfg = Config.load(None)
    cfg.data["thresholds"] = thresholds or {}
    return cfg


# ------------------------------------------------------------------ yomi.py 鏡写し
def test_yomi_flags_words_learner_cannot_read():
    # 学習側 clean_text(raise_yomi_error=True) が YomiError を投げる語 (実機で FAIL 確認済み)
    assert yomi.unreadable_words("䶮さん、こちらへ。") == ["䶮"]
    assert yomi.unreadable_words("龘と龍。") == ["龘"]
    assert yomi.unreadable_words("ヷヸヹヺ。") == ["ヷヸヹヺ"]


def test_yomi_no_false_positive_on_learner_stripped_or_readable():
    # 絵文字・キリル文字・ダッシュは学習側 normalizer が解析前に消す/素通り → 誤爆禁止
    assert yomi.unreadable_words("こんにちは、世界。") == []
    assert yomi.unreadable_words("😀すごい。") == []
    assert yomi.unreadable_words("──あの、すみません。") == []
    assert yomi.unreadable_words("100円と50%、確認お願いします。") == []
    assert yomi.unreadable_words("№5、㎡、㈱サンプル。") == []
    assert yomi.unreadable_words("垰さん、こちらへ。") == []  # OpenJTalk が読める希少漢字 (タオ)
    assert yomi.unreadable_words("……") == []
    assert yomi.unreadable_words("") == []


def test_g2p_check_records_unreadable_words():
    g = s3.g2p_check("䶮さん、お願いします。")
    assert g["g2p_ok"] is False and g["unknown_tokens"] == 1
    assert "䶮" in g["yomi_bad_words"]
    ok = s3.g2p_check("お願いします。")
    assert ok["g2p_ok"] is True and ok["unknown_tokens"] == 0
    assert "yomi_bad_words" not in ok


# ------------------------------------------------------------------ stage6 振り分け
def test_yomi_failure_excluded_by_default_actions():
    r = mk(
        "y",
        g2p_ok=False,
        unknown_tokens=1,
        yomi_bad_words="䶮",
        mora_count=5,
        asr_conf=0.9,
        align_conf=0.9,
        seg_dur=1.0,
    )
    r.text = "䶮です"
    cfg = cfg_with({"on_g2p_fail": "exclude", "on_unknown_tokens": "exclude"})
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "excluded"
    assert r.dropped_by is not None
    assert "g2p" in r.dropped_by and "unknown_tokens" in r.dropped_by


def test_yomi_failure_dict_data_routes_to_review():
    r = mk(
        "y",
        g2p_ok=False,
        unknown_tokens=1,
        yomi_bad_words="䶮",
        mora_count=5,
        asr_conf=0.9,
        align_conf=0.9,
        seg_dur=1.0,
    )
    r.text = "䶮です"
    cfg = cfg_with({"on_g2p_fail": "dict_data", "on_unknown_tokens": "dict_data"})
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "pending_review" and "dict_data" in (r.review_reason or "")


# ------------------------------------------------------------------ rescore での引き直し
def test_refresh_yomi_scores_backfills_old_manifest():
    bad, clean = mk("b", mora_count=5, char_count=6), mk("c", mora_count=5, char_count=6)
    bad.text, clean.text = "䶮です。", "こんにちは。"
    for r in (bad, clean):
        r.scores["unknown_tokens"] = 0  # 旧 (死んでいた) チェックの値で止まっている旧 manifest 相当
    assert s3.refresh_yomi_scores([bad, clean]) == 1
    assert bad.scores["unknown_tokens"] == 1 and bad.scores["g2p_ok"] is False
    assert bad.scores["yomi_bad_words"] == "䶮"
    assert (
        clean.scores["unknown_tokens"] == 0 and "g2p_ok" not in clean.scores
    )  # 能動的に True にはしない
    # rescore 相当の再判定で on_g2p_fail=exclude が効く
    cfg = cfg_with({"on_g2p_fail": "exclude", "min_chars": 4})
    s6.evaluate_checks([bad, clean], cfg)
    s6.finalize_statuses([bad, clean])
    assert bad.status == "excluded" and clean.status == "adopted"


def test_report_lists_yomi_words(tmp_path):
    r = mk("y", g2p_ok=False, unknown_tokens=1, yomi_bad_words="䶮", mora_count=5)
    r.text = "䶮です"
    r.status = "excluded"
    r.dropped_by = ["g2p"]
    s6.write_report([r], tmp_path, cfg_with({}))
    assert "䶮" in (tmp_path / "report.html").read_text(encoding="utf-8")
