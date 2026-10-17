"""P0-P2 新機能の回帰テスト (モデルダウンロード不要の高速テストのみ。重い検証は既定オフ)。

カバー対象:
  filler_ratio の出現回数化 / kana_ratio の分母 (漢字・句読点で薄めない) / text_features 集約
  _cer と asr2 クロス検証の配線 (バックエンドは monkeypatch)
  トークン確率統計 / itn 記号展開・残留記号 / 新ゲートの CHECKS 配線
  diversity_select: gain_lambda / max_bucket_share / max_source_share / target_count
  text_overrides.jsonl 再適用 (G2P・モーラ再計算) と status 上書き / repaired→見送り /
    report.html が見送り一覧と text_overrides の手順 (凡例) を出す
"""

import json
from pathlib import Path

import pytest

from make_dataset import itn
from make_dataset import stage3_asr as s3
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


# ------------------------------------------------------------------ text_features
def test_filler_ratio_counts_occurrences():
    assert s3.text_features("えっとえっとえっと")["filler_ratio"] == 1.0  # 9/9 (旧実装は 3/9)
    f = s3.text_features("えっとあいうえおかきくけこ")  # 3/13
    assert f["filler_ratio"] < 0.4


def test_kana_ratio_not_diluted_by_kanji_or_punctuation():
    # 旧実装 (÷len(text)) は 7/16=0.44 で min_kana_ratio=0.5 を誤爆していた
    assert s3.text_features("本日、会議は中止となりました。")["kana_ratio"] == 1.0
    assert s3.text_features("hello world")["kana_ratio"] == 0.0  # 英語テキストは従来どおり 0
    assert (
        s3.text_features("今日はABCに行こう")["kana_ratio"] == 0.57
    )  # かな4/(4+英数3)。漢字は分母に入れない


# ------------------------------------------------------------------ 文末の開き (text_tail_open)
def test_tail_openness_zero_for_closed_sentences():
    """文を閉じうる形 (終止形・命令形・口語の未然形/仮定縮約) は一切誤爆しないこと。"""
    for t in (
        "おかえりなさいませ。",  # 動詞 基本形
        "お帰りなさいませ。",  # 助動詞 命令ｅ (招きのご挨拶 = 完文)
        "気持ちいいでしょ。",  # 助動詞 未然形 (口語の完文)
        "失敗しないようにしなきゃ。",  # 仮定縮約 (〜なければならない = 完文)
        "ちょっと待って。",  # 連用タ接続の口語依頼 = 完文
        "本日はメイドのご指名。",  # 名詞止め (体言止め) = 完文
        "お客様は。",  # 助詞止めは口語の普通形なので検出しない (実コーパスで 34%)
    ):
        assert s3.tail_openness(t) == (0, ""), t


def test_tail_openness_two_for_truncated():
    """納品してはいけない「途中で切れている」を 2 で拾う。"""
    assert s3.tail_openness("お客様の体温と混じっ") == (2, "no_sent_end")  # 文末記号すらない
    # ASR が句点を打っても、促音止めの動詞語幹 (読みがッで終わる) なら 2 のまま
    assert s3.tail_openness("お客様の体温と混じっ。") == (2, "verb_cut")


def test_tail_openness_one_for_trailing_off():
    """条件節・連用形中止・連体止めは 1 (既定は採用。分布を見て判断できるように記録だけ残す)。"""
    assert s3.tail_openness("してほしいことがあったら。") == (1, "cform:仮定形")
    assert s3.tail_openness("ご満足いただけたようで。") == (1, "cform:連用形")
    assert s3.tail_openness("お客様の好きな。") == (1, "cform:体言接続")
    assert s3.tail_openness("冷凍食品とか使わず。") == (1, "cform:連用ニ接続")


def test_tail_openness_empty_and_decoration_only():
    # 空テキストは empty_text 不変条件が拾うので二重に叩かない
    assert s3.tail_openness("") == (0, "")
    # 記号だけ = 解析対象の実詞が無い (min_chars / residual_symbols 側が拾う)
    assert s3.tail_openness("…")[0] == 0


def test_g2p_check_records_tail_openness():
    """G2P と同じ run_frontend を流用してスコアを記録する (同じ解析を二回やらない)。"""
    out = s3.g2p_check("お客様の体温と混じっ")
    assert out["text_tail_open"] == 2 and out["text_tail_open_reason"] == "no_sent_end"
    assert s3.g2p_check("おかえりなさい。")["text_tail_open"] == 0


def test_refresh_text_tail_scores_backfills_old_manifest():
    """旧 manifest (text_tail_open 無し) を rescore すると全件「検証不能」で落ちる。引き直して救う。"""
    r = mk("old")
    r.text = "お客様の体温と混じっ"
    assert "text_tail_open" not in r.scores
    assert s3.refresh_text_tail_scores([r]) == 1
    assert r.scores["text_tail_open"] == 2 and r.scores["text_tail_open_reason"] == "no_sent_end"


def _tail_rec(sev, utt="t"):
    r = mk(utt, text_tail_open=sev, mora_count=8, asr_conf=0.9, align_conf=0.9, seg_dur=1.0)
    r.text = "お客様の体温と混じっ" if sev == 2 else "してほしいことがあったら。"
    return r


def test_max_text_tail_open_gate():
    """ゲート 1 = 2 (切れ) だけ落とす / ゲート 0 = 1 (言い淀み) も落とす / キー削除で無効。"""
    strict = cfg_with({"max_text_tail_open": 0})
    for sev, cfg, want in (
        (2, cfg_with({"max_text_tail_open": 1}), "excluded"),  # 2 は常に落とす
        (1, cfg_with({"max_text_tail_open": 1}), "adopted"),  # 1 (言い淀み) は既定で採用
        (1, strict, "excluded"),  # ゲートを 0 にすると 1 も落ちる
        (2, cfg_with({}), "adopted"),  # キー無し = チェック自体が走らない
    ):
        r = _tail_rec(sev)
        s6.evaluate_checks([r], cfg)
        s6.finalize_statuses([r])
        assert r.status == want, (sev, cfg.thresholds.get("max_text_tail_open"), r.status)
        assert ("max_text_tail_open" in (r.dropped_by or [])) == (want == "excluded"), r.dropped_by


# ------------------------------------------------------------------ itn / normalize_text
def test_symbol_readings_expand_and_residual_counted():
    t, n = itn.expand_symbols("50%です")
    assert t == "50パーセントです" and n == 1
    assert itn.residual_symbols("50%です") == 1
    assert itn.residual_symbols("50パーセントです。") == 0  # 展開後 + 句読点は残留ではない


def test_normalize_text_expands_symbols_toggle():
    assert s3.normalize_text("50%です") == "50パーセントです"
    assert s3.normalize_text("50%です", expand_symbols=False) == "50%です"


# ------------------------------------------------------------------ CER / asr2 配線
def test_cer_basic():
    assert s3._cer("あいう", "あいう") == 0.0
    assert abs(s3._cer("あいう", "あいうえお") - 0.4) < 1e-9  # 編集距離 2 / 最大長 5


def test_asr2_crosscheck_records_x_cer(monkeypatch):
    import make_dataset.stage0_ingest as s0

    monkeypatch.setattr(s0, "load_cached", lambda r, c, sr: __import__("numpy").zeros(16000))

    def fake(audio, cfg, prefix="asr"):
        if prefix == "asr":
            return "今日はいい天気ですね", 0.9, {}
        return "今日はいい天気だ", 0.8, {}  # 末尾だけ違う第 2 系統

    monkeypatch.setattr(s3, "_asr_whisper", fake)
    cfg = cfg_with(
        {},
        **{
            "asr.backend": "faster-whisper",
            "asr.model": "tiny",
            "asr2.enabled": True,
            "asr2.backend": "faster-whisper",
            "asr2.model": "small",
        },
    )
    ok = mk("u0", seg_dur=1.0)
    garbage = mk("u1", seg_dur=0.1)  # 「明らかなゴミ」カットオフ (0.3 秒未満) でスキップ
    s3.transcribe_segments([ok, garbage], cfg, Path("."))
    # x_cer は読みレベル (コンニチハイイテンキデスネ ⇄ ...キダ)。表層値は x_cer_surface に残る
    assert ok.scores["x_cer"] == round(
        s3._cer(s3.reading_key("今日はいい天気ですね"), s3.reading_key("今日はいい天気だ")), 3
    )
    assert ok.scores["x_cer_surface"] == round(3 / 10, 3)  # 表層: 編集距離 3 / 最大長 10
    assert ok.scores["x_cer"] < ok.scores["x_cer_surface"]
    assert ok.scores["asr2_text"] == "今日はいい天気だ"
    assert garbage.scores.get("asr2_skipped") == "obvious_garbage" and "x_cer" not in garbage.scores


def test_max_x_cer_gate_excludes():
    r = mk("x", x_cer=0.5, mora_count=8, asr_conf=0.9, align_conf=0.9, seg_dur=1.0)
    r.text = "あいうえお"
    cfg = cfg_with({"max_x_cer": 0.3})
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "excluded" and "max_x_cer" in (r.dropped_by or [])


# ------------------------------------------------------------------ / 新ゲート配線
def test_token_stats_from_probs():
    st = s3._token_stats([0.9, 0.8, 0.2])
    assert st["token_prob_min"] == 0.2
    assert 0.2 <= st["token_prob_p10"] <= 0.8  # 下位 1 分位は min と平均の間に収まる
    assert st["token_nll"] > 0


@pytest.mark.parametrize(
    "score_key",
    [
        "x_cer",
        "token_prob_min",
        "token_nll",
        "tag_hit",
        "mos",
        "ctc_align_score",
        "unaligned_edge_ratio",
        "lm_ppl",
        "residual_symbols",
        "text_tail_open",
    ],
)
def test_new_scores_are_gated(score_key):
    # token_prob_p10 は可視化専用 (ゲートは min / nll) なので CHECKS に入らない
    assert score_key in {s for _, s, _ in s6.CHECKS}


# ------------------------------------------------------------------ LM レビュー閾値
def test_review_lm_ppl_routes_to_pending():
    r = mk("l", lm_ppl=500.0, mora_count=8, asr_conf=0.9, align_conf=0.9, seg_dur=1.0)
    r.text = "あいうえお"
    cfg = cfg_with({"review_lm_ppl": 100.0})
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "pending_review" and "lm_ppl" in (r.review_reason or "")


def test_max_lm_ppl_gate_excludes():
    r = mk("l", lm_ppl=500.0, mora_count=8, asr_conf=0.9, align_conf=0.9, seg_dur=1.0)
    r.text = "あいうえお"
    cfg = cfg_with({"max_lm_ppl": 200.0})
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert "max_lm_ppl" in (r.dropped_by or [])


# ------------------------------------------------------------------ 多様性選別
def _rec(utt, text, q, src="x.mp3", f0=None, rate=None):
    sc = {"mora_count": 8, "quality_score": q, "asr_conf": 0.9, "snr": 20.0}
    if f0 is not None:
        sc["f0_median_hz"] = f0
    if rate is not None:
        sc["mora_rate"] = rate
    r = mk(utt, **sc)
    r.text = text
    r.src = src
    return r


def _div_cfg2(enabled=True, protect=False, **over):
    base = {
        "enabled": enabled,
        "metric": "phoneme",
        "sim_threshold": 0.6,
        "protect_coverage": protect,
    }
    base.update(over)
    return cfg_with({}, **{"diversity": base})


def _judge(recs, cfg):
    s6.evaluate_checks(recs, cfg)
    s5.dedup(recs, cfg)
    s5.diversity_select(recs, cfg)
    s6.finalize_statuses(recs)


def test_diversity_gain_and_target_selects_coverage_king():
    # λ=0 なら高品質の短い A が採られるところ、λ>0 + target_count=1 では trigram 供給量の多い B が採られる
    A = _rec("A", "あいうえお", 4.0)
    B = _rec("B", "かきくけこさしすせそたちつてと", 3.5)
    _judge([A, B], _div_cfg2(gain_lambda=100.0, target_count=1))
    assert B.status == "adopted"
    assert A.status == "excluded" and "diversity:target_size" in (A.dropped_by or [])


def test_diversity_lambda_zero_keeps_quality_order():
    A = _rec("A", "あいうえお", 4.0)
    B = _rec("B", "かきくけこさしすせそたちつてと", 3.5)
    _judge([A, B], _div_cfg2(target_count=1))  # λ=0 なら品質順で A のみ
    assert A.status == "adopted" and "diversity:target_size" in (B.dropped_by or [])


def test_diversity_source_cap():
    recs = [
        _rec("S1", "今日はいい天気ですね", 4.0, src="same.mp3"),
        _rec("S2", "猫が寝転んだ", 3.9, src="same.mp3"),
        _rec("S3", "明日は雨でしょう", 3.8, src="same.mp3"),
        _rec("S4", "りんごをたべました", 3.7, src="same.mp3"),
        _rec("O1", "電車が遅延しています", 3.6, src="other.mp3"),
    ]
    _judge(recs, _div_cfg2(max_source_share=0.4))  # cap = ceil(0.4*5) = 2 本/ソース
    same = [r for r in recs if r.src == "same.mp3"]
    assert sum(r.status == "adopted" for r in same) == 2
    capped = [r for r in same if r.status == "excluded"]
    assert len(capped) == 2
    assert all("diversity:cap:source:same.mp3" in (r.dropped_by or []) for r in capped)
    assert recs[4].status == "adopted"  # 別ソースは影響を受けない


def test_diversity_prosody_bucket_cap():
    # F0 三分位で「低 F0」ビンに 3 本固まる構成 (100,100,100 / 200 / 300)。
    # シェア上限 0.4 (cap=ceil(0.4*5)=2) で、ビン内最低品質の 1 本だけが整流化対象になる
    recs = [
        _rec("P1", "今日はいい天気ですね", 4.0, f0=100.0, rate=7.0),
        _rec("P2", "猫が寝転んだ", 3.9, f0=100.0, rate=7.0),
        _rec("P3", "明日は雨でしょう", 3.8, f0=100.0, rate=7.0),
        _rec("P4", "りんごをたべました", 3.7, f0=200.0, rate=7.0),
        _rec("P5", "電車が遅延しています", 3.6, f0=300.0, rate=7.0),
    ]
    _judge(recs, _div_cfg2(max_bucket_share=0.4))
    assert sum(r.status == "adopted" for r in recs) == 4
    dropped = [r for r in recs if r.status == "excluded"]
    assert len(dropped) == 1 and dropped[0].utt_id == "P3"  # 低 F0・低品質の 1 本だけが間引かれる
    assert any(x.startswith("diversity:cap:prosody:") for x in (dropped[0].dropped_by or []))


def test_diversity_cap_protects_sole_phoneme_supplier():
    # protect_coverage=true なら cap 超過でも「ある音素の唯一の供給源」は落ちない。
    # P3 の「ヴぁいぶぁい」は pyopenjtalk 読みが a/i のみで、P1(子音+モーラ音素)/P2 と別音素を供給する
    recs = [
        _rec("P1", "今日はいい天気ですね", 4.0, f0=100.0, rate=7.0),
        _rec("P2", "猫が寝転んだ", 3.9, f0=100.0, rate=7.0),
        _rec("P3", "ヴぁいぶぁい", 3.8, f0=100.0, rate=7.0),
        _rec("P4", "りんごをたべました", 3.7, f0=200.0, rate=7.0),
        _rec("P5", "電車が遅延しています", 3.6, f0=300.0, rate=7.0),
    ]
    _judge(recs, _div_cfg2(protect=True, max_bucket_share=0.4))
    assert all(r.status == "adopted" for r in recs)  # cap=2 だが供給源保護で P3 も救済


# ------------------------------------------------------------------ text_overrides 再適用
def test_text_override_recomputes_derived_scores(tmp_path):
    from make_dataset.overrides import apply_text_overrides

    r = mk("u1", seg_dur=2.0, mora_count=1, asr_conf=0.9, align_conf=0.9)
    r.end = 2.0  # 実音声長 2 秒 (mora_rate の分母は end-start)
    r.text = "あ"
    (tmp_path / "text_overrides.jsonl").write_text(
        json.dumps({"utt_id": "u1", "text": "猫が寝転んだ"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert apply_text_overrides([r], tmp_path) == 1
    assert r.text == "猫が寝転んだ"
    assert r.scores["mora_count"] > 1 and r.scores["char_count"] == 6
    assert r.scores["mora_rate"] == round(r.scores["mora_count"] / 2.0, 2)
    assert r.scores["text_override"] is True


def test_status_override_beats_failed_checks(tmp_path):
    from make_dataset.overrides import apply_text_overrides

    r = mk("u2", seg_dur=1.0, mora_count=5, asr_conf=0.9, align_conf=0.9)
    r.text = "あいうえお"
    (tmp_path / "text_overrides.jsonl").write_text(
        json.dumps({"utt_id": "u2", "status": "adopt"}) + "\n", encoding="utf-8"
    )
    apply_text_overrides([r], tmp_path)
    cfg = cfg_with({"min_chars": 999})  # 全件 min_chars で落ちる設定でも…
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "adopted" and r.scores.get("manual_override") is True  # 人手判定が優先


def test_status_override_review_and_exclude(tmp_path):
    from make_dataset.overrides import apply_text_overrides

    r = mk("u3", mora_count=5)
    r.text = "あいうえお"
    (tmp_path / "text_overrides.jsonl").write_text(
        json.dumps({"utt_id": "u3", "status": "review"}) + "\n", encoding="utf-8"
    )
    apply_text_overrides([r], tmp_path)
    s6.evaluate_checks([r], cfg_with({}))
    s6.finalize_statuses([r])
    assert r.status == "pending_review"


def test_repaired_audio_never_auto_adopted():
    r = mk("rp", mora_count=5, repaired=True)
    r.text = "あいうえお"
    s6.evaluate_checks([r], cfg_with({}))
    s6.finalize_statuses([r])
    assert r.status == "pending_review" and r.review_reason == "repaired"


def test_report_lists_pending_with_override_howto(tmp_path):
    """人手レビューページ (旧 review.html) をやめたので、見送り一覧と text_overrides の手順は
    report.html 側に出す。ここが消えると「見送りをどう確定するか」がどこにも書かれない。"""
    r = mk("q1", asr_conf=0.4)
    r.text = "あいうえお"
    r.status = "pending_review"
    r.review_reason = "dict_data"
    s6.write_report([r], tmp_path, cfg_with({}))
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "pending_review" in html and "dict_data" in html and "q1" in html
    assert "text_overrides.jsonl" in html and "status" in html  # 確定手順を同じページに載せる
    assert "見送り" in html
    assert not (tmp_path / "review.html").exists()  # 旧レビュー用ページは作らない


def test_cli_exposes_only_run_and_rescore():
    """人手レビュー用の review コマンド (review.html の HTTP 配信) を廃止した回帰ガード。

    判定は機械、人が動かすのは run / rescore だけ、という運用を CLI から外さないための歯止め。
    """
    from make_dataset.cli import app

    names = {c.callback.__name__ for c in app.registered_commands if c.callback is not None}
    assert names == {"run", "rescore"}


# ------------------------------------------------------------------ レポートの多様性メトリクス
def test_report_contains_diversity_metrics(tmp_path):
    A = _rec("A", "今日はいい天気ですね", 4.0)
    B = _rec("B", "今日はいい天気でした", 3.0)  # A と似た内容 → 多様性で落ちる
    C = _rec("C", "猫が寝転んだ", 2.0)
    _judge([A, B, C], _div_cfg2(enabled=True, protect=False))
    s6.write_report([A, B, C], tmp_path, _div_cfg2())
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "多様性メトリクス" in html and "音素三量被覆率" in html


def test_report_shows_tail_openness_distribution(tmp_path):
    """severity 2 は落として終わり、にしない。分布と一覧を report に出す (max_text_tail_open=0
    へ rescore し直すための判断材料)。"""
    A = _rec("A", "今日はいい天気ですね", 4.0)
    A.scores["text_tail_open"] = 0
    B = _rec("B", "してほしいことがあったら。", 3.0)
    B.scores.update({"text_tail_open": 1, "text_tail_open_reason": "cform:仮定形"})
    C = _rec("C", "お客様の体温と混じっ", 2.0)
    C.scores.update({"text_tail_open": 2, "text_tail_open_reason": "no_sent_end"})
    cfg = _div_cfg2()
    cfg.data["thresholds"]["max_text_tail_open"] = 1
    _judge([A, B, C], cfg)
    s6.write_report([A, B, C], tmp_path, cfg)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "文末の開き" in html
    assert "no_sent_end" in html and "cform:仮定形" in html
    assert "max_text_tail_open = 1" in html


# ------------------------------------------------------------------ x_cer は読みレベル
def test_reading_key_absorbs_notation_diff():
    """x_cer は表層でなく発音で取る。句読点・同音異字・文末の伸ばしは「一致」扱いにする。"""
    assert s3.reading_key("モフモフの番点。") == s3.reading_key("モフモフのバンテン")
    assert s3.reading_key("どうぞ。") == s3.reading_key("どうぞー")
    assert s3.reading_key("グリグリ。") == s3.reading_key("グリグリ")
    assert s3.reading_key("お客様の体温と混じっ") == s3.reading_key("お客様の体温とまじっ")
    # 語中の長音は保持するので おばあさん/おばさん は潰れない (文末の ー だけ落とす)
    assert s3.reading_key("おばあさん") != s3.reading_key("おばさん")
    # 頭字語の句点は g2p が語ごと読むのをやめるので先に消す (実測で「S.F.とか。」が誤検出された)
    assert s3.reading_key("S.F.とか。") == s3.reading_key("SFとか")
    assert s3.reading_key("") == ""
    assert s3.reading_key("—…、。") == ""  # 記号のみは空 (退避経路で落ちない)


def test_x_cer_gate_passes_homophone_and_fails_real_mismatch(monkeypatch):
    """別系統 ASR が同音異字でしか違わない発話は採用され、本当に食い違う発話は落ちる。"""
    import make_dataset.stage0_ingest as s0

    monkeypatch.setattr(s0, "load_cached", lambda r, c, sr: __import__("numpy").zeros(16000))
    same_sound = {"asr": "モフモフの番点。", "asr2": "もふもふのバンテン"}
    different = {"asr": "モフモフの番点。", "asr2": "ゆっくり息を吸って"}

    state = {"pair": True}

    def fake(audio, cfg, prefix="asr"):
        pair = same_sound if state["pair"] else different
        return pair[prefix], 0.9, {}

    monkeypatch.setattr(s3, "_asr_whisper", fake)
    cfg = cfg_with(
        {"max_x_cer": 0.3},
        **{
            "asr.backend": "faster-whisper",
            "asr.model": "tiny",
            "asr2.enabled": True,
            "asr2.backend": "faster-whisper",
            "asr2.model": "small",
        },
    )
    good, bad = mk("g0", seg_dur=1.5), mk("b0", seg_dur=1.5)
    state["pair"] = True
    s3.transcribe_segments([good], cfg, Path("."))
    state["pair"] = False
    s3.transcribe_segments([bad], cfg, Path("."))
    assert good.scores["x_cer"] == 0.0 and good.scores["x_cer_surface"] > 0.3
    assert bad.scores["x_cer"] > 0.3
    s6.evaluate_checks([good, bad], cfg)
    s6.finalize_statuses([good, bad])
    assert "max_x_cer" not in (good.dropped_by or []) and good.status == "adopted"
    assert "max_x_cer" in (bad.dropped_by or []) and bad.status == "excluded"


def test_refresh_asr2_scores_adds_x_cer_without_rerun(monkeypatch, tmp_path):
    """run 済みの manifest に [asr2] を後から有効化しても、run を回さずに x_cer を引ける。

    第 2 ASR の再実行は高いので、rescore で足りないスコアだけ引き直す (rescore と同じ考え方)。
    同じ `X_CER_ALGO` を持つレコードは触らないこと。版数が違うものだけ引き直す。
    """
    import make_dataset.stage0_ingest as s0

    monkeypatch.setattr(s0, "load_cached", lambda r, c, sr=44100: __import__("numpy").zeros(16000))
    monkeypatch.setattr(
        s3, "_asr_whisper", lambda audio, cfg, prefix="asr": ("もふもふの番点", 0.8, {})
    )
    cfg = cfg_with(
        {},
        **{"asr2.enabled": True, "asr2.backend": "faster-whisper", "asr2.model": "small"},
    )
    old = mk("u0", seg_dur=1.5)
    old.text = "モフモフの番点。"
    garbage = mk("u1", seg_dur=0.2)  # 「明らかなゴミ」は run と同じくスキップ扱い
    garbage.text = "あ"
    assert s3.refresh_asr2_scores([old, garbage], cfg, tmp_path) == 1
    assert old.scores["x_cer"] == 0.0  # 読みレベルでは同音異字 = 一致
    assert old.scores["x_cer_surface"] > 0.0  # 表層では「。」ぶんズレる
    assert old.scores["x_cer_algo"] == s3.X_CER_ALGO
    assert garbage.scores["asr2_skipped"] == "obvious_garbage" and "x_cer" not in garbage.scores
    old.scores["x_cer"] = 0.99  # 既存値 (同じ版数の x_cer) は上書きしない
    assert s3.refresh_asr2_scores([old], cfg, tmp_path) == 0 and old.scores["x_cer"] == 0.99
    # reading_key を変えて X_CER_ALGO を上げたら、古い正規化の値は自動的に引き直される
    old.scores["x_cer_algo"] = s3.X_CER_ALGO - 1
    assert s3.refresh_asr2_scores([old], cfg, tmp_path) == 1 and old.scores["x_cer"] == 0.0


def test_review_x_cer_routes_to_review(tmp_path):
    """食い違いを機械的に落とさず見送りにする受け皿 (主 ASR が正しいケースを救う)。

    report.html の見送り一覧には第 2 系統の出力も出す。本文だけ見ても「どちらが間違ったか」は
    判断できないので、レビューページを廃止した今はレポート側がその役割を担う。
    """
    r = mk("x", x_cer=0.3, mora_count=8, seg_dur=1.0)
    r.text = "あいうえお"
    r.scores["asr2_text"] = "あいてお"
    cfg = cfg_with({"review_x_cer": 0.25})
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "pending_review" and "x_cer" in (r.review_reason or "")
    s6.write_report([r], tmp_path, cfg)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "第 2 系統: あいてお" in html and "x_cer" in html


def test_report_shows_x_cer_distribution(tmp_path):
    """ゲートで落として終わりにならないよう、第 2 系統の出力と並べて rescore の判断材料を出す。"""
    A = _rec("A", "今日はいい天気ですね", 4.0)
    A.scores.update(
        {"x_cer": 0.0, "x_cer_surface": 0.1, "asr2_text": "今日はいい天気ですね", "asr2_conf": 0.7}
    )
    B = _rec("B", "こっちいいと思いますよ", 3.0)
    B.scores.update(
        {"x_cer": 0.92, "x_cer_surface": 0.95, "asr2_text": "ごちそうさまでした", "asr2_conf": 0.3}
    )
    cfg = _div_cfg2()
    cfg.data["thresholds"]["max_x_cer"] = 0.45
    _judge([A, B], cfg)
    s6.write_report([A, B], tmp_path, cfg)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "x_cer" in html and "ごちそうさまでした" in html and "max_x_cer = 0.45" in html
    # 分布には第 2 系統の確信度も出す (食い違い帯の確信度が低ければ除外の証拠は弱い)
    assert "第2確信度 0.70" in html and "第2確信度 0.30" in html
    assert B.status == "excluded" and A.status == "adopted"
