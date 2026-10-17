"""report.html のデザイン確認用サンプル生成 (音声・モデル不要の合成レコード)。

実データで `run` せずにレポートの見た目 (凡例・感度表・分布・見送り一覧) を確認するための道具。
clean なスコアで作り、ランダムに 1 ゲートだけ割らせて「除外内訳」と「ゲート別感度」が
それらしく並ぶようにする (adopted 0 件だと分布が全部空になって見えない)。

    uv run python scripts/preview_report.py            # Data/_report_preview/report.html
    open Data/_report_preview/report.html              # ブラウザで開く (単一 HTML・外部依存なし)

中身は合成データで、実コーパスの実測分布は反映していない (デザイン確認専用)。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from make_dataset import stage6_output as s6
from make_dataset.config import Config
from make_dataset.manifest import Record

TEXTS = [
    "今日はいい天気ですね。",
    "お客様の体温と混じっ",
    "してほしいことがあったら。",
    "グリグリ。",
    "ゆっくり押して。",
    "モフモフの番点。",
    "どうしたの。",
    "電車が遅延しています。",
    "りんごをたべました",
    "あのー、えーと、つまりですね。",
    "<script>alert(1)</script>",  # HTML エスケープの確認も兼ねる
    "氷柱が落ちるほどの寒さでした。",
    "ではまた明日お会いしましょう。",
    "ゆっくり息を吸って。",
]
SRCS = ["korisu_01.mp3", "yuka_long.mp4", "nashu.mp3", "a&b<x>.mp3"]

# 割らせるゲート → (clean 値, 割らせた値)。1 レコードにつき 1 ゲートだけ割らせる。
BREAKS = {
    "min_snr_db": ("snr", lambda r: r.uniform(6, 34), lambda r: r.uniform(0.5, 4.9)),
    "min_asr_conf": ("asr_conf", lambda r: r.uniform(0.35, 0.98), lambda r: r.uniform(0.05, 0.29)),
    "min_quality_pesq": ("pesq", lambda r: r.uniform(1.8, 3.4), lambda r: r.uniform(1.0, 1.49)),
    "max_reverb_score": (
        "reverb_score",
        lambda r: r.uniform(0.1, 0.45),
        lambda r: r.uniform(0.51, 0.9),
    ),
    "min_speaker_sim": (
        "speaker_sim",
        lambda r: r.uniform(0.6, 0.95),
        lambda r: r.uniform(0.2, 0.49),
    ),
    "max_clip_ratio": (
        "clip_ratio",
        lambda r: r.uniform(0.0, 0.0015),
        lambda r: r.uniform(0.0021, 0.02),
    ),
    "max_digital_silence": (
        "digital_silence_ratio",
        lambda r: r.uniform(0.0, 0.4),
        lambda r: r.uniform(0.51, 0.9),
    ),
    "min_voiced_ratio": (
        "voiced_ratio",
        lambda r: r.uniform(0.2, 0.8),
        lambda r: r.uniform(0.01, 0.09),
    ),
    "max_lm_ppl": ("lm_ppl", lambda r: r.uniform(5, 150), lambda r: r.uniform(210, 900)),
    "max_x_cer": ("x_cer", lambda r: r.uniform(0.0, 0.2), lambda r: r.uniform(0.46, 0.95)),
    "min_token_prob": (
        "token_prob_min",
        lambda r: r.uniform(0.3, 0.9),
        lambda r: r.uniform(0.01, 0.24),
    ),
    "max_token_nll": ("token_nll", lambda r: r.uniform(0.0, 0.25), lambda r: r.uniform(0.31, 1.2)),
    "max_filler_ratio": (
        "filler_ratio",
        lambda r: r.uniform(0.0, 0.4),
        lambda r: r.uniform(0.61, 0.95),
    ),
    "min_kana_ratio": (
        "kana_ratio",
        lambda r: r.uniform(0.6, 0.99),
        lambda r: r.uniform(0.05, 0.49),
    ),
    "max_mora_rate_dev": (
        "mora_rate_dev",
        lambda r: r.uniform(0.0, 0.9),
        lambda r: r.uniform(1.21, 3.0),
    ),
    "min_hf_energy_ratio": (
        "hf_energy_ratio",
        lambda r: r.uniform(0.001, 0.02),
        lambda r: r.uniform(0, 4e-4),
    ),
    "max_dc_offset": (
        "dc_offset_abs",
        lambda r: r.uniform(0.0, 0.015),
        lambda r: r.uniform(0.021, 0.2),
    ),
}
THRESHOLDS = {
    "min_seg_sec": 0.5,
    "max_seg_sec": 12.0,
    "min_snr_db": 5.0,
    "min_asr_conf": 0.3,
    "min_align_conf": 0.3,
    "min_mora_rate": 2.0,
    "max_mora_rate": 14.0,
    "max_mora_rate_dev": 1.2,
    "min_hf_energy_ratio": 0.0005,
    "max_reverb_score": 0.5,
    "max_jitter_local": 0.02,
    "min_quality_pesq": 1.5,
    "min_chars": 4,
    "max_filler_ratio": 0.6,
    "min_speaker_sim": 0.5,
    "min_kana_ratio": 0.5,
    "max_clip_ratio": 0.002,
    "max_dc_offset": 0.02,
    "max_digital_silence": 0.5,
    "min_voiced_ratio": 0.1,
    "max_x_cer": 0.45,
    "min_token_prob": 0.25,
    "max_token_nll": 0.3,
    "reject_repetition": True,
    "reject_alpha_only": True,
    "reject_music": True,
    "reject_audio_tags": True,
    "max_lm_ppl": 200.0,
    "max_text_tail_open": 1,
    "on_g2p_fail": "exclude",
    "on_unknown_tokens": "exclude",
}


def build(n: int = 300, seed: int = 7) -> tuple[list[Record], Config]:
    rng = random.Random(seed)
    cfg = Config.load(None)
    cfg.data["thresholds"] = dict(THRESHOLDS)
    cfg.data["diversity"] = {
        "enabled": True,
        "metric": "phoneme",
        "sim_threshold": 0.6,
        "protect_coverage": True,
        "gain_lambda": 0.0,
        "max_bucket_share": 0.0,
        "max_source_share": 0.0,
        "target_count": 0,
    }
    records: list[Record] = []
    for i in range(n):
        r = Record(
            utt_id=f"{i:016x}-00{i % 4}", src=SRCS[i % len(SRCS)], start=round(i * 9.1, 1), end=0.0
        )
        dur = rng.uniform(0.6, 9.0)
        r.end = r.start + dur
        r.text = TEXTS[i % len(TEXTS)]
        rate = rng.uniform(3.0, 8.5)
        tail = 2 if i % 101 == 0 else (1 if i % 29 == 0 else 0)
        s = {
            "seg_dur": round(dur, 2),
            "snr": 20.0,
            "hf_energy_ratio": 0.005,
            "reverb_score": 0.2,
            "jitter_local": 0.006,
            "pesq": 2.6,
            "asr_conf": 0.8,
            "align_conf": 0.8,
            "char_count": len(r.text),
            "mora_count": max(1, round(rate * dur)),
            "mora_rate": round(rate, 2),
            "mora_rate_dev": 0.3,
            "kana_ratio": 0.85,
            "filler_ratio": 0.1,
            "speaker_sim": 0.8,
            "clip_ratio": 0.0005,
            "dc_offset_abs": 0.005,
            "digital_silence_ratio": 0.1,
            "voiced_ratio": 0.5,
            "f0_median_hz": round(rng.uniform(110, 330), 1),
            "x_cer": 0.0,
            "x_cer_surface": round(rng.uniform(0.0, 0.3), 3),
            "asr2_conf": round(rng.uniform(0.3, 0.8), 2),
            "asr2_text": r.text if i % 4 else "ごちそうさまでした",
            "token_prob_min": 0.6,
            "token_nll": 0.05,
            "lm_ppl": 40.0,
            "tag_hit": False,
            "music_flag": False,
            "alpha_only": False,
            "repeat_block": False,
            "text_tail_open": tail,
            "text_tail_open_reason": {0: "", 1: "cform:仮定形", 2: "no_sent_end"}[tail],
            "g2p_ok": True,
            "unknown_tokens": 0,
            "yomi_bad_words": "",
            "limited": i % 41 == 0,
        }
        if rng.random() < 0.33:  # 1 ゲートだけ割らせて除外内訳に厚みを出す
            key = rng.choice(list(BREAKS))
            s[BREAKS[key][0]] = BREAKS[key][2](rng)
        if i % 83 == 0:  # 読めない語 (ユーザー辞書登録の決定材料)
            s.update({"g2p_ok": False, "unknown_tokens": 1, "yomi_bad_words": "䶮,邉"})
        if i % 47 == 3:
            s["tag_hit"] = True
        if i % 59 == 7:
            s["music_flag"] = True
        if i % 73 == 9:
            s["repeat_block"] = True
        if i % 101 == 5:
            s["stage1_skipped"] = "too_short"
        if i % 127 == 11:
            s["errors"] = "asr_failed: cuda oom"
        if i % 71 == 13:
            s["repaired"] = True  # 自動採用せず見送り行きになる
        r.scores.update(s)
        records.append(r)

    for i in range(0, 48, 4):  # 重複グループ (代表 1 本以外は dedup で落ちる)
        for j in range(i, i + 4):
            records[j].dup_group_id = f"dup{i // 4:03d}"
            if j != i:
                records[j].dup_reason = "phoneme"
    for i in range(60, 84):  # 多様性選別 (Stage 5b) の間引き
        records[i].diversity_reason = f"similar:{records[i - 1].utt_id}"
    for i in range(100, 108):
        records[i].diversity_reason = "cap:prosody:1-1"
    records[140].diversity_reason = "cap:source:korisu_01.mp3"
    records[141].diversity_reason = "target_size"

    s6.evaluate_checks(records, cfg)
    s6.finalize_statuses(records)
    return records, cfg


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Data/_report_preview")
    out.mkdir(parents=True, exist_ok=True)
    records, cfg = build()
    s6.write_report(records, out, cfg)
    n = {
        st: sum(1 for r in records if r.status == st)
        for st in ("adopted", "excluded", "pending_review")
    }
    print(f"{out / 'report.html'} を生成 ({(out / 'report.html').stat().st_size:,} bytes)")
    print("  " + " / ".join(f"{k}={v}" for k, v in n.items()))


if __name__ == "__main__":
    main()
