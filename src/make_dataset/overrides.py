"""text_overrides.jsonl の再適用 — 人手レビューページを作らずに判定を確定する唯一の受け皿。

機械の自動判定を、再 run なしで上書きするために使う:
- `Data/<話者>/text_overrides.jsonl` に 1 行 1 レコードで置く (utt_id キー):
    {"utt_id": "ab12cd34ef567890-003", "text": "修正後のセリフ"}          # テキスト修正
    {"utt_id": "...", "status": "adopted"}                                # 判定の上書き
  両方同時でもよい。`make_dataset rescore Data/<話者>` で再適用され、
  テキスト修正は G2P・モーラ・話速・テキストフィルタ系スコアを**すべて引き直す**。
- 中身の一覧と書き方は `report.html` の「見送り (pending_review)」と「見方・凡例 6」に出す
  (音声プレイヤー付きレビューページは作らない。聴くなら `.cache/<utt_id>.flac` を直接開く)。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# 旧レビュー UI 時代の別名 (review / adopt / exclude) も受け付ける (既存の overrides ファイルを壊さない)
_STATUS_ALIASES = {
    "adopt": "adopted",
    "adopted": "adopted",
    "exclude": "excluded",
    "excluded": "excluded",
    "review": "pending_review",
    "pending_review": "pending_review",
}


def apply_text_overrides(records, outdir: Path, cfg=None) -> int:
    """text_overrides.jsonl があれば読み、テキスト/判定の上書きをレコードへ反映する (件数を返す)。

    呼ぶ順序は重要: dedup / 多様性選別・話速中央値より**前**に適用しないと、修正後テキストでの
    重複判定や話速計算が古いままになる (cli._judge_and_write の先頭で呼ばれる)。
    """
    path = Path(outdir) / "text_overrides.jsonl"
    if not path.exists():
        return 0

    from .stage3_asr import g2p_check, normalize_text, text_features

    expand = True if cfg is None else bool(cfg.get("text.symbol_readings"))
    by_id = {r.utt_id: r for r in records}
    n = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no} が JSON として読めない: {e}") from e
        r = by_id.get(d.get("utt_id"))
        if r is None:
            continue  # 別ランの utt_id が残っていても黙って無視 (rescore を止めない)
        if d.get("text") is not None:
            t = normalize_text(str(d["text"]), expand_symbols=expand)
            r.text = t
            g = g2p_check(t) if t else {"g2p_ok": False, "unknown_tokens": -1, "mora_count": 0}
            dur = r.end - r.start
            r.scores.update(g)
            r.scores["mora_rate"] = (
                round(g["mora_count"] / dur, 2) if g["mora_count"] and dur > 0 else 0.0
            )
            r.scores["repeat_block"] = bool(re.search(r"(.{3,})\1{2,}", t))
            r.scores.update(text_features(t))
            r.scores["text_override"] = True
        st = d.get("status")
        if st:
            if st not in _STATUS_ALIASES:
                raise ValueError(
                    f"{path}:{line_no} status={st!r} が不正 (使用可: {sorted(_STATUS_ALIASES)})"
                )
            r.scores["status_override"] = _STATUS_ALIASES[st]
        n += 1
    return n
