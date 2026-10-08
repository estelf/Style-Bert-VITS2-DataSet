"""Stage 5: 重複排除 (安い順: テキスト完全一致 → テキストファジィ → 音素列). 代表1本を自動採用.

ドラフト段階の注記: ④ズレ切り出し音声フィンガープリント (Chromaprint 系) は未実装。
"""

from __future__ import annotations

from .config import Config
from .manifest import Record

_REASON_PRIORITY = {"exact": 0, "fuzzy_text": 1, "fuzzy_audio": 2}

# カタカナ読み → 音素風トークン列 (ドラフト用の決定論的変換。拗音は母音トークンに吸収される)
_VOW = dict(zip("アイウエオ", "aiueo"))
_ROWS = [
    ("カキクケコ", "k"), ("ガギグゲゴ", "g"), ("サシスセソ", "s"), ("ザジズゼゾ", "z"),
    ("タチツテト", "t"), ("ダヂヅデド", "d"), ("ナニヌネノ", "n"), ("ハヒフヘホ", "h"),
    ("バビブベボ", "b"), ("パピプペポ", "p"), ("マミムメモ", "m"), ("ヤユヨ", "y"),
    ("ラリルレロ", "r"), ("ワヰヱヲ", "w"),
]
_MORA = {k: (c, v) for row, c in _ROWS for k, v in zip(row, "アイウエオ")}
_MORA.update({k: ("", v) for k, v in _VOW.items()})


def _kana_to_phonemes(kana: str) -> list[str]:
    out = []
    for c in kana:
        if c in _MORA:
            cons, v = _MORA[c]
            out.append(cons + v)
        elif c == "ン":
            out.append("N")
        # ッ(促音)・ー(長音) はドラフトでは無視 (決定的変換なので Jaccard 比較には効く)
    return out


def _shingles(text: str, n: int = 2) -> set:
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 0))} if len(text) >= n else {text}


def _phoneme_trigrams(text: str) -> set:
    import pyopenjtalk as pj  # pyopenjtalk-plus

    try:
        pron = "".join(f["pron"] for f in pj.run_frontend(text))
    except Exception:
        return set()
    ph = _kana_to_phonemes(pron)
    return {"".join(ph[i : i + 3]) for i in range(max(len(ph) - 2, 0))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def dedup(records: list[Record], cfg: Config) -> None:
    """重複グループを決定し、代表1本以外に dup_group_id / dup_reason / dropped_by を記録する."""
    t_jac, p_jac = cfg.get("dedup.text_jaccard"), cfg.get("dedup.phoneme_jaccard")
    texts = [r.text for r in records]
    sh = [_shingles(t) for t in texts]
    ph = [_phoneme_trigrams(t) if t else set() for t in texts]

    # 1) 候補ペア検出 (安い順: 完全一致 → ファジィテキスト → 音素列)
    pairs: list[tuple[int, int, str]] = []
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            if texts[i] and texts[i] == texts[j]:
                pairs.append((i, j, "exact"))
            elif _jaccard(sh[i], sh[j]) >= t_jac:
                pairs.append((i, j, "fuzzy_text"))
            elif ph[i] and _jaccard(ph[i], ph[j]) >= p_jac:
                pairs.append((i, j, "fuzzy_audio"))

    # 2) union-find でグループ化 (優先度の高い理由を採用)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reasons: dict[int, str] = {}
    for i, j, r in sorted(pairs, key=lambda p: _REASON_PRIORITY[p[2]]):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
            reasons.setdefault(ri, r)
        elif find(i) == find(j):
            cur = reasons.get(find(i))
            if cur and _REASON_PRIORITY[r] < _REASON_PRIORITY[cur]:
                reasons[find(i)] = r

    # 3) グループ代表1本のみ自動採用 (基準: quality → asr_conf → snr の決定式)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for gi, (root, idxs) in enumerate(sorted(groups.items())):
        if len(idxs) < 2:
            continue
        gid = f"dup{gi:03d}"
        reason = reasons.get(root, "duplicate")
        rep = max(
            idxs,
            key=lambda i: (
                records[i].scores.get("quality_score") or 0,
                records[i].scores.get("asr_conf") or 0,
                records[i].scores.get("snr") or -99,
            ),
        )
        for i in idxs:
            r = records[i]
            r.dup_group_id = gid
            if i != rep:
                r.status = "excluded"
                r.dup_reason = reason
                r.dropped_by = f"dedup:{reason}"
