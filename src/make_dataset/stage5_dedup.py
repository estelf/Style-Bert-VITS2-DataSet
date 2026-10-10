"""Stage 5: 重複排除 (安い順: テキスト完全一致 → テキストファジィ → 音素列 → 音声指紋). 代表1本を自動採用.

- F3: 全ペア O(N²) を取らず、MinHash/LSH (テキスト・音素) と band LSH (音声内容指紋) で候補ペアを
  生成してから厳密判定する (DESIGN.md Stage 5「MinHash/LSH で O(N²) を回避」)。
- D8: scores["fp_sig"] (Stage 2 が保存した MFCC フレーム列の MinHash 内容指紋。話者ベクトルでは
  ない) の一致率で、テキストが微妙に違うズレ切り出し重複を検出する (reason=fuzzy_audio)。
- B4: union-find のマージ時にグループ理由を正しく引き継ぐ (優先度が高い理由を採用)。
- B5: 代表は「閾値チェックを通過したもの」から quality → asr_conf → snr で選ぶ。代表候補が後で
  閾値落ちしてグループごと消えることを防ぐ (evaluate_checks が先に走っている前提)。
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .manifest import Record

_REASON_PRIORITY = {"exact": 0, "fuzzy_text": 1, "fuzzy_audio": 2}

# カタカナ読み → 音素風トークン列 (ドラフト用の決定論的変換)
_VOW = dict(zip("アイウエオ", "aiueo", strict=True))
_ROWS = [
    ("カキクケコ", "k"),
    ("ガギグゲゴ", "g"),
    ("サシスセソ", "s"),
    ("ザジズゼゾ", "z"),
    ("タチツテト", "t"),
    ("ダヂヅデド", "d"),
    ("ナニヌネノ", "n"),
    ("ハヒフヘホ", "h"),
    ("バビブベボ", "b"),
    ("パピプペポ", "p"),
    ("マミムメモ", "m"),
    ("ヤユヨ", "y"),
    ("ラリルレロ", "r"),
    ("ワヰヱヲ", "w"),
]
# 母音はローマ字 (_VOW と表記を揃える)。ヤ行だけ 3 モーラで "aiueo" より短いので strict=False
# で切り詰め、イ・エ に相当するモーラを作らない (C12)。
_MORA = {k: (c, v) for row, c in _ROWS for k, v in zip(row, "aiueo", strict=False)}
_MORA.update({k: ("", v) for k, v in _VOW.items()})
# C12: 小文字・促音・長音を脱落させない (短文での誤重複除外リスク)。pyopenjtalk-plus は拗音を
# 大文字へ展開するが、ッ・ー・ィ等はそのまま残るので明示的にトークン化する。
_MORA.update(
    {
        "ィ": ("", "i"),
        "ゥ": ("", "u"),
        "ェ": ("", "e"),
        "ォ": ("", "o"),
        "ャ": ("y", "a"),
        "ュ": ("y", "u"),
        "ョ": ("y", "o"),
    }
)


def _kana_to_phonemes(kana: str) -> list[str]:
    out: list[str] = []
    last_v = ""
    for c in kana:
        if c in _MORA:
            cons, v = _MORA[c]
            out.append(cons + v)
            last_v = v
        elif c == "ン":
            out.append("N")
        elif c == "ッ":
            out.append("cl")  # 促音: 独立トークン (「マツト」と「マット」を区別できる)
        elif c == "ー" and last_v:
            out.append(last_v)  # 長音: 直前モーラの母音の繰り返しとして扱う
    return out


def phoneme_tokens(text: str) -> list[str]:
    """テキスト → 音素トークン列 (G2 被覆率レポートと音素 Jaccard の共通基盤)."""
    import pyopenjtalk as pj  # pyopenjtalk-plus

    try:
        pron = "".join(f["pron"] for f in pj.run_frontend(text))
    except Exception:  # noqa: BLE001
        # 入力により多种の例外がありうる。指紋が引けない場合は空列にして重複判定だけを弱める
        return []
    return _kana_to_phonemes(pron)


def _shingles(text: str, n: int = 2) -> set:
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 0))} if len(text) >= n else {text}


def _phoneme_trigrams(text: str) -> set:
    ph = phoneme_tokens(text)
    return {"".join(ph[i : i + 3]) for i in range(max(len(ph) - 2, 0))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------------------------------------------------------------- F3: LSH 候補ペア生成

_PRIME = (1 << 61) - 1


def _token_hashes(tokens) -> np.ndarray:
    import hashlib

    def one(t):
        b = t if isinstance(t, bytes) else t.encode("utf-8")
        return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")

    return np.fromiter((one(t) for t in tokens), dtype=np.uint64)


def minhash_signature(tokens, num_perm: int = 32, seed: int = 42) -> list[int]:
    """トークン集合の MinHash 署名 (seed 固定 → 実行間・rescore 間で比較可能). D8 の内容指紋に使う."""
    if not tokens:
        return []
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64)
    h = _token_hashes(tokens).astype(object)
    sig = np.min((a.astype(object) * h[:, None] + b.astype(object)) % _PRIME, axis=0)
    return [int(v) for v in sig]


def _minhash_candidate_pairs(
    sets: list[set], num_perm: int = 96, rows: int = 8, seed: int = 0
) -> set[tuple[int, int]]:
    """集合群から MinHash band 衝突で候補ペアを出す (Jaccard 高しきい値なら十分正確)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64)
    sigs: list[np.ndarray | None] = []
    for s in sets:
        if not s:
            sigs.append(None)
            continue
        h = _token_hashes(s).astype(object)
        sig = np.min((a.astype(object) * h[:, None] + b.astype(object)) % _PRIME, axis=0)
        sigs.append(sig)
    buckets: dict[tuple, list[int]] = {}
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for band in range(num_perm // rows):
            key = (band, tuple(sig[band * rows : (band + 1) * rows].tolist()))
            buckets.setdefault(key, []).append(i)
    pairs: set[tuple[int, int]] = set()
    for idxs in buckets.values():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                pairs.add((min(idxs[x], idxs[y]), max(idxs[x], idxs[y])))
    return pairs


def _passes(r: Record) -> bool:
    return not (r.scores.get("failed_checks") or r.scores.get("errors"))


def dedup(records: list[Record], cfg: Config) -> None:
    """重複グループを決定し、代表1本以外に dup_group_id / dup_reason を記録する (確定は finalize)."""
    t_jac, p_jac, a_jac = (
        cfg.get("dedup.text_jaccard"),
        cfg.get("dedup.phoneme_jaccard"),
        cfg.get("dedup.audio_jaccard"),
    )
    for r in records:  # rescore 再実行時に前回のグループ判定を持ち越さない
        r.dup_group_id = r.dup_reason = None
    texts = [r.text for r in records]
    sh = [_shingles(t) if t else set() for t in texts]
    ph = [_phoneme_trigrams(t) if t else set() for t in texts]

    # 1) 候補ペア検出 (F3: 全ペア比較ではなく LSH で候補を出す。完全一致はハッシュで O(N))
    pairs: list[tuple[int, int, str]] = []
    by_text: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        if t:
            by_text.setdefault(t, []).append(i)
    for idxs in by_text.values():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                pairs.append((idxs[x], idxs[y], "exact"))

    for i, j in sorted(_minhash_candidate_pairs(sh, seed=2)):
        if texts[i] != texts[j] and _jaccard(sh[i], sh[j]) >= t_jac:
            pairs.append((i, j, "fuzzy_text"))
    for i, j in sorted(_minhash_candidate_pairs(ph, seed=3)):
        if ph[i] and _jaccard(ph[i], ph[j]) >= p_jac:
            pairs.append((i, j, "fuzzy_audio"))

    # D8: 内容指紋 (fp_sig) 一致によるズレ切り出し重複。band 衝突候補のみ厳密比較する
    sigs = [r.scores.get("fp_sig") or [] for r in records]
    if any(len(s) == 32 for s in sigs):
        seen_pairs = {(i, j) for i, j, _ in pairs}
        buckets: dict[tuple, list[int]] = {}
        for i, sig in enumerate(sigs):
            if not sig:
                continue
            for band in range(4):
                buckets.setdefault((band, tuple(sig[band * 8 : (band + 1) * 8])), []).append(i)
        for idxs in buckets.values():
            for x in range(len(idxs)):
                for y in range(x + 1, len(idxs)):
                    i, j = min(idxs[x], idxs[y]), max(idxs[x], idxs[y])
                    if (i, j) in seen_pairs:
                        continue
                    agree = float(np.mean(np.array(sigs[i]) == np.array(sigs[j])))
                    if agree >= a_jac:
                        pairs.append((i, j, "fuzzy_audio"))

    # 2) union-find でグループ化 (B4: 優先度の高い理由をグループへ正しく引き継ぐ)
    parent = list(range(n := len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reasons: dict[int, str] = {}
    for i, j, r in sorted(pairs, key=lambda p: _REASON_PRIORITY[p[2]]):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[rj] = ri
        cand = [x for x in (reasons.pop(ri, None), reasons.pop(rj, None), r) if x]
        reasons[ri] = min(cand, key=lambda x: _REASON_PRIORITY[x])

    # 3) グループ代表1本のみ自動採用 (B5: 閾値通過者を優先し、quality → asr_conf → snr の決定式)
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
                _passes(
                    records[i]
                ),  # B5: 閾値を通過しているものを代表に (全員落ちのグループを作らない)
                records[i].scores.get("quality_score") or 0,
                records[i].scores.get("asr_conf") or 0,
                records[i].scores.get("snr") or -99,
            ),
        )
        for i in idxs:
            r = records[i]
            r.dup_group_id = gid
            if i != rep:
                r.dup_reason = reason  # 除外確定 (dropped_by への反映) は finalize で行う
