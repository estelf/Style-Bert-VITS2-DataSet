"""Stage 5: 重複排除 (安い順: テキスト完全一致 → テキストファジィ → 音素列 → 音声指紋). 代表1本を自動採用.

- 全ペア O(N²) を取らず、MinHash/LSH (テキスト・音素) と band LSH (音声内容指紋) で候補ペアを
  生成してから厳密判定する (DESIGN.md Stage 5「MinHash/LSH で O(N²) を回避」)。
- scores["fp_sig"] (Stage 2 が保存した MFCC フレーム列の MinHash 内容指紋。話者ベクトルでは
  ない) の一致率で、テキストが微妙に違うズレ切り出し重複を検出する (reason=fuzzy_audio)。
- union-find のマージ時にグループ理由を正しく引き継ぐ (優先度が高い理由を採用)。
- 代表は「閾値チェックを通過したもの」から quality → asr_conf → snr で選ぶ。代表候補が後で
  閾値落ちしてグループごと消えることを防ぐ (evaluate_checks が先に走っている前提)。
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from .config import Config
from .manifest import Record

_REASON_PRIORITY = {"exact": 0, "fuzzy_text": 1, "fuzzy_audio": 2}

# 内容指紋 fp_sig の MinHash 署名パラメータ (Stage 2 が FP_NUM_PERM 長で生成 / ここで band 分割して照合)。
# 一方だけ変えると band 切り出しが静かに壊れるため、生成側 (stage2) もこの定数を import して揃える。
FP_NUM_PERM = 32
FP_BANDS = 4

# カタカナ読み → 音素風トークン列 (辞書引きなしの決定論的変換)
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
# で切り詰め、イ・エ に相当するモーラを作らない。
_MORA = {k: (c, v) for row, c in _ROWS for k, v in zip(row, "aiueo", strict=False)}
_MORA.update({k: ("", v) for k, v in _VOW.items()})
# 小文字・促音・長音を脱落させない (短文での誤重複除外リスク)。pyopenjtalk-plus は拗音を
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
    """テキスト → 音素トークン列 (被覆率レポートと音素 Jaccard の共通基盤)."""
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


# ---------------------------------------------------------------- LSH 候補ペア生成

_PRIME = (1 << 61) - 1


def _token_hashes(tokens) -> np.ndarray:
    import hashlib

    def one(t):
        b = t if isinstance(t, bytes) else t.encode("utf-8")
        return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")

    return np.fromiter((one(t) for t in tokens), dtype=np.uint64)


def minhash_signature(tokens, num_perm: int = 32, seed: int = 42) -> list[int]:
    """トークン集合の MinHash 署名 (seed 固定 → 実行間・rescore 間で比較可能). 内容指紋に使う."""
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

    # 1) 候補ペア検出 (全ペア比較ではなく LSH で候補を出す。完全一致はハッシュで O(N))
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

    # 内容指紋 (fp_sig) 一致によるズレ切り出し重複。band 衝突候補のみ厳密比較する
    sigs = [r.scores.get("fp_sig") or [] for r in records]
    if any(len(s) == FP_NUM_PERM for s in sigs):
        seen_pairs = {(i, j) for i, j, _ in pairs}
        band_w = FP_NUM_PERM // FP_BANDS
        buckets: dict[tuple, list[int]] = {}
        for i, sig in enumerate(sigs):
            if len(sig) != FP_NUM_PERM:
                continue
            for band in range(FP_BANDS):
                buckets.setdefault(
                    (band, tuple(sig[band * band_w : (band + 1) * band_w])), []
                ).append(i)
        for idxs in buckets.values():
            for x in range(len(idxs)):
                for y in range(x + 1, len(idxs)):
                    i, j = min(idxs[x], idxs[y]), max(idxs[x], idxs[y])
                    if (i, j) in seen_pairs:
                        continue
                    agree = float(np.mean(np.array(sigs[i]) == np.array(sigs[j])))
                    if agree >= a_jac:
                        pairs.append((i, j, "fuzzy_audio"))

    # 2) union-find でグループ化 (優先度の高い理由をグループへ正しく引き継ぐ)
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

    # 3) グループ代表1本のみ自動採用 (閾値通過者を優先し、quality → asr_conf → snr の決定式)
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
                ),  # 閾値を通過しているものを代表に (全員落ちのグループを作らない)
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


# ---------------------------------------------------------------- 多様性選別 (Stage 5b)


def _trigram_sets(tokens: list[list[str]]) -> list[set]:
    """各発話の音素トークン列から三量文字列集合を作る (phoneme 類似度の基盤。1 発話=1 集合)."""
    return [{"".join(t[i : i + 3]) for i in range(max(len(t) - 2, 0))} for t in tokens]


def _quality_key(r: Record) -> tuple:
    """代表/採用の優先順位 (dedup 代表と同じ決定式: quality → asr_conf → snr)."""
    return (
        r.scores.get("quality_score") or 0,
        r.scores.get("asr_conf") or 0,
        r.scores.get("snr") or -99,
    )


def _redundant_adjacency(recs: list[Record], tokens: list[list[str]], metric: str, thr: float):
    """「類似しすぎ」ペアの隣接表を metric 別に作る (局所 index → 類似している局所 index 集合).

    metric は内容ベースのみ (phoneme=音素三量文字列 / text=shingle / audio=内容指紋 fp_sig)。
    全ペア O(N²) を取らず LSH/band 衝突で候補を出して厳密判定する (大コーパスでもメモリ有界)。
    ※ 話者/音響ベクトル emb は使わない: 単一話者コーパスでは全発話が emb 上で近接し別内容まで
      間引いてしまう。類似は「何を言ったか (内容)」で測る。
    """
    adj: dict[int, set] = defaultdict(set)

    def link(i: int, j: int) -> None:
        adj[i].add(j)
        adj[j].add(i)

    if metric in ("phoneme", "text"):
        if metric == "text":
            feats = [_shingles(r.text) if r.text else set() for r in recs]
        else:
            feats = _trigram_sets(tokens)
        for i, j in _minhash_candidate_pairs(feats, num_perm=128, rows=4, seed=11):
            if _jaccard(feats[i], feats[j]) >= thr:
                link(i, j)
    elif metric == "audio":
        sigs = [r.scores.get("fp_sig") or [] for r in recs]
        bw = 2  # 低しきい値でも recall を出すため細かめの band
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(sigs):
            if len(s) != FP_NUM_PERM:
                continue
            for b in range(FP_NUM_PERM // bw):
                buckets[(b, tuple(s[b * bw : (b + 1) * bw]))].append(i)
        for idxs in buckets.values():
            for x in range(len(idxs)):
                for y in range(x + 1, len(idxs)):
                    i, j = idxs[x], idxs[y]
                    if (
                        j not in adj[i]
                        and float(np.mean(np.array(sigs[i]) == np.array(sigs[j]))) >= thr
                    ):
                        link(i, j)
    return adj


def _tercile_bins(vals: dict) -> dict:
    """値 → 三分位ビン (0/1/2)。既知値が 4 未満か値が None なら "na" (キャップ免除)。"""
    known = [v for v in vals.values() if v is not None]
    if len(known) < 4:
        return dict.fromkeys(vals, "na")
    p33, p66 = np.percentile(known, [33.0, 66.0])
    return {
        i: ("na" if v is None else (0 if v < p33 else 1 if v < p66 else 2)) for i, v in vals.items()
    }


def diversity_select(records: list[Record], cfg: Config) -> None:
    """重複排除後に残った候補の貪欲選択 (多様性向上)。既定設定は従来の「品質順+類似間引き」と同動作。

    採用集合の総対ペア類似度を sim_threshold 未満に抑える独立集合を貪欲に作る。
    選択のたびに「新規音素 three-gram 利得」を足せる (gain_lambda) ので、
    protect_coverage の「唯一の供給源保護」を連続値に一般化した形になっている。

    [diversity] の追加ノブ (すべて既定は無効 = 旧動作):
    - gain_lambda: 音素 three-gram カバレッジ利得の重み λ。選択スコア = 品質順位 + λ×(新 trigram 数/全体)
    - max_bucket_share: 韻律バケット (F0 三分位 × 話速三分位) ありの最大採用シェア。
      読み上げ調などへの偏りを整流化する (韻律は emb と違い「話者」ではなく「話し方」の軸)。
    - max_source_share: 投入ソース 1 本の最大採用シェア (長尺 1 本のスタイルに引きずられるのを防ぐ)。
    - target_count: 目標本数に達したら選択を打ち切る (残りは target_size で除外。
      多様性最大化の選択を「K 本選ぶ」形にできる。λ>0 と組み合わせるのが想定)。
    protect_coverage=true の間、ある音素の唯一の供給源は similar / cap / target_size のいずれでも
    落とさない。除外理由は diversity_reason に残す (similar:<代表utt> / cap:prosody:<bin> /
    cap:source:<src> / target_size)。
    """
    import math
    from collections import Counter

    d = cfg.data.get("diversity", {})
    for r in records:  # rescore 再実行時に前回の多様性判定を持ち越さない
        r.diversity_reason = None
    if not d.get("enabled"):
        return
    thr = d.get("sim_threshold", 0.6)
    metric = d.get("metric", "phoneme")
    protect = d.get("protect_coverage", True)
    gain_lambda = float(d.get("gain_lambda", 0.0) or 0.0)
    max_bucket_share = float(d.get("max_bucket_share", 0.0) or 0.0)
    max_source_share = float(d.get("max_source_share", 0.0) or 0.0)
    target_count = int(d.get("target_count", 0) or 0)
    if metric not in ("phoneme", "text", "audio"):
        raise ValueError(
            f"[diversity] metric = {metric!r} は使用不可 (内容ベースの 'phoneme' / 'text' / 'audio' のいずれか)。"
            " 話者/音響ベクトル emb は単一話者コーパスで別内容まで間引くため意図に反する"
        )
    dedup_key = {"phoneme": "phoneme_jaccard", "text": "text_jaccard"}.get(metric)
    if dedup_key is not None and thr >= cfg.get(f"dedup.{dedup_key}"):
        warnings.warn(
            f"[diversity] sim_threshold={thr} >= dedup.{dedup_key}={cfg.get(f'dedup.{dedup_key}')}:"
            " 重複排除が既にその以上の類似を代表1本にまとめるので、多様性選別は何も間引けない"
            " (sim_threshold を dedup 閾値未満に下げて『重複ではないが似た』帯を狙う)",
            stacklevel=2,
        )

    # 候補 = 閾値チェックを通過し、重複排除で代表採用された発話のみ (既に落ちている/重複のものは対象外)
    cand = [i for i, r in enumerate(records) if _passes(r) and not r.dup_reason]
    if len(cand) < 2:
        return
    cand_recs = [records[i] for i in cand]
    tokens = [phoneme_tokens(r.text) if r.text else [] for r in cand_recs]
    adj = _redundant_adjacency(cand_recs, tokens, metric, thr)
    ph_sets = [set(t) for t in tokens]  # 音素カバレッジ保護用
    tri_sets = _trigram_sets(
        tokens
    )  # 利得とカバレッジは音素 three-gram 基準 (metric によらない発音の広がり)
    total_tri = len(set().union(*tri_sets)) if tri_sets else 0

    n = len(cand)
    order_q = sorted(range(n), key=lambda c: _quality_key(cand_recs[c]), reverse=True)
    q_norm = {c: 1.0 - rank / (n - 1) for rank, c in enumerate(order_q)}  # 品質降順の正規化順位

    # キャップ (韻律バケット / 投入ソース)。0 = 無効。バケット "na" (計測不能) は免除
    bucket_cap = math.ceil(max_bucket_share * n) if max_bucket_share > 0 else None
    source_cap = math.ceil(max_source_share * n) if max_source_share > 0 else None
    f0_bins = _tercile_bins({c: cand_recs[c].scores.get("f0_median_hz") for c in range(n)})
    rate_bins = _tercile_bins({c: cand_recs[c].scores.get("mora_rate") for c in range(n)})

    def bucket_of(c: int):
        b0, b1 = f0_bins[c], rate_bins[c]
        return None if (b0 == "na" or b1 == "na") else f"{b0}-{b1}"

    def src_of(c: int) -> str:
        return cand_recs[c].src

    cnt_bucket: Counter = Counter()
    cnt_src: Counter = Counter()
    covered_ph: set = set()
    covered_tri: set = set()
    kept: list[int] = []
    undecided: set[int] = set(range(n))

    def new_phoneme(c: int) -> bool:  # この発話だけが供給する新音素を持つか (カバレッジ保護)
        return bool(ph_sets[c] - covered_ph)

    while undecided:
        if target_count and len(kept) >= target_count:
            break  # 目標到達。残りは下の後始末で target_size / 供給源保護
        pickable = []
        for c in undecided:
            b = bucket_of(c)
            if bucket_cap is not None and b is not None and cnt_bucket[b] >= bucket_cap:
                continue  # バケット満杯: 後回し (最終処理で cap:prosody 行き)
            if source_cap is not None and cnt_src[src_of(c)] >= source_cap:
                continue  # ソース上限: 後回し (最終処理で cap:source 行き)
            g = len(tri_sets[c] - covered_tri) / total_tri if (gain_lambda and total_tri) else 0.0
            # 同順位タイは cand 出現順 (旧実装の安定ソートと同順) になるよう -c で決める
            pickable.append((q_norm[c] + gain_lambda * g, -c))
        if not pickable:
            break
        c = -max(pickable)[1]
        kept.append(c)
        undecided.discard(c)
        b = bucket_of(c)
        if b is not None:
            cnt_bucket[b] += 1
        cnt_src[src_of(c)] += 1
        covered_ph |= ph_sets[c]
        covered_tri |= tri_sets[c]
        for j in adj.get(c, ()):  # 新規採用と類似しすぎたものを落とす (新音素の供給源は保護)
            if j in undecided and not (protect and new_phoneme(j)):
                records[cand[j]].diversity_reason = f"similar:{cand_recs[c].utt_id}"
                undecided.discard(j)

    # 後回し/残余の確定: 供給源保護 > target_size > cap:prosody > cap:source > 救済採用
    for c in sorted(undecided):
        if protect and new_phoneme(c):
            kept.append(c)
            covered_ph |= ph_sets[c]
            b = bucket_of(c)
            if b is not None:
                cnt_bucket[b] += 1
            cnt_src[src_of(c)] += 1
            continue
        if target_count and len(kept) >= target_count:
            records[cand[c]].diversity_reason = "target_size"
            continue
        b = bucket_of(c)
        if bucket_cap is not None and b is not None and cnt_bucket[b] >= bucket_cap:
            records[cand[c]].diversity_reason = f"cap:prosody:{b}"
            continue
        if source_cap is not None and cnt_src[src_of(c)] >= source_cap:
            records[cand[c]].diversity_reason = f"cap:source:{src_of(c)}"
            continue
        kept.append(c)  # 安全弁 (通常フローでは到達しない)
        covered_ph |= ph_sets[c]
