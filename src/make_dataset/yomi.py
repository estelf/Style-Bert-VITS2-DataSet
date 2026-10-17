"""学習側「読みチェック」(YomiError) の事前打診。納品後の esd.list 不通過を防ぐ。

学習側 preprocess.text (既定 yomi_error=raise) は Style-Bert-VITS2 の
`clean_text → g2p → text_to_sep_kata` で、pyopenjtalk の `pron` が「、」 (=読めない) かつ
単語が句読点でない語を **YomiError** として弾く。読みのカタカナ変換に失敗する語も
`__kata_to_phoneme_list` の ValueError で同じく行ごと落ちる。実運用で引っかかるのは
**希少漢字 (䶮・垚・妧…)・古いカタカナ (ヷヸヹヺ)・正規化を生き残る記号** などで、
ASR 誤変換やガチャが混ざった低品質発話の目印そのもの。

本モジュールは学習側 `nlp/japanese/normalizer.py` (句読点正規化 + allow-list 除去) と
`g2p.py::text_to_sep_kata` の失敗条件を同一エンジン (pyopenjtalk-plus) で鏡写しにし、
「学習側で落ちる語」を返す。学習側は **正規化してから** run_frontend に掛けるため
(絵文字・キリル文字等は allow-list 外で解析前に消える = 学習側では無害)、生テキストに
そのまま掛けるのは誤爆になる。必ず learner_normalize() 経由で打診すること。

同期元 (Style-Bert-VITS2 側を変更したらここも追従させる):
- style_bert_vits2/nlp/symbols.py::PUNCTUATIONS
- style_bert_vits2/nlp/japanese/normalizer.py::__REPLACE_MAP / __PUNCTUATION_CLEANUP_PATTERN / replace_punctuation
- style_bert_vits2/nlp/japanese/g2p.py::text_to_sep_kata / __kata_to_phoneme_list

意図的な差分 (読みチェックの判定に影響しないものだけ省略):
- 数字→漢字変換 (num2words)。数字は OpenJTalk が読める上に allow-list 除去で消えるため、
  変換の有無で YomiError の可否は変わらない (学習側は変換後に読める漢字になるだけ)。
- word2ph 合計や align_tones のアサーション系 (稀なエッジケース) は鏡写ししない。
  実運用の読みチェック失敗はほぼ全て YomiError (pron=「、」) 経由。
"""

from __future__ import annotations

import re
import unicodedata

# 学習側 symbols.py::PUNCTUATIONS と同期
PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]

# 学習側 normalizer.py::__REPLACE_MAP と同期 (記号類の正規化マップ)
_REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    # NFKC 正規化後のハイフン・ダッシュの変種を全て通常半角ハイフン - \u002d に変換
    "\u02d7": "\u002d",  # ˗, Modifier Letter Minus Sign
    "\u2010": "\u002d",  # ‐, Hyphen
    "\u2012": "\u002d",  # ‒, Figure Dash
    "\u2013": "\u002d",  # –, En Dash
    "\u2014": "\u002d",  # —, Em Dash
    "\u2015": "\u002d",  # ―, Horizontal Bar
    "\u2043": "\u002d",  # ⁃, Hyphen Bullet
    "\u2212": "\u002d",  # −, Minus Sign
    "\u23af": "\u002d",  # ⎯, Horizontal Line Extension
    "\u23e4": "\u002d",  # ⏤, Straightness
    "\u2500": "\u002d",  # ─, Box Drawings Light Horizontal
    "\u2501": "\u002d",  # ━, Box Drawings Heavy Horizontal
    "\u2e3a": "\u002d",  # ⸺, Two-Em Dash
    "\u2e3b": "\u002d",  # ⸻, Three-Em Dash
    "「": "'",
    "」": "'",
}
_REPLACE_PATTERN = re.compile("|".join(re.escape(p) for p in _REPLACE_MAP))
# 学習側 normalizer.py::__PUNCTUATION_CLEANUP_PATTERN と同期。
# 残すのは: ひらがな・カタカナ・漢字 (基本+拡張A)・々・半角/全角アルファベット・ギリシャ文字・PUNCTUATIONS のみ
_CLEANUP_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    + r"\u0041-\u005A\u0061-\u007A"
    + r"\uFF21-\uFF3A\uFF41-\uFF5A"
    + r"\u0370-\u03FF\u1F00-\u1FFF"
    + "".join(PUNCTUATIONS)
    + r"]+"
)
# 学習側 g2p.py::__KATAKANA_PATTERN と同期 (__kata_to_phoneme_list の fullmatch 条件)
_KATAKANA_PATTERN = re.compile(r"[\u30A0-\u30FF]+")
_PUNCT_SET = set(PUNCTUATIONS)


def replace_punctuation(text: str) -> str:
    """学習側 normalizer.py::replace_punctuation と同一処理 (句読点置換 + allow-list 外除去)。"""
    return _CLEANUP_PATTERN.sub("", _REPLACE_PATTERN.sub(lambda x: _REPLACE_MAP[x.group()], text))


def learner_normalize(text: str) -> str:
    """学習側 normalizer.py::normalize_text の鏡写し (数字変換のみ意図的に省略、モジュール docstring 参照)。

    学習側が run_frontend に渡す直前のテキスト形态を作る。ここを通さないと
    絵文字 (allow-list 外 = 学習側では解析前に消える) を誤爆させる。
    """
    res = unicodedata.normalize("NFKC", text)
    res = res.replace("~", "ー").replace("～", "ー").replace("〜", "ー")  # 長音扱い (学習側と同順)
    res = replace_punctuation(res)
    res = res.replace("\u3099", "").replace("\u309a", "")  # 結合文字の濁点・半濁点
    return res


def unreadable_words(text: str) -> list[str]:
    """学習側 text_to_sep_kata (raise_yomi_error=True) で YomiError / ValueError になる語を返す。

    判定条件は学習側と同一:
    - 語 (正規化後) が句読点のみ → 読みチェック対象外 (学習側も素通り)
    - pron == "、" → 学習側 YomiError («Cannot read»)
    - pron がカタカナ (ー含む) のみでない → 学習側 ValueError («Input must be katakana only»)
    空テキスト・句読点のみのテキストは語を持たないので []。
    """
    import pyopenjtalk as pj  # pyopenjtalk-plus (学習側と同一エンジン)

    norm = learner_normalize(text)
    if not norm:
        return []
    bad: list[str] = []
    for part in pj.run_frontend(norm):
        word = replace_punctuation(part["string"])
        yomi = str(part.get("pron", "")).replace("’", "")
        if not word or set(word) <= _PUNCT_SET:
            continue  # 句読点・記号語 (学習側は yomi=word として素通り)
        if yomi == "、" or not _KATAKANA_PATTERN.fullmatch(yomi):
            bad.append(word)
    return bad
