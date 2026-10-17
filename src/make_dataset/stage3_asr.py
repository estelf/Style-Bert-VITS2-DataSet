"""Stage 3: 文字起こしと言語系チェック (ASR + G2P 整合 3段構え + クロス検証).

- ASR バックエンドは `filters.toml [asr] backend` で差し替え可能:
    backend="qwen3-asr" (既定, Qwen/Qwen3-ASR-1.7B-hf を transformers native で実行) /
    backend="faster-whisper" (CPU で軽い代替, model は Whisper サイズ名)
  model にはそれぞれ HF リポジトリID / Whisper サイズ名 を入れる。
- G2P 整合①: pyopenjtalk-plus (学習側と同一エンジン) が正しく処理できるか・unknown トークン検出。
  unknown = 学習側の読みチェック (YomiError) で実際に落ちる語 (pron=「、」の読めない語、yomi.py で鏡写し)
- G2P 整合②: モーラ数 ÷ 実音声長 = 話速が話者中央値から乖離ならフラグ (2文混入・ASR欠落・ガチャ検出)
- G2P 整合③: アライメント尤度 → align_conf として記録
- ハルシネーション検出 — no_speech_prob / compression_ratio / 同一文言の繰り返し (ブロック) を記録し、
  Stage 6 の閾値で弾く (不良データ混入量の最大要因)。
- ♪ 等の音楽記号は「音楽だと判定した」強いシグナルなので除去せず music_flag として残す。
- NFKC は全角｜を半角 | に変換して esd.list の行フォーマットを壊すため、| を ・ に置換する。
- mora_rate の分母は実音声長 (Stage 1 が無音パディングなしで切るため end-start がそのまま音声長)。
- 第 2 ASR クロス検証 — [asr2] で別系統を走らせ、2 系統間 CER を**読みレベル** (reading_key) で
  x_cer に記録する (表層文字値は x_cer_surface。句読点と同音異字の実測寄与が大きい)。
  安価な音声フィルタを通過した候補だけ走らせる (第 2 システムのコストは採用候補分だけ)。
- 信頼度は平均だけでなく分布 (min / p10 / 平均NLL) で記録 — 平均が高いのに 1 語だけ極端に
  低い = そこだけ聞き間違い、という典型を token_prob_min / token_nll で拾う。
- 任意で CTC 強制アライメント (align.py) と MLM 擬似 PPL (lm.py) も同じ段で記録する。
- 記号は読み上げ展開してから G2P にかける (itn.py。% → パーセント 等)。
"""

from __future__ import annotations

import contextlib
import re
import unicodedata
import warnings

import numpy as np
from tqdm import tqdm

from . import itn
from .config import Config

FILLERS = ["あの", "まあ", "えーと", "えっと", "なんか", "やっぱり", "まぁ"]
# 装飾記号等はセリフから除去 (学習側 dict_data 連携での ITN 正規化は今後の課題)
_STRIP_CHARS = "♪♡♥★☆※―…"
_MUSIC_CHARS = "♪♫"  # 除去する前に検出して music_flag に残す
# 同一文字列の 3 回以上の連続 = whisper ハルシネーションの典型 (「ありがとう」無限ループ等)
_REPEAT_RE = re.compile(r"(.{3,})\1{2,}")
# 文末の「開き」検出。text_tail_open は 0=文として閉じている / 1=言い淀み止め (続きがありそうだが
#   演技上は成立する) / 2=途中で切れている (納品してはいけない類)。ゲート max_text_tail_open が許す上限を決める。
#   文末記号として許す文字 (NFKC 前後の形をカバー)。読点・助詞止めは「開いている」側なので入れない。
_SENT_END = "。．.！!？?…‥"
# 閉じ括弧・引用符の後ろに文末記号が来る形 (「〜。」で終わるテキスト) を許す
_TAIL_TRIM = "」』）)】》]\"'’”"
# JTalk 読みで語の途中で切れていることを示す小文字 (混じっ→マジッ)。… は読みでは全サイズになるので対象外
_TAIL_CUT_KANA = "ッ"
# 活用形を見る対象品詞 (実詞)。名詞・副詞・感動詞止めはそれ自体で文を閉じうる
_TAIL_OPEN_POS = {"動詞", "形容詞", "助動詞"}
# 文を閉じない活用形 (中止法・条件節・連体止め)。終止形/連体形/命令形/未然形 (意志・依頼) は文を閉じるので対象外。
#   連用タ接続 (「待って。」= 口語の依頼) と連用テ接続 (「早く。」) も口語では完文になるので対象外にしている。
_TAIL_OPEN_CFORM = {"仮定形", "連用形", "体言接続", "連用ニ接続"}


def is_sentence_terminated(text: str, raw: str | None = None) -> bool:
    """テキストが文末記号で終わっているか。

    判定は ASR 生テキスト (raw) があればそちらで行う。normalize_text は装飾記号として … を落とすため、
    正規化後テキストだけを見ると「ごめんね…」が終端なし=切れ目と誤爆する。raw がない
    (人手オーバーライド・rescore 再計算など) 場合は正規化後テキストで保守的に見る。
    """
    tail = (text if raw is None else raw).strip().rstrip(_TAIL_TRIM).strip()
    return bool(tail) and tail[-1] in _SENT_END


def tail_openness(text: str, feats: list | None = None, raw: str | None = None) -> tuple[int, str]:
    """文末が開いている深刻度 (0/1/2) と理由を返す。発話・文字起こしが途中で切れているものの検出。

    - 2 (切れている): 文末記号が全く無い / 動詞・形容詞・助動詞の読みが促音で止まる
      (「お客様の体温と混じっ」「〜言っ」型。語の途中で発話が終わっている)。納品対象にしない。
    - 1 (言い淀み): 条件節 (「〜したら。」)・連用形中止 (「〜だし。」)・連体止め (「〜ような。」) など、
      続きがありそうだが演技上の言い淀みとして成立する。既定は採用し、分布を見て判断できるよう記録だけ残す。
    - 0: 文として閉じている。

    品詞・活用形は G2P と同じ pyopenjtalk (学習側と同一エンジン) の run_frontend を使う。feats を渡すと
    G2P で使ったものを流用する (同じ解析を二回やらない)。助詞止め (「今から。」) は口語で完文になるため
    敢えて検出しない (実コーパス 3,539 件で 34% が該当し、欠陥ではなく口語の普通形だった)。
    """
    if not text.strip():
        return 0, ""
    if not is_sentence_terminated(text, raw):
        return 2, "no_sent_end"
    if feats is None:
        import pyopenjtalk as pj

        try:
            feats = pj.run_frontend(text)
        except Exception:  # noqa: BLE001
            return 0, ""  # 解析できないテキストを「切れている」扱いにして全件落とすのを防ぐ
    words = [f for f in feats or [] if (f.get("pos") or "") != "記号"]
    if not words:
        return 0, ""
    w = words[-1]
    pos, cform = w.get("pos") or "", w.get("cform") or ""
    pron = (w.get("pron") or "").upper()
    if pos in _TAIL_OPEN_POS and pron.endswith(_TAIL_CUT_KANA):
        return 2, "verb_cut"
    if pos in _TAIL_OPEN_POS and cform in _TAIL_OPEN_CFORM:
        return 1, f"cform:{cform}"
    return 0, ""


def normalize_text(text: str, expand_symbols: bool = True) -> str:
    """全角/半角正規化: NFKC + esd.list 区切り保護 + 記号読み上げ + 装飾記号除去 + 空白圧縮."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("|", "・")  # NFKC が全角｜を半角に変換するので、区切り文字は安全な記号へ
    if expand_symbols:
        text, _ = itn.expand_symbols(text)  # % → パーセント 等 (G2P 失敗ゲートの誤爆削減)
    text = "".join(c for c in text if c not in _STRIP_CHARS)
    return re.sub(r"\s+", " ", text).strip()


def text_features(text: str) -> dict:
    """テキストフィルタ用スコア (メイン ASR / asr2 / 人手オーバーライド共通の唯一の計算源)。

    修正:
    - kana_ratio の分母は かな+英数字 のみ。旧実装は len(text) 割りなので句読点や漢字に薄められ、
      「本日、会議は中止となりました」のような漢字多めの正常な日本語発話で誤爆していた。
      英数もかなもない発話 (純漢字・純記号) は「非日本語の証拠」が無いので 1.0 とする
      (純記号ゴミは residual_symbols と min_chars が別途捕まえる)。
    - filler_ratio は出現回数で数える。旧実装は各語 1 回出現かどうかだけで、「えっと」×3 でも 3 文字分
      しか数えていなかった。
    文末の開き (text_tail_open) は品詞・活用形が必要なので g2p_check 側で記録する。
    """
    kana = sum(1 for c in text if "\u3041" <= c <= "\u3096" or "\u30a1" <= c <= "\u30fa")
    alnum = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isdigit()))
    alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    filler_chars = sum(len(f) * text.count(f) for f in FILLERS)
    return {
        "char_count": len(text),
        "kana_ratio": round(kana / (kana + alnum), 2) if (kana + alnum) else 1.0,
        "alpha_only": bool(alpha and kana == 0 and text),
        "filler_ratio": round(filler_chars / max(len(text), 1), 2),
        "residual_symbols": itn.residual_symbols(text),  # 展開後も残る記号 (G2P リスクの可視化)
    }


def _cer(a: str, b: str) -> float:
    """文字レベル編集距離 ÷ 最大長 (2 系統クロス検証用。日本語は文字単位で十分)。"""
    if a == b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[len(b)] / max(len(a), len(b), 1)


_SMALL_KANA = str.maketrans({"ャ": "ヤ", "ュ": "ユ", "ョ": "ヨ", "ヴ": "ブ", "ヷ": "バ"})

# x_cer の正規化アルゴリズム版数。reading_key を変えたら**上げる**。rescore の refresh_asr2_scores は
# 「この版数を持たないレコードだけ」第 2 ASR を引き直すので、上げ忘れると古い正規化で計算済みの
# x_cer が manifest に残り続けて、修正がデータセットに反映されない (下げてはいけない)。
X_CER_ALGO = 2


def reading_key(text: str) -> str:
    """発音 (読み) レベルの正規化キー。2 系統 ASR の比較を表層文字でなくこれで取る。

    実測 (6 話者 10,369 候補のうち asr2 計測済み 4,050 件、Qwen3-ASR ⇄ faster-whisper small):
    表層文字のままでは x_cer 中央値 0.11・完全一致 1% なのに、読み化すると中央値 0.00・完全一致 79%。
    差分の大半は句読点 (whisper は「。」を打たない) と同音異字 (例: モフモフのバンテン ⇄ もふもふの番点)
    で、発話の教師としては実質一致している。表層でゲートを締めると正常発話を大量に落とすことになる。
    文末の長音だけ落とすのは実測で多かった伸ばし (どうぞ ⇄ どうぞー) 対策で、語中の長音 (オバーサン) は
    保持するのでおばあさん/おばさんのような最小対は潰れない。g2p が読めない入力は句読点だけ落とした
    表層形に退避する (比較不能で全滅させるより弱い指標でも残す)。
    """
    if not text:
        return ""
    # 頭字語の中の句点 (S.F. / U.S.A.) を先に消す。pyopenjtalk は語中の句点を境にそこだけ読まず、
    # 「S.F.とか。」から「トカ」だけを返して第 2 系統の「SFとか」と一致しなくなる (実測で誤検出 1 件)
    text = re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z])", "", text)
    try:
        import pyopenjtalk as pj  # pyopenjtalk-plus (学習側と同一エンジン)

        kana = pj.g2p(text, kana=True)
    except Exception:  # noqa: BLE001  # 読み取り失敗は表層形で比較
        kana = ""
    kana = "".join(kana) if isinstance(kana, list) else str(kana)
    kana = re.sub(r"[^ァ-ヴー]", "", kana).translate(_SMALL_KANA)
    if not kana:
        return re.sub(r"[\W_]+", "", text)  # 記号のみ等の退避: 句読点だけ落とした表層形で比較する
    return kana[:-1] if kana.endswith("ー") and len(kana) > 1 else kana


def _token_stats(probs: list[float]) -> dict:
    """トークン/ワード確率の分布統計。平均だけでは「1 語だけ極端に低い」聞き間違いを隠す。"""
    if not probs:
        return {}
    p = np.asarray(probs, dtype=np.float64)
    return {
        "token_prob_min": round(float(p.min()), 3),
        "token_prob_p10": round(float(np.percentile(p, 10)), 3),
        "token_nll": round(float(-np.log(np.clip(p, 1e-6, 1.0)).mean()), 3),
    }


def g2p_check(text: str, raw: str | None = None) -> dict:
    """pyopenjtalk-plus が通るか・モーラ数・学習側読みチェック (YomiError) の失敗語・文末の開きを返す.

    学習側 preprocess.text (既定 yomi_error=raise) は pron=「、」の語 (読めない漢字・古いカタカナ等)
    を YomiError で行ごと弾く。旧実装の unknown_tokens は pron に出ない「*」「?」を見ていて実質
    死んでいたため、yomi.py で学習側の失敗条件そのものを鏡写しして数える (誤爆防止のため学習側と
    同じ正規化を通してから run_frontend に掛ける)。読めない語は yomi_bad_words に残す
    (ユーザー辞書候補・report.html 表示用)。

    文末の開き (text_tail_open) はここで G2P と同じ run_frontend 結果を流用して決める
    (同じ解析を二回やらない)。raw は ASR 生テキスト (… 除去による誤爆防止)。
    """
    import pyopenjtalk as pj  # pyopenjtalk-plus (学習側と同一エンジン)

    from . import yomi

    out = {"g2p_ok": True, "unknown_tokens": 0, "mora_count": 0}
    phrases: list = []
    try:
        phrases = pj.run_frontend(text)
        out["mora_count"] = sum(f["mora_size"] for f in phrases)
        if out["mora_count"] == 0:
            out["g2p_ok"] = False
    except Exception:  # noqa: BLE001
        # pyopenjtalk-plus は入力により多种の例外を投げうるので、G2P 失敗は全て g2p_ok=False に寄せる
        out["g2p_ok"] = False
    with contextlib.suppress(Exception):
        # 判定に失敗したものを「切れている」寄りにして全件落とさない (品詞解析は入力により多种の例外を投げる)
        out["text_tail_open"], out["text_tail_open_reason"] = tail_openness(
            text, feats=phrases, raw=raw
        )
    try:
        bad = yomi.unreadable_words(text)
    except Exception:  # noqa: BLE001
        bad = ["<run_frontend failed>"]  # 打診自体の失敗も学習側は通さない (保守的に不合格側へ)
    out["unknown_tokens"] = len(bad)
    if bad:
        out["g2p_ok"] = False  # YomiError は yomi_error=raise の学習側でハードフォール
        out["yomi_bad_words"] = ",".join(dict.fromkeys(bad))[:120]
    return out


def refresh_yomi_scores(records) -> int:
    """rescore 用: manifest のテキストから読みチェック (yomi.py) だけ引き直す (音声不要・安価).

    旧 manifest の unknown_tokens は旧 (死んでいた) チェックの値なので、再計算しないと新しい
    ゲートが rescore で効かない。エンジン起動に失敗したランで全件誤って不合格にする事故を防ぐため、
    最初の 1 件で打診に失敗したら何も触れず 0 を返す。
    """
    import warnings

    from . import yomi

    n_flagged = 0
    probed = False
    for r in records:
        if not (r.text or "").strip():
            continue
        try:
            bad = yomi.unreadable_words(r.text)
        except Exception as e:  # noqa: BLE001
            if not probed:  # 初回失敗 = エンジン自体が起動していない疑い。全件を道連れにしない
                warnings.warn(
                    f"読みチェック (yomi.py) を初期化できないためスキップ: {e!r}", stacklevel=2
                )
                return 0
            bad = ["<run_frontend failed>"]
        probed = True
        r.scores["unknown_tokens"] = len(bad)
        if bad:
            r.scores["g2p_ok"] = False
            r.scores["yomi_bad_words"] = ",".join(dict.fromkeys(bad))[:120]
            n_flagged += 1
        else:
            r.scores.pop("yomi_bad_words", None)
    return n_flagged


def refresh_text_tail_scores(records) -> int:
    """rescore 用: manifest のテキストから文末の開き を引き直す (音声・ASR 不要・安価)。

    旧 manifest には text_tail_open が無いままなので、ゲートを有効にして rescore すると
    「検証不能=不合格」で全件落ちる。テキストから引き直せば再 run なしで閾値を再調整できる。
    raw (ASR 生テキスト) は manifest に残っていないため正規化後テキストで判定する。
    normalize_text が落とすのは装飾記号だけで、実コーパス 3,539 件でも正規化後に文末記号が
    消えた発話は 0 件 (raw 判定と同じ結果) なので、run 時の値と食い違わない。
    """
    n = 0
    for r in records:
        if not (r.text or "").strip():
            continue
        v, why = tail_openness(r.text)
        r.scores["text_tail_open"] = v
        if why:
            r.scores["text_tail_open_reason"] = why
        else:
            r.scores.pop("text_tail_open_reason", None)
        n += 1
    return n


# ---------------------------------------------------------------- ASR バックエンド

_WHISPER = {}


def _asr_whisper(audio16k: np.ndarray, cfg: Config, prefix: str = "asr") -> tuple[str, float, dict]:
    """faster-whisper で文字起こし → (テキスト, 信頼度, 追加スコア). word probability は align_conf 用.

    prefix を変えると別セクション (例: "asr2") の設定で同じバックエンドを動かせる (クロス検証)。
    """
    from faster_whisper import WhisperModel

    from .config import resolve_device

    device = cfg.get(f"{prefix}.device")
    if device == "auto":
        device = resolve_device(cfg)
    compute_type = cfg.get(f"{prefix}.compute_type")
    if device.startswith("cuda") and compute_type == "int8":
        compute_type = "float16"  # int8 は CPU 量子化。GPU では float16 が速い (docs/GPU.md)
    key = (cfg.get(f"{prefix}.model"), device, compute_type)
    if key not in _WHISPER:
        _WHISPER[key] = WhisperModel(key[0], device=key[1], compute_type=key[2])
    model = _WHISPER[key]
    import inspect

    kwargs = {}  # batch_size 対応版の faster-whisper なら渡す (古い版では内部バッチに任せる)
    if "batch_size" in inspect.signature(model.transcribe).parameters:
        kwargs["batch_size"] = cfg.get(f"{prefix}.batch_size")
    segs, _info = model.transcribe(
        audio16k,
        language=cfg.get(f"{prefix}.language"),
        beam_size=cfg.get(f"{prefix}.beam_size"),
        word_timestamps=True,
        condition_on_previous_text=False,
        **kwargs,
    )
    texts, confs, word_probs = [], [], []
    nsp, cr = 0.0, 0.0
    for seg in segs:
        texts.append(seg.text.strip())
        confs.append(float((1.0 - seg.no_speech_prob) * np.exp(min(0.0, seg.avg_logprob))))
        nsp = max(nsp, float(seg.no_speech_prob))  # 「無声だと言う確率」
        cr = max(cr, float(seg.compression_ratio))  # 圧縮率 = 繰り返しガチャの代表指標
        word_probs.extend(
            float(w.probability) for w in (seg.words or [])
        )  # words が None でも落ちない
    text = "".join(texts)
    extra = {
        "align_conf": round(float(np.mean(word_probs)), 3) if word_probs else 0.0,
        "no_speech_prob": round(nsp, 3),
        "compression_ratio": round(cr, 2),
        **_token_stats(word_probs),
    }
    return text, (float(np.mean(confs)) if confs else 0.0), extra


_QWEN = {}


def _asr_qwen(audio16k: np.ndarray, cfg: Config, prefix: str = "asr") -> tuple[str, float, dict]:
    """Qwen3-ASR (既定バックエンド, transformers native). greedy generate のトークン平均確率を信頼度にする.

    出力は "language <LANG><asr_text>..." 形式なので decode(return_format="transcription_only") で剥がす。
    no_speech_prob / compression_ratio は whisper 固有のハルシネーション指標なのでここでは記録しない
    (filters.toml の該当閾値は qwen3-asr ではコメントアウト済み。有効にすると欠損スコアで全件落ちる)。
    prefix を変えると別セクション (例: "asr2") の設定で動かせる (クロス検証)。
    """
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from .config import resolve_device

    repo = cfg.get(f"{prefix}.model")
    device = cfg.get(f"{prefix}.device")
    if device == "auto":
        device = resolve_device(cfg)
    if (repo, device) not in _QWEN:
        model = AutoModelForMultimodalLM.from_pretrained(repo, dtype=torch.bfloat16).to(device)
        _QWEN[(repo, device)] = (model.eval(), AutoProcessor.from_pretrained(repo))
    model, processor = _QWEN[(repo, device)]

    inputs = processor.apply_transcription_request(
        audio=audio16k, language=cfg.get(f"{prefix}.language") or None  # numpy 16k を直接渡せる
    ).to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max(64, int(len(audio16k) / 16000 * 40) + 32),
        )
    gen = out.sequences[:, inputs["input_ids"].shape[1] :]
    probs = []
    for step, logits in enumerate(out.scores):
        lg = logits[:, -1] if logits.ndim > 2 else logits  # (B, V) を得る
        lp = torch.log_softmax(lg.float(), dim=-1)
        tok = gen[:, step].clamp(max=lp.shape[-1] - 1)
        probs.append(float(lp.gather(-1, tok[:, None]).exp().mean().item()))
    text = processor.decode(gen[0], return_format="transcription_only")
    conf = float(np.mean(probs)) if probs else 0.0
    return (
        text,
        conf,
        {
            "align_conf": round(conf, 3),  # トークン確率をアライメント信頼度に流用
            **_token_stats(probs),  # 平均 (= asr_conf) 以外の分布統計
        },
    )


def _asr2_eligible(r) -> bool:
    """「明らかにゴミ」の発話だけ第 2 ASR をスキップする (第 2 システムのコストを節約)。

    カットオフは**固定値**でユーザー閾値から意図的に切り離している: rescore で min_snr_db 等を
    緩めたときに、asr2 スキップ済みのレコードだけ x_cer 欠損で全滅する罠を作らないため。
    ここで弾くのはどんな閾値設定でも音声ゲートで落ちるレベル (無音スライス・激しいクリップ等) だけ。
    """
    s = r.scores
    if s.get("errors") or s.get("stage1_skipped") or not (r.text or "").strip():
        return False
    if (s.get("seg_dur") or 0.0) < 0.3:
        return False
    if (s.get("digital_silence_ratio") or 0.0) > 0.9:
        return False
    if (s.get("clip_ratio") or 0.0) > 0.05:
        return False
    snr = s.get("snr")
    return not (snr is not None and snr < -5.0)


def refresh_asr2_scores(records, cfg: Config, cache_dir) -> int:
    """[asr2] を**後から**有効にした旧 manifest に x_cer を足す (rescore 用)。

    第 2 ASR は音声を聞くしかなく、run を回し直すとメイン ASR・AST・CTC・LM まで全部やり直しになる
    （コストの実測は docs/DECISIONS.md）。rescore の「閾値再判定だけで納品物を
    作り直す」考え方に従い、足りないスコアだけ中間キャッシュから引き直す。既に x_cer を持つ
    レコードには触らないので、run 済みのデータセットで gate を緩める再判定は数秒で終わる。
    計算は transcribe_segments と同一の経路 (同一モデル・同一前処理)。既に同じ `X_CER_ALGO` を持つ
    レコードは触らない (毎回の rescore で 80 分払わせないため)。`reading_key` を変えたら版数を上げる
    ので古い正規化の値は自動的に引き直される。ただし `[asr2]` の**モデルを変えた**場合は検出できない
    ので `run` を使うこと (rescore はスコアの版数しか見ず、モデル変更を知らない)。
    """
    from .stage0_ingest import load_cached

    if not cfg.get("asr2.enabled"):
        return 0
    backends = {"faster-whisper": _asr_whisper, "qwen3-asr": _asr_qwen}
    b2 = cfg.get("asr2.backend")
    if b2 not in backends:
        raise ValueError(f"unknown asr2 backend: {b2!r} (使用可: 'qwen3-asr' / 'faster-whisper')")
    fn2 = backends[b2]
    expand = bool(cfg.get("text.symbol_readings"))
    todo = [r for r in records if (r.scores or {}).get("x_cer_algo") != X_CER_ALGO]
    n = 0
    for r in tqdm(todo, desc="stage3 asr2 (rescore)", unit="utt"):
        if not _asr2_eligible(r):  # run と同じ扱いで「明らかなゴミ」は第 2 ASR を省く
            r.scores["asr2_skipped"] = "obvious_garbage"
            continue
        try:
            t2, c2, _ = fn2(load_cached(r, cache_dir, 16000), cfg, "asr2")
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("asr2_errors", []).append(repr(e))
            continue
        t2 = normalize_text(t2, expand_symbols=expand)
        r.scores["asr2_text"] = t2
        r.scores["asr2_conf"] = round(c2, 3)
        r.scores["x_cer"] = round(_cer(reading_key(r.text or ""), reading_key(t2)), 3)
        r.scores["x_cer_surface"] = round(_cer(r.text or "", t2), 3)
        r.scores["x_cer_algo"] = X_CER_ALGO
        n += 1
    return n


def transcribe_segments(records, cfg: Config, cache_dir) -> None:
    """各 Record に ASR テキスト・信頼度・テキストフィルタ用スコアを書き込む (判定は Stage 6)."""
    from .stage0_ingest import load_cached

    backends = {"faster-whisper": _asr_whisper, "qwen3-asr": _asr_qwen}
    backend = cfg.get("asr.backend")
    if backend not in backends:
        raise ValueError(
            f"unknown asr backend: {backend!r} (使用可: 'qwen3-asr' / 'faster-whisper')"
        )
    fn = backends[backend]
    expand = bool(cfg.get("text.symbol_readings"))

    fn2 = None  # 第 2 ASR (クロス検証)
    if cfg.get("asr2.enabled"):
        b2 = cfg.get("asr2.backend")
        if b2 not in backends:
            raise ValueError(
                f"unknown asr2 backend: {b2!r} (使用可: 'qwen3-asr' / 'faster-whisper')"
            )
        if b2 == backend and cfg.get("asr2.model") == cfg.get("asr.model"):
            warnings.warn(
                "[asr2] メイン ASR と同じ backend/model のため x_cer は常に 0 に近く、クロス検証にならない"
                " (別系統 — 例: qwen3-asr に対して faster-whisper small 等 — に切り替えること)",
                stacklevel=2,
            )
        fn2 = backends[b2]
    align_on = bool(cfg.get("alignment.enabled"))
    lm_on = bool(cfg.get("lm.enabled"))
    for r in tqdm(records, desc="stage3 asr", unit="utt"):
        try:  # 1 本の異常セグメントでラン全体を止めない
            audio16k = load_cached(r, cache_dir, 16000)
            raw_text, conf, extra = fn(audio16k, cfg)
            text = normalize_text(raw_text, expand_symbols=expand)
            r.text = text
            g = (
                g2p_check(text, raw=raw_text)
                if text
                else {
                    "g2p_ok": False,
                    "unknown_tokens": -1,
                    "mora_count": 0,
                    "text_tail_open": 0,
                }
            )  # 空テキストは empty_text 不変条件で既に不合格なので文末判定は回さない
            dur = r.end - r.start  # Stage 1 が無音パディングなしで切るため、これが実音声長
            sc = r.scores
            sc["asr_conf"] = round(conf, 3)
            sc.update(extra)
            sc.update(g)
            sc["mora_rate"] = (
                round(g["mora_count"] / dur, 2) if g["mora_count"] and dur > 0 else 0.0
            )
            # 繰り返しブロック (whisper ハルシネーション) をブールで記録
            sc["repeat_block"] = bool(_REPEAT_RE.search(raw_text))
            # 音楽記号は「音楽だと判定した」シグナルとして残す (除去だけして黙殺しない)
            sc["music_flag"] = any(c in raw_text for c in _MUSIC_CHARS)
            # テキストフィルタ用スコア (Stage 6 の閾値で棄却可否が決まる)。は text_features に集約
            sc.update(text_features(text))
            # --- 任意の追加検証 (既定は全オフ。失敗は errors ではなく個別 errors キーに寄せる) ---
            if (
                fn2 is not None
            ):  # 2 系統間 CER を**読みレベル**で取る。不一致は Stage 6 の max_x_cer で弾く/見送りに回す
                if _asr2_eligible(r):
                    try:
                        t2, c2, _ = fn2(audio16k, cfg, "asr2")
                        t2 = normalize_text(t2, expand_symbols=expand)
                        sc["asr2_text"] = t2
                        sc["asr2_conf"] = round(c2, 3)
                        sc["x_cer"] = round(_cer(reading_key(text), reading_key(t2)), 3)
                        sc["x_cer_surface"] = round(
                            _cer(text, t2), 3
                        )  # 参考値: 句読点と同音異字を含む表層比較
                        sc["x_cer_algo"] = X_CER_ALGO
                    except Exception as e:  # noqa: BLE001
                        sc.setdefault("asr2_errors", []).append(repr(e))
                else:
                    sc["asr2_skipped"] = "obvious_garbage"
            if align_on:  # ctc_align_score / unaligned_edge_ratio
                from .align import align_record

                align_record(r, audio16k, cfg)
            if lm_on:  # lm_ppl (MLM 擬似パープレキシティ)
                from .lm import score_text

                score_text(r, cfg)
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("errors", []).append(f"stage3: {e!r}")


def flag_mora_rate_outliers(records) -> None:
    """G2P 整合②: 話速 (モーラ数/秒) が話者中央値から乖離なら mora_rate_dev に記録.

    中央値は「クリーンな発話」(パイプラインエラーなし・モーラ取得済み・スキップでない) だけで
    求める。不良発話やスキップ記録を交れると中央値が歪み max_mora_rate_dev が機能しなくなる
    (C: 汚染回避)。rescore でも最新化されるよう Stage 6 直前で呼ぶ。
    """
    rates = [
        r.scores["mora_rate"]
        for r in records
        if r.scores.get("mora_count", -1) > 0
        and not r.scores.get("errors")
        and not r.scores.get("stage1_skipped")
    ]
    med = float(np.median(rates)) if rates else 0.0
    for r in records:
        v = r.scores.get("mora_rate", 0.0)
        r.scores["mora_rate_median"] = round(med, 2)
        r.scores["mora_rate_dev"] = round(abs(v - med) / med, 2) if med and v else 0.0
