"""CTC 強制アライメント — text↔audio の実一致検証と、先頭/末尾の「テキストに無い発話」検出。

既存の align_conf は qwen では asr_conf (トークン平均確率) の使い回しで、2 ゲートが実質同じものを
見ていた。ここでは日本語 CTC ASR モデルで実アライメントを計算し、
- ctc_align_score      : アライメント経路上の平均後部確率 (低い = 音声とテキストが本当に合わない)
- unaligned_edge_ratio : 最初/最後のアライメント済み文字の外側にある音声の割合
                         (高い = 切り出し境界の外に発話が残っている = 2 文混入・境界ズレ)
を記録する。DESIGN の whisperX 境界補正計画の計測基盤も兼ねる。

既定モデルの jonatasgrosman/wav2vec2-large-xlsr-53-japanese (Common Voice 日本語の orthographic
文字語彙) は、ASR 出力 (漢字混じり) をそのままターゲットにできる実測分離性を持つ
 (実テキスト 0.76 対 別テキスト 0.00。音素ハブート系は 0.53 対 0.23 と分離が弱く採用外)。
注意: 発音が同じで漢字/仮名表記が違う場合 (ASR と CTC モデルの漢字変換のズレ) はスコアが下がる。
ゲートを締める前に実測分布を見て、表記ズレで落ちない水準に決めること。
"""

from __future__ import annotations

_MODEL: dict = {}


def _load(cfg):
    from .config import resolve_device

    repo, device = cfg.get("alignment.model"), resolve_device(cfg)
    if (repo, device) not in _MODEL:
        from transformers import AutoModelForCTC, AutoProcessor

        model = AutoModelForCTC.from_pretrained(repo).eval().to(device)
        _MODEL[(repo, device)] = (model, AutoProcessor.from_pretrained(repo))
    return _MODEL[(repo, device)]


def _text_targets(text: str, tokenizer) -> list[int]:
    """テキストを CTC トークナイザで文字 ID 列へ。blank/special/UNK は検証不能なので落とす。

    語彙に無い文字 (記号など) は黙って除外 — 落ちる量が大きいレコードはターゲット不足で
    スコアが出ないので、ゲート側で検証不能扱いになる。
    """
    unk = getattr(tokenizer, "unk_token_id", None)
    skip = set(tokenizer.all_special_ids) | ({unk} if unk is not None else set())
    return [int(i) for i in tokenizer(text)["input_ids"] if int(i) not in skip]


def align_record(r, audio16k, cfg) -> None:
    """1 レコードを強制アライメントして ctc_align_score / unaligned_edge_ratio を記録する。

    失敗は errors (常に不合格) ではなく align_errors に寄せる: ゲートが無効な間にアライメントが
    失敗しても、そのレコードまで道連れにしない。ゲートが有効ならスコア欠損=検証不能=不合格になる。
    """
    try:
        import torch
        import torchaudio.functional as AF

        model, processor = _load(cfg)
        targets = _text_targets(r.text or "", processor.tokenizer)
        if len(targets) < 2:
            r.scores.setdefault("align_errors", []).append("no_alignable_chars")
            return

        with torch.inference_mode():
            inp = processor(audio16k.tolist(), sampling_rate=16000, return_tensors="pt").to(
                model.device
            )
            logits = model(**inp).logits  # (1, T, V)
            lp = torch.log_softmax(logits.float(), dim=-1)
            T = lp.shape[1]
            tgt = torch.tensor([targets], dtype=torch.int32, device=lp.device)
            # torchaudio の forced_align は (path, scores) を返す。path の値はブランク 0 を除き
            # **トークン ID 直呼び** (k2 系の ID+1 規約ではないので -1 しないこと)。
            # input_lengths も log_probs と同じデバイス (CUDA 時は CUDA) でないとカーネルが落ちる
            path, _scores = AF.forced_align(
                lp, tgt, input_lengths=torch.tensor([T], dtype=torch.int32, device=lp.device)
            )
            path = path[0]  # (T,) 各フレームのラベル (0=blank)
        nz = path.nonzero().flatten()
        if len(nz) == 0:
            r.scores.setdefault("align_errors", []).append("empty_align_path")
            return
        sel = lp[0, nz, path[nz].long()]  # 経路上の各フレームで正しいトークンの対数確率
        r.scores["ctc_align_score"] = round(float(sel.exp().mean()), 3)
        first, last = int(nz[0]), int(nz[-1])
        # 先頭側 / 末尾側の「どの文字にも割り当てられなかった音声」の大きい方 (2 文混入・境界ズレの検出)
        r.scores["unaligned_edge_ratio"] = round(max(first / T, 1.0 - (last + 1) / T), 3)
    except Exception as e:  # noqa: BLE001
        r.scores.setdefault("align_errors", []).append(repr(e))
