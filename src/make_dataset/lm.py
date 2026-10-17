"""MLM 擬似パープレキシティ — 文脈的に不自然な文字起こしの検出 (誤変換の検出手がかり)。

同音異字で「発音が変わらない」誤変換は TTS 学習では無害だが、発音が変わる誤変換
 (「全然大丈夫」→「全然大平」等) は学習を汚す。後者は往々にして文としてありえない語順・接続に
なるため、日本語 MLM (DeBERTa-v2 日本語) による擬似 PPL が突出して高くなる。
判定はゲート (max_lm_ppl=除外) と見送り閾値 (review_lm_ppl=pending_review 行き) の二段構えにできる。

擬似 PPL は「1 位置ずつ mask して真のトークンの対数確率を拾う」方式を**バッチ 1 回**でやる
 (全位置同時 mask は文脈そのものを殺してしまい予測がほぼ一様になるため不可。1 位置ずつの
 厳密版を総位置数バッチに畳んでコスト 1 forward にしている)。
"""

from __future__ import annotations

_MODEL: dict = {}


def _load(cfg):
    from .config import resolve_device

    repo, device = cfg.get("lm.model"), resolve_device(cfg)
    if (repo, device) not in _MODEL:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model = AutoModelForMaskedLM.from_pretrained(repo).eval().to(device)
        _MODEL[(repo, device)] = (model, AutoTokenizer.from_pretrained(repo))
    return _MODEL[(repo, device)]


def score_text(r, cfg) -> None:
    """Record.text の MLM 擬似 PPL を lm_ppl に記録する (失敗は lm_errors に寄せる)。"""
    try:
        import numpy as np
        import torch

        model, tok = _load(cfg)
        enc = tok(r.text or "", return_tensors="pt", truncation=True, max_length=512)
        ids = enc["input_ids"][0]
        special = set(tok.all_special_ids)
        pos = [i for i in range(len(ids)) if int(ids[i]) not in special]
        if len(pos) < 2:
            return  # 短すぎて測れない (None のまま = ゲート有効時のみ検証不能)
        # 1 位置ずつ mask した行をバッチに畳む (全位置同時 mask は文脈を殺して予測がほぼ一様になるため不可)
        rows = ids.unsqueeze(0).repeat(len(pos), 1)
        rows[torch.arange(len(pos)), torch.tensor(pos)] = tok.mask_token_id
        attn = enc["attention_mask"].expand(len(pos), -1)
        with torch.inference_mode():
            logits = model(
                input_ids=rows.to(model.device), attention_mask=attn.to(model.device)
            ).logits  # (P, T, V)
        sel = logits[torch.arange(len(pos)), torch.tensor(pos)].float()  # mask 位置の予測のみ
        lp = torch.log_softmax(sel, dim=-1)
        nll = -lp.gather(-1, ids[pos][:, None].to(model.device)).mean()
        ppl = float(np.exp(float(nll)))
        r.scores["lm_ppl"] = round(ppl, 1) if np.isfinite(ppl) else None
    except Exception as e:  # noqa: BLE001
        r.scores.setdefault("lm_errors", []).append(repr(e))
