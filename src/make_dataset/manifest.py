"""データ契約: manifest.jsonl 1行 = 1発話レコード (DESIGN.md「データ契約」).

全 Stage はこのレコードの scores にスコアを積み上げ、最終段でのみ閾値で採用/除外を決める。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any


def _finite_or_none(v: Any) -> Any:
    """A2: NaN / ±inf が JSON に裸の NaN/-Infinity として混入するのを防ぐ (非有限値は null 化).

    「異常値ほどゲートを通る」(nan < 閾値 == False) を起きさせないため、記録時点で None にして
    Stage 6 側で「スコア欠損=検証不能=不合格」扱いにする。
    """
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, list):
        return [_finite_or_none(x) for x in v]
    return v


@dataclass
class Record:
    """manifest.jsonl 1行."""

    utt_id: str
    src: str
    start: float
    end: float
    text: str = ""
    lang: str = "JP"
    scores: dict[str, Any] = field(
        default_factory=dict
    )  # スコア袋。float / list / bool / errors list 等が混在するので Any
    dup_group_id: str | None = None
    dup_reason: str | None = None
    status: str = "pending"  # pending -> adopted / excluded / pending_review (Stage 6 で確定)
    dropped_by: list[str] | None = None  # E4: 全失格理由を最後まで残す (単数理由で隠さない)
    review_reason: str | None = None  # E3/E6: ユーザー辞書候補等の人手レビュー行き先

    def to_dict(self) -> dict[str, Any]:
        return {
            "utt_id": self.utt_id,
            "src": self.src,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "lang": self.lang,
            "scores": {k: _finite_or_none(v) for k, v in self.scores.items()},
            "dup_group_id": self.dup_group_id,
            "dup_reason": self.dup_reason,
            "status": self.status,
            "dropped_by": self.dropped_by,
            "review_reason": self.review_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        return cls(**d)


def read_manifest(path: Path) -> list[Record]:
    fields = {f.name for f in dc_fields(Record)}
    with open(path, encoding="utf-8") as f:
        return [
            Record.from_dict({k: v for k, v in json.loads(line).items() if k in fields})
            for line in f
            if line.strip()
        ]


def write_manifest(records: list[Record], path: Path) -> None:
    """A2: allow_nan=False で書いて、非有限値は上で null 化済み。不正 JSON を二度と出さない."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
