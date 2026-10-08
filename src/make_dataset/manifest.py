"""データ契約: manifest.jsonl 1行 = 1発話レコード (DESIGN.md「データ契約」).

全 Stage はこのレコードの scores にスコアを積み上げ、最終段でのみ閾値で採用/除外を決める。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path


@dataclass
class Record:
    """manifest.jsonl 1行."""

    utt_id: str
    src: str
    start: float
    end: float
    text: str = ""
    lang: str = "JP"
    scores: dict = field(default_factory=dict)
    dup_group_id: str | None = None
    dup_reason: str | None = None
    status: str = "pending"  # pending -> adopted / excluded (Stage 6 で確定)
    dropped_by: str | None = None

    def to_dict(self) -> dict:
        return {
            "utt_id": self.utt_id,
            "src": self.src,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "lang": self.lang,
            "scores": self.scores,
            "dup_group_id": self.dup_group_id,
            "dup_reason": self.dup_reason,
            "status": self.status,
            "dropped_by": self.dropped_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
