"""Load the local medical knowledge base (JSONL) and split it into passages/sentences.

Each line of a *.jsonl file in KNOWLEDGE_BASE_DIR is:
    {"id": "...", "title": "...", "text": "...", "source": "..."}
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def load_passages(kb_dir: Path) -> list[Passage]:
    files = sorted(Path(kb_dir).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl knowledge-base files found in {kb_dir}")

    passages: list[Passage] = []
    seen_ids: set[str] = set()
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    passage = Passage(
                        id=str(row["id"]),
                        title=str(row["title"]),
                        text=str(row["text"]),
                        source=str(row.get("source", "unknown")),
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"{path}:{line_no}: invalid knowledge-base row ({exc})") from exc
                if passage.id in seen_ids:
                    raise ValueError(f"{path}:{line_no}: duplicate passage id {passage.id!r}")
                seen_ids.add(passage.id)
                passages.append(passage)
    return passages
