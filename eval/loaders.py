"""Dataset loaders that produce a canonical Record struct.

Every record yields the question, the gold answer, and a list of ordered
Subtask entries so the same prompt/runner code can handle all three datasets.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Subtask:
    idx: int
    question: str
    gold: str
    options: list[str] | None = None
    paragraph_support_title: str = ""


@dataclass
class Record:
    dataset: str
    id: str
    question: str
    gold: str
    gold_aliases: list[str] = field(default_factory=list)
    context_payload: dict = field(default_factory=dict)
    subtasks: list[Subtask] = field(default_factory=list)
    origin: str | None = None


def _iter_musique(path: Path) -> Iterator[Record]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            subs = [
                Subtask(idx=i, question=s["question"], gold=s["answer"])
                for i, s in enumerate(r.get("question_decomposition", []))
            ]
            yield Record(
                dataset="musique",
                id=r["id"],
                question=r["question"],
                gold=r["answer"],
                gold_aliases=list(r.get("answer_aliases") or []),
                context_payload={"paragraphs": r["paragraphs"]},
                subtasks=subs,
            )


def _iter_morehopqa(path: Path) -> Iterator[Record]:
    data = json.loads(path.read_text())
    for r in data:
        subs = [
            Subtask(idx=i, question=s["question"], gold=s["answer"],
                    paragraph_support_title=s.get("paragraph_support_title") or "")
            for i, s in enumerate(r.get("question_decomposition", []))
        ]
        yield Record(
            dataset="morehopqa",
            id=r["id"],
            question=r["question"],
            gold=r["answer"],
            gold_aliases=list(r.get("answer_aliases") or []),
            context_payload={"context": r["context"]},
            subtasks=subs,
        )


def _iter_stepcot(path: Path) -> Iterator[Record]:
    data = json.loads(path.read_text())
    for r in data:
        chain = r.get("vqa_chain", [])
        subs = [
            Subtask(
                idx=i,
                question=s["question"],
                gold=s["answer"],
                options=s.get("options"),
            )
            for i, s in enumerate(chain)
        ]
        final = chain[-1] if chain else None
        yield Record(
            dataset="stepcot",
            id=r["id"],
            question=final["question"] if final else "",
            gold=final["answer"] if final else "",
            gold_aliases=[],
            context_payload={
                "report": r["report"],
                "final_options": final.get("options") if final else None,
            },
            subtasks=subs,
            origin=r.get("origin"),
        )


_LOADERS = {
    "musique": _iter_musique,
    "morehopqa": _iter_morehopqa,
    "stepcot": _iter_stepcot,
}


def iter_records(dataset: str, path: str | Path) -> Iterator[Record]:
    if dataset not in _LOADERS:
        raise ValueError(f"unknown dataset flavor: {dataset}")
    return _LOADERS[dataset](Path(path))


def pilot_sample(records: list[Record], n: int, seed: int = 42) -> list[Record]:
    """Deterministic pilot subset. Sorted by id before sampling to be robust
    against upstream ordering changes."""
    records = sorted(records, key=lambda r: r.id)
    if n >= len(records):
        return records
    rng = random.Random(seed)
    idxs = rng.sample(range(len(records)), n)
    idxs.sort()
    return [records[i] for i in idxs]
