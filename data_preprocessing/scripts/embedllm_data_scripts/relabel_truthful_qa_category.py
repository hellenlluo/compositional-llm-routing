"""Overwrite the `category` field of TruthfulQA rows in prompts_subset.jsonl
with the fine-grained `Category` column from TruthfulQA.csv
(e.g. "Misconceptions", "Stereotypes", "Indexical Error: Time", ...).

Matches by normalized question text. Rows whose question isn't in the CSV
are left untouched and reported.

Run:
  python -m eval.embedllm_baseline.relabel_truthful_qa_category
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TQA_CSV = REPO_ROOT / "data_preprocessing" / "data" / "embedllm-datasets" / "TruthfulQA" / "TruthfulQA.csv"
SUBSET = REPO_ROOT / "outputs" / "embedllm" / "prompts_subset.jsonl"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _extract_question(prompt: str) -> str | None:
    """Target question is the last `Q: ...\\nA:` block for raw EmbedLLM TQA prompts."""
    tail = prompt.rstrip()
    if not tail.endswith("A:"):
        return None
    body = tail[: -len("A:")].rstrip()
    idx = body.rfind("\nQ:")
    if idx < 0 and body.startswith("Q:"):
        idx = 0
    if idx < 0:
        return None
    q_line = body[idx:].lstrip("\n")
    return q_line[len("Q:") :].strip() if q_line.startswith("Q:") else None


def main() -> None:
    rows = [json.loads(l) for l in SUBSET.read_text().splitlines() if l.strip()]

    with open(TQA_CSV) as f:
        cat_by_q = {_norm(r["Question"]): r["Category"].strip() for r in csv.DictReader(f)}

    tqa_idx = [i for i, r in enumerate(rows) if r.get("family") == "TruthfulQA"]
    updated = 0
    missing = []
    cat_hist: Counter[str] = Counter()

    for i in tqa_idx:
        r = rows[i]
        q = r.get("source_question") or _extract_question(r["prompt"])
        if q is None:
            missing.append((r["prompt_id"], None))
            continue
        cat = cat_by_q.get(_norm(q))
        if cat is None:
            missing.append((r["prompt_id"], q))
            continue
        r["category"] = cat
        cat_hist[cat] += 1
        updated += 1

    backup = SUBSET.with_suffix(SUBSET.suffix + ".bak")
    shutil.copyfile(SUBSET, backup)
    print(f"backed up current file to {backup.name}")

    with SUBSET.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nupdated {updated} / {len(tqa_idx)} TruthfulQA rows")
    if missing:
        print(f"unmatched {len(missing)}:")
        for pid, q in missing:
            print(f"  prompt_id={pid} q={q!r}")
    print("\ncategory histogram:")
    for cat, n in cat_hist.most_common():
        print(f"  {n:>3}  {cat}")


if __name__ == "__main__":
    main()
