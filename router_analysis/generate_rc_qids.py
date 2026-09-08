#!/usr/bin/env python3
"""
Generate the set of heterogeneous R+C question IDs (RETRIEVAL + COMPUTATION
type-set, complex ≥3 hops) for morehopqa and musique.

Output: outputs/rc_qids.json  — {"morehopqa": [...], "musique": [...]}

Uses the same filter as heterogeneous_accuracy.py:
  - complex: len(question_decomposition) >= 3
  - typeset == {RETRIEVAL, COMPUTATION}  (exactly these two labels)
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CLEANED_PATHS = {
    "morehopqa": REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data" / "morehopqa_cleaned.json",
    "musique":   REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data" / "musique_cleaned.jsonl",
}

RC_TYPESET = frozenset({"RETRIEVAL", "COMPUTATION"})


def load_cleaned(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f]
    return json.loads(path.read_text())


def main() -> None:
    rc_qids: dict[str, list[str]] = {}

    for dataset, path in CLEANED_PATHS.items():
        items = load_cleaned(path)
        qids = []
        for item in items:
            decomp = item.get("question_decomposition", [])
            if len(decomp) < 3:
                continue
            typeset = frozenset(s.get("label", "UNLABELLED") for s in decomp)
            if typeset == RC_TYPESET:
                qids.append(item["id"])
        rc_qids[dataset] = qids
        print(f"{dataset}: {len(qids)} R+C questions (complex ≥3 hops)")

    out_path = REPO_ROOT / "outputs" / "rc_qids.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rc_qids, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
