"""Fill in the TruthfulQA rows that weren't matched by
`replace_truthful_qa.py` (their questions aren't in TruthfulQA.csv).

Strategy: randomly sample replacements from TruthfulQA.csv, excluding every
question that's already present in prompts_subset.jsonl (so no duplicates),
and rewrite those rows in the same MCQ format used for the 64 matched rows.

Run:
  python -m eval.embedllm_baseline.fill_missing_truthful_qa
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import string
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TQA_CSV = REPO_ROOT / "data_preprocessing" / "data" / "embedllm-datasets" / "TruthfulQA" / "TruthfulQA.csv"
SUBSET = REPO_ROOT / "outputs" / "embedllm" / "prompts_subset.jsonl"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _split_semi(s: str) -> list[str]:
    return [x.strip() for x in s.split(";") if x.strip()]


def _format_mcq(question: str, choices: list[str]) -> str:
    letters = string.ascii_uppercase
    assert len(choices) <= len(letters), f"too many choices ({len(choices)})"
    out = [f"Question: {question}", "Choices:"]
    for letter, ch in zip(letters, choices):
        out.append(f"{letter}. {ch}")
    out.append("Answer:")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in SUBSET.read_text().splitlines() if l.strip()]

    # CSV pool keyed by normalized question.
    pool: dict[str, dict] = {}
    with open(TQA_CSV) as f:
        for r in csv.DictReader(f):
            q = r["Question"].strip()
            correct = _split_semi(r["Correct Answers"])
            incorrect = _split_semi(r["Incorrect Answers"])
            if not correct or not incorrect:
                continue
            pool[_norm(q)] = {
                "question": q,
                "category": r["Category"].strip(),
                "correct": correct,
                "incorrect": incorrect,
            }

    # Questions already used (by `source_question` on matched rows).
    used_norm = {
        _norm(r["source_question"])
        for r in rows
        if r.get("family") == "TruthfulQA" and "source_question" in r
    }
    print(f"already-used TruthfulQA questions: {len(used_norm)}")

    candidates = [v for k, v in pool.items() if k not in used_norm]
    print(f"csv pool size: {len(pool)} / available after dedup: {len(candidates)}")

    missing_idx = [
        i
        for i, r in enumerate(rows)
        if r.get("family") == "TruthfulQA" and "gold_answer" not in r
    ]
    print(f"unmatched TruthfulQA rows to fill: {len(missing_idx)}")
    assert len(candidates) >= len(missing_idx), "not enough candidates to fill without duplicates"

    rng = random.Random(args.seed)
    picks = rng.sample(candidates, len(missing_idx))

    # Stable pairing: order missing rows by prompt_id so reruns are deterministic.
    missing_idx_sorted = sorted(missing_idx, key=lambda i: rows[i]["prompt_id"])
    letters = string.ascii_uppercase
    for i, entry in zip(missing_idx_sorted, picks):
        pid = rows[i]["prompt_id"]
        choices = entry["correct"] + entry["incorrect"]
        random.Random(pid).shuffle(choices)  # per-prompt deterministic shuffle
        correct_set = set(entry["correct"])
        gold = [letters[k] for k, ch in enumerate(choices) if ch in correct_set]

        rows[i] = {
            "prompt_id": pid,
            "category": entry["category"],
            "family": "TruthfulQA",
            "prompt": _format_mcq(entry["question"], choices),
            "gold_answer": gold,
            "source": "truthfulqa_csv_mcq",
            "source_question": entry["question"],
        }

    backup = SUBSET.with_suffix(SUBSET.suffix + ".bak")
    shutil.copyfile(SUBSET, backup)
    print(f"backed up current file to {backup.name}")

    with SUBSET.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Integrity checks.
    seen: set[str] = set()
    dups: list[str] = []
    for r in rows:
        if r.get("family") == "TruthfulQA" and "source_question" in r:
            k = _norm(r["source_question"])
            if k in seen:
                dups.append(r["source_question"])
            seen.add(k)
    print(f"\nfinal TruthfulQA unique questions: {len(seen)}, duplicates: {len(dups)}")

    print("\nnewly filled rows:")
    for i in missing_idx_sorted:
        r = rows[i]
        print(f"  prompt_id={r['prompt_id']:>5} category={r['category']!r:<30} "
              f"gold={r['gold_answer']} q={r['source_question']!r}")


if __name__ == "__main__":
    main()
