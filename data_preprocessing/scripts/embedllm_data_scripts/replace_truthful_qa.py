"""Reformat the TruthfulQA rows in prompts_subset.jsonl into generative MCQ.

Original EmbedLLM TruthfulQA prompts are a 6-shot Q/A prefix followed by a
bare target question and `A:` — designed for loglikelihood scoring of each
candidate continuation (MC1/MC2). Our generative pipeline can't reproduce
that, so we rewrite each row as an explicit multiple-choice question:

  - Pool = every "Correct Answers" + "Incorrect Answers" entry (semicolon-
    separated) from TruthfulQA.csv for the matching question.
  - Choices are deterministically shuffled per-question (seeded by prompt_id)
    and labelled A, B, C, ...
  - `gold_answer` is a LIST of letters pointing at the correct-answer subset
    (multi-select, since several correct paraphrases are typically present).

Source:
  data_preprocessing/data/embedllm-datasets/TruthfulQA/TruthfulQA.csv
    columns: Type, Category, Question, Best Answer, Best Incorrect Answer,
             Correct Answers, Incorrect Answers, Source

Target:
  outputs/embedllm/prompts_subset.jsonl (modified in place; backup at .bak)

Run:
  python -m eval.embedllm_baseline.replace_truthful_qa
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import string
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TQA_CSV = REPO_ROOT / "data_preprocessing" / "data" / "embedllm-datasets" / "TruthfulQA" / "TruthfulQA.csv"
SUBSET = REPO_ROOT / "outputs" / "embedllm" / "prompts_subset.jsonl"


def _norm(s: str) -> str:
    """Normalize a question string for matching."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _split_semi(s: str) -> list[str]:
    return [x.strip() for x in s.split(";") if x.strip()]


def _load_tqa_lookup() -> dict[str, dict]:
    """question (normalized) -> {question, correct: [...], incorrect: [...]}"""
    lookup: dict[str, dict] = {}
    with open(TQA_CSV) as f:
        for row in csv.DictReader(f):
            q = row["Question"].strip()
            lookup[_norm(q)] = {
                "question": q,
                "correct": _split_semi(row["Correct Answers"]),
                "incorrect": _split_semi(row["Incorrect Answers"]),
            }
    return lookup


def _extract_question(prompt: str) -> str:
    """Grab the final `Q: ... \\nA:` target question from an EmbedLLM TQA prompt."""
    tail = prompt.rstrip()
    if not tail.endswith("A:"):
        raise ValueError(f"prompt doesn't end in 'A:': {tail[-200:]!r}")
    body = tail[: -len("A:")].rstrip()  # drop trailing 'A:' and whitespace
    # The last `Q: ...` before the empty `A:` contains the target question.
    idx = body.rfind("\nQ:")
    if idx < 0:
        idx = 0 if body.startswith("Q:") else -1
    if idx < 0:
        raise ValueError(f"could not locate last Q: in prompt: {tail[-200:]!r}")
    q_line = body[idx:].lstrip("\n")
    assert q_line.startswith("Q:"), q_line[:50]
    return q_line[len("Q:") :].strip()


def _format_mcq(question: str, choices: list[str]) -> str:
    letters = string.ascii_uppercase
    assert len(choices) <= len(letters), f"too many choices ({len(choices)}) for A-Z labels"
    lines = [f"Question: {question}", "Choices:"]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(SUBSET))
    args = ap.parse_args()

    subset_path = Path(args.subset)
    rows = [json.loads(l) for l in subset_path.read_text().splitlines() if l.strip()]
    tqa_indices = [i for i, r in enumerate(rows) if r.get("family") == "TruthfulQA"]
    print(f"found {len(tqa_indices)} TruthfulQA rows in {subset_path.name}")

    lookup = _load_tqa_lookup()
    print(f"TruthfulQA CSV: {len(lookup)} questions")

    matched = 0
    missing: list[tuple[int, str]] = []
    choice_sizes: Counter[int] = Counter()
    correct_sizes: Counter[int] = Counter()

    for i in tqa_indices:
        row = rows[i]
        q = _extract_question(row["prompt"])
        entry = lookup.get(_norm(q))
        if entry is None:
            missing.append((row["prompt_id"], q))
            continue

        correct = list(entry["correct"])
        incorrect = list(entry["incorrect"])
        if not correct or not incorrect:
            # Degenerate row; treat as missing.
            missing.append((row["prompt_id"], q))
            continue

        # Deterministic shuffle seeded by prompt_id for reproducibility.
        rng = random.Random(row["prompt_id"])
        all_choices = correct + incorrect
        rng.shuffle(all_choices)

        correct_set = set(correct)
        letters = string.ascii_uppercase
        gold = [letters[k] for k, ch in enumerate(all_choices) if ch in correct_set]

        rows[i] = {
            "prompt_id": row["prompt_id"],
            "category": "truthfulqa_mc",
            "family": "TruthfulQA",
            "prompt": _format_mcq(entry["question"], all_choices),
            "gold_answer": gold,
            "source": "truthfulqa_csv_mcq",
            "source_question": entry["question"],
        }
        matched += 1
        choice_sizes[len(all_choices)] += 1
        correct_sizes[len(correct)] += 1

    backup = subset_path.with_suffix(subset_path.suffix + ".bak")
    shutil.copyfile(subset_path, backup)
    print(f"backed up current file to {backup.name}")

    with subset_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nmatched {matched} / {len(tqa_indices)} TruthfulQA rows")
    if missing:
        print(f"missing {len(missing)} rows:")
        for pid, q in missing[:10]:
            print(f"  prompt_id={pid} q={q!r}")
    print(f"\nchoices-per-question dist: {dict(sorted(choice_sizes.items()))}")
    print(f"correct-per-question dist: {dict(sorted(correct_sizes.items()))}")

    print("\nExample replaced row:")
    sample = next(rows[i] for i in tqa_indices if "gold_answer" in rows[i])
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
