"""Replace the PIQA rows in prompts_subset.jsonl with freshly-sampled PIQA
validation items reformatted as 2-choice MCQ.

EmbedLLM's original PIQA prompts are loglikelihood-scored: the goal is shown
without the two candidate solutions, and correctness is determined by comparing
P(sol1 | goal) vs P(sol2 | goal). Our generative inference pipeline can't
reproduce that, so we substitute a fresh PIQA sample in explicit MCQ form so
the model picks A or B and we can letter-extract the answer.

Source:
  data_preprocessing/data/embedllm-datasets/PIQA/valid.jsonl       (1838 items)
  data_preprocessing/data/embedllm-datasets/PIQA/valid-labels.lst  (1838 labels, 0=sol1, 1=sol2)

Target:
  outputs/embedllm/prompts_subset.jsonl (modified in place; backup at .bak)

Run:
  python -m eval.embedllm_baseline.replace_piqa
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIQA_DIR = REPO_ROOT / "data_preprocessing" / "data" / "embedllm-datasets" / "PIQA"
SUBSET = REPO_ROOT / "outputs" / "embedllm" / "prompts_subset.jsonl"

MCQ_TEMPLATE = (
    "Question: {goal}\n"
    "Choices:\n"
    "A. {sol1}\n"
    "B. {sol2}\n"
    "Answer:"
)


def _load_piqa_valid() -> list[dict]:
    valid = [json.loads(l) for l in (PIQA_DIR / "valid.jsonl").read_text().splitlines() if l.strip()]
    labels = [int(l.strip()) for l in (PIQA_DIR / "valid-labels.lst").read_text().splitlines() if l.strip()]
    assert len(valid) == len(labels), f"valid {len(valid)} != labels {len(labels)}"
    return [{**item, "label": lbl, "source_idx": i} for i, (item, lbl) in enumerate(zip(valid, labels))]


def _format_mcq(item: dict) -> str:
    return MCQ_TEMPLATE.format(goal=item["goal"].strip(), sol1=item["sol1"].strip(), sol2=item["sol2"].strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", default=str(SUBSET))
    args = ap.parse_args()

    subset_path = Path(args.subset)
    rows = [json.loads(l) for l in subset_path.read_text().splitlines() if l.strip()]
    piqa_indices = [i for i, r in enumerate(rows) if r.get("family") == "PIQA"]
    print(f"found {len(piqa_indices)} PIQA rows in {subset_path.name}")

    valid = _load_piqa_valid()
    print(f"PIQA validation pool: {len(valid)} items")
    assert len(piqa_indices) <= len(valid), "not enough PIQA valid items to sample from"

    rng = random.Random(args.seed)
    sampled = rng.sample(valid, len(piqa_indices))

    piqa_indices_sorted = sorted(piqa_indices, key=lambda i: rows[i]["prompt_id"])
    for pos, item in zip(piqa_indices_sorted, sampled):
        rows[pos] = {
            "prompt_id": rows[pos]["prompt_id"],
            "category": "piqa",
            "family": "PIQA",
            "prompt": _format_mcq(item),
            "gold_answer": "A" if item["label"] == 0 else "B",
            "source": "piqa_valid_mcq",
            "source_idx": item["source_idx"],
        }

    backup = subset_path.with_suffix(subset_path.suffix + ".bak")
    shutil.copyfile(subset_path, backup)
    print(f"backed up original to {backup.name}")

    with subset_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ans_a = sum(1 for i in piqa_indices if rows[i]["gold_answer"] == "A")
    ans_b = sum(1 for i in piqa_indices if rows[i]["gold_answer"] == "B")
    print(f"wrote {len(rows)} total rows; replaced {len(piqa_indices)} PIQA rows (gold A={ans_a}, B={ans_b})")

    print("\nExample replaced row:")
    print(json.dumps(rows[piqa_indices_sorted[0]], indent=2, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()
