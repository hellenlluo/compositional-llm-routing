"""Replace the SocialIQA rows in prompts_subset.jsonl with freshly-sampled
SocialIQA test items reformatted as 3-choice MCQ.

EmbedLLM's original SocialIQA prompts are loglikelihood-scored: the context +
question are shown without any candidate answers, and correctness is
determined by comparing P(answer_i | prompt) across the three candidates. Our
generative inference pipeline can't reproduce that, so we substitute a fresh
SocialIQA sample in explicit MCQ form so the model picks A/B/C and we can
letter-extract the answer.

Source:
  data_preprocessing/data/embedllm-datasets/SocialIQA/socialIQa_v1.4_tst.jsonl
    fields: context, question, answerA, answerB, answerC, correct

Target:
  outputs/embedllm/prompts_subset.jsonl (modified in place; backup at .bak)

Run:
  python -m eval.embedllm_baseline.replace_social_iqa
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIQA_DIR = REPO_ROOT / "data_preprocessing" / "data" / "embedllm-datasets" / "SocialIQA"
SUBSET = REPO_ROOT / "outputs" / "embedllm" / "prompts_subset.jsonl"

MCQ_TEMPLATE = (
    "Context: {context}\n"
    "Question: {question}\n"
    "Choices:\n"
    "A. {answerA}\n"
    "B. {answerB}\n"
    "C. {answerC}\n"
    "Answer:"
)


def _load_siqa() -> list[dict]:
    items = [
        json.loads(l)
        for l in (SIQA_DIR / "socialIQa_v1.4_tst.jsonl").read_text().splitlines()
        if l.strip()
    ]
    return [{**it, "source_idx": i} for i, it in enumerate(items)]


def _format_mcq(item: dict) -> str:
    return MCQ_TEMPLATE.format(
        context=item["context"].strip(),
        question=item["question"].strip(),
        answerA=item["answerA"].strip(),
        answerB=item["answerB"].strip(),
        answerC=item["answerC"].strip(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", default=str(SUBSET))
    args = ap.parse_args()

    subset_path = Path(args.subset)
    rows = [json.loads(l) for l in subset_path.read_text().splitlines() if l.strip()]
    siqa_indices = [i for i, r in enumerate(rows) if r.get("family") == "SocialIQA"]
    print(f"found {len(siqa_indices)} SocialIQA rows in {subset_path.name}")

    pool = _load_siqa()
    print(f"SocialIQA test pool: {len(pool)} items")
    assert len(siqa_indices) <= len(pool), "not enough SocialIQA items to sample from"

    rng = random.Random(args.seed)
    sampled = rng.sample(pool, len(siqa_indices))

    siqa_indices_sorted = sorted(siqa_indices, key=lambda i: rows[i]["prompt_id"])
    for pos, item in zip(siqa_indices_sorted, sampled):
        rows[pos] = {
            "prompt_id": rows[pos]["prompt_id"],
            "category": "social_iqa",
            "family": "SocialIQA",
            "prompt": _format_mcq(item),
            "gold_answer": item["correct"].strip().upper(),
            "source": "social_iqa_tst_mcq",
            "source_idx": item["source_idx"],
        }

    backup = subset_path.with_suffix(subset_path.suffix + ".bak")
    shutil.copyfile(subset_path, backup)
    print(f"backed up original to {backup.name}")

    with subset_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    gold_hist = Counter(rows[i]["gold_answer"] for i in siqa_indices)
    print(f"wrote {len(rows)} total rows; replaced {len(siqa_indices)} SocialIQA rows "
          f"(gold A={gold_hist['A']}, B={gold_hist['B']}, C={gold_hist['C']})")

    print("\nExample replaced row:")
    print(json.dumps(rows[siqa_indices_sorted[0]], indent=2, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
