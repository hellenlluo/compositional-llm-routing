"""Evaluate all 8 models on one category from the EmbedLLM prompt set.

Usage (on the cluster):
  python embedllm_eval/run_category.py --category gpqa_main_n_shot
  python embedllm_eval/run_category.py --category asdiv --backend hf --device cuda
  python embedllm_eval/run_category.py --category mmlu_anatomy --models qwen1.5-0.5b-chat phi-4-mini-instruct

Designed so each SLURM job handles one category across all (or selected) models.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROMPTS_FILE = REPO_ROOT / "data_preprocessing" / "embedllm-sampled" / "prompts_subset.jsonl"
MODELS_CFG = REPO_ROOT / "eval" / "configs" / "models.yaml"
RESULTS_DIR = REPO_ROOT / "embedllm_eval" / "results"

MODEL_IDS = [
    "qwen3-4b-thinking-2507",
    "deepseek-r1-distill-llama-8b",
    "medgemma-4b-it",
    "llama-3.1-8b-instruct",
    "qwen3-30b-a3b",
    "phi-4-mini-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen1.5-0.5b-chat",
]


def load_models_cfg() -> dict:
    return yaml.safe_load(MODELS_CFG.read_text())


def get_model_cfg(models_cfg: dict, model_id: str) -> dict:
    for m in models_cfg["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"unknown model id: {model_id}")


def load_category_questions(category: str) -> list[dict]:
    """Load all prompts for the given category, sorted by prompt_id."""
    questions = []
    with PROMPTS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["category"] == category:
                questions.append(entry)
    questions.sort(key=lambda q: q["prompt_id"])
    return questions


def extract_answer(response: str, gold_answer) -> str:
    """Best-effort extraction of the model's answer from its response.

    Strategy:
    1. Look for an explicit 'ANSWER:' line (matches existing eval convention).
    2. Look for letter-in-parentheses pattern like (A), (B), etc.
    3. Fall back to the first non-empty token of the response.
    """
    text = response.strip()

    answer_match = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    if answer_match:
        text = answer_match.group(1).strip()

    if isinstance(gold_answer, list) or (isinstance(gold_answer, str) and len(gold_answer) == 1 and gold_answer.isalpha()):
        letter = re.search(r"\(?([A-G])\)?", text)
        if letter:
            return letter.group(1).upper()

    return text.split("\n")[0].strip().rstrip(".").strip()


def check_correct(extracted: str, gold_answer) -> int:
    """Return 1 if the extracted answer matches gold, 0 otherwise.

    gold_answer can be:
      - a single letter string like "B"
      - a numeric string like "366"
      - a descriptive string like "19 (peaches)"
      - a list of acceptable letters like ["A", "C", "F"]
    """
    extracted_clean = extracted.strip().upper()

    if isinstance(gold_answer, list):
        return 1 if extracted_clean in [g.strip().upper() for g in gold_answer] else 0

    gold_str = str(gold_answer).strip()

    if len(gold_str) == 1 and gold_str.isalpha():
        return 1 if extracted_clean == gold_str.upper() else 0

    if extracted_clean == gold_str.upper():
        return 1

    num_match = re.match(r"^[\d,]+\.?\d*", gold_str)
    if num_match:
        gold_num = num_match.group(0).replace(",", "")
        extracted_num = re.match(r"^[\d,]+\.?\d*", extracted_clean.replace(",", ""))
        if extracted_num and extracted_num.group(0) == gold_num:
            return 1

    return 0


def make_backend(model_cfg: dict, backend: str = "auto", device: str = "cuda"):
    """Reuses the existing inference module from eval/."""
    from eval.inference import make_backend as _make_backend
    return _make_backend(model_cfg, backend=backend, device=device)


def run_model_on_questions(
    backend,
    model_id: str,
    questions: list[dict],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    """Run a single model on a list of questions. Returns per-question result dicts."""
    from eval.inference import timed_generate

    backend.gen_kwargs.max_new_tokens = max_new_tokens
    backend.gen_kwargs.temperature = 0.0

    results = []
    prompts_all = [q["prompt"] for q in questions]

    for i in range(0, len(prompts_all), batch_size):
        batch_prompts = prompts_all[i : i + batch_size]
        batch_questions = questions[i : i + batch_size]

        responses, wall_times = timed_generate(backend, batch_prompts)

        for q, resp, wall_ms in zip(batch_questions, responses, wall_times):
            extracted = extract_answer(resp, q["gold_answer"])
            correct = check_correct(extracted, q["gold_answer"])
            results.append({
                "prompt_id": q["prompt_id"],
                "model_id": model_id,
                "category": q["category"],
                "family": q["family"],
                "correct": correct,
                "extracted_answer": extracted,
                "gold_answer": q["gold_answer"],
                "response": resp,
                "wall_ms": wall_ms,
            })

        done = min(i + batch_size, len(prompts_all))
        print(f"  [{model_id}] {done}/{len(prompts_all)} questions processed")

    return results


def save_results(model_id: str, category: str, results: list[dict]) -> Path:
    out_dir = RESULTS_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{category}.json"
    results_sorted = sorted(results, key=lambda r: r["prompt_id"])
    out_path.write_text(json.dumps(results_sorted, indent=2, ensure_ascii=False))
    print(f"  saved {out_path.relative_to(REPO_ROOT)} ({len(results)} entries)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate models on one EmbedLLM category")
    ap.add_argument("--category", required=True, help="Category name from prompts_subset.jsonl")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model ids to evaluate. Default: all 8 models.")
    ap.add_argument("--backend", choices=["auto", "hf", "vllm"], default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    questions = load_category_questions(args.category)
    if not questions:
        print(f"No questions found for category '{args.category}'")
        print("Available categories can be listed with: python embedllm_eval/list_categories.py")
        return

    print(f"Category: {args.category}")
    print(f"  Questions: {len(questions)}")
    print(f"  Family: {questions[0]['family']}")

    models_cfg = load_models_cfg()
    pick_models = args.models or MODEL_IDS

    for model_id in pick_models:
        model_cfg = get_model_cfg(models_cfg, model_id)
        print(f"\n=== {model_id} ({model_cfg['hf_repo']}) ===")

        t0 = time.monotonic()
        backend = make_backend(model_cfg, backend=args.backend, device=args.device)
        try:
            results = run_model_on_questions(
                backend, model_id, questions,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            save_results(model_id, args.category, results)

            n_correct = sum(r["correct"] for r in results)
            print(f"  accuracy: {n_correct}/{len(results)} = {n_correct / len(results):.3f}")
        finally:
            backend.close()

        elapsed = time.monotonic() - t0
        print(f"  elapsed: {elapsed:.1f}s")

    print(f"\nDone evaluating category '{args.category}'")


if __name__ == "__main__":
    main()
