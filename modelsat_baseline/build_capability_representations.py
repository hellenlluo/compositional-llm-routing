#!/usr/bin/env python3
"""
Build MODEL-SAT-style capability representations for all 10 models.

Methodology (following MODEL-SAT paper, arXiv:2502.17282):
  1. Collect per-model accuracy on all 57 MMLU subcategories from
     embedllm-eval/results/{model}/summary.json
  2. Rank subcategories by variance of accuracy across the 10-model zoo.
     Tasks where models are most split (half correct, half incorrect) carry
     the most distinguishing signal — consistent with the paper's core task
     sampling criterion.
  3. Keep the top 50 most discriminative subcategories.
  4. Format each model's scores as a natural-language capability string,
     matching the MODEL-SAT capability instruction format:
       "The model achieves X% on <task>, Y% on <task>, ..."

Output: capability_representations.json
"""

import json
import os
import statistics

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_DIR, "embedllm-eval", "results")
OUT_FILE     = os.path.join(SCRIPT_DIR, "modelsat_data", "capability_representations.json")

MODELS = [
    "deepseek-r1-distill-llama-8b",
    "llama-3.1-8b-instruct",
    "llama-3.1-nemotron-nano-8b",
    "mathstral-7b",
    "medgemma-4b-it",
    "mistral-7b-instruct-v0.3",
    "phi-4-mini-instruct",
    "qwen1.5-0.5b-chat",
    "qwen3-30b-a3b",
    "qwen3-4b-thinking-2507",
]

TOP_K = 50  # number of MMLU tasks to keep (MODEL-SAT uses 50)


def pretty_category(cat: str) -> str:
    """Convert mmlu_high_school_mathematics -> High School Mathematics."""
    name = cat.removeprefix("mmlu_")
    return name.replace("_", " ").title()


def load_mmlu_accuracies(model: str) -> dict[str, float]:
    path = os.path.join(RESULTS_DIR, model, "summary.json")
    with open(path) as f:
        data = json.load(f)
    return {
        k: v["accuracy"]
        for k, v in data["accuracy_by_category"].items()
        if k.startswith("mmlu_")
    }


def select_top_k_by_variance(
    all_accs: dict[str, dict[str, float]], k: int
) -> list[str]:
    """
    Rank all MMLU subcategories by variance of accuracy across models,
    return the top-k most discriminative ones.
    """
    categories = list(next(iter(all_accs.values())).keys())
    variances = {}
    for cat in categories:
        scores = [all_accs[m][cat] for m in all_accs]
        variances[cat] = statistics.variance(scores)

    ranked = sorted(variances, key=lambda c: variances[c], reverse=True)
    print(f"\nTop-{k} most discriminative MMLU subcategories (by variance):")
    for i, cat in enumerate(ranked[:k], 1):
        scores = [all_accs[m][cat] for m in all_accs]
        mean   = statistics.mean(scores)
        var    = variances[cat]
        print(f"  {i:2d}. {pretty_category(cat):<45s}  mean={mean:.2%}  var={var:.4f}")

    print(f"\nDropped (low variance):")
    for cat in ranked[k:]:
        scores = [all_accs[m][cat] for m in all_accs]
        mean   = statistics.mean(scores)
        var    = variances[cat]
        print(f"      {pretty_category(cat):<45s}  mean={mean:.2%}  var={var:.4f}")

    return ranked[:k]


def build_capability_string(
    model: str, accs: dict[str, float], selected_cats: list[str]
) -> str:
    """
    Build a MODEL-SAT-style natural language capability representation.
    Format: "The model achieves X% on <Task>, Y% on <Task>, ..."
    """
    parts = [
        f"{round(accs[cat] * 100)}% on {pretty_category(cat)}"
        for cat in selected_cats
    ]
    return "The model achieves " + ", ".join(parts) + "."


def main():
    print("Loading MMLU accuracies for all models...")
    all_accs = {model: load_mmlu_accuracies(model) for model in MODELS}

    # Verify all models have the same categories
    cat_sets = [set(v.keys()) for v in all_accs.values()]
    assert all(s == cat_sets[0] for s in cat_sets), "Category mismatch across models!"
    print(f"  {len(cat_sets[0])} MMLU categories found, consistent across all {len(MODELS)} models.")

    # Select top-50 most discriminative categories
    selected_cats = select_top_k_by_variance(all_accs, TOP_K)

    # Build capability representations
    representations = {}
    for model in MODELS:
        cap_str = build_capability_string(model, all_accs[model], selected_cats)
        representations[model] = {
            "capability_string": cap_str,
            "selected_categories": selected_cats,
            "accuracy_by_selected_category": {
                cat: all_accs[model][cat] for cat in selected_cats
            },
        }

    # Save
    with open(OUT_FILE, "w") as f:
        json.dump(representations, f, indent=2)
    print(f"\nSaved capability representations to: {OUT_FILE}")

    # Print a sample
    print("\n--- Sample capability representation (llama-3.1-8b-instruct) ---")
    print(representations["llama-3.1-8b-instruct"]["capability_string"])


if __name__ == "__main__":
    main()
