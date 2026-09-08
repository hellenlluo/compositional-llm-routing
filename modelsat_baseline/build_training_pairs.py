#!/usr/bin/env python3
"""
Build MODEL-SAT training pairs from per-model judgments.

For each of the 10 models, reads embedllm-eval/results/{model}/all_results.json
and emits one JSONL line per (model, question) pair:

  {
    "model_name":        str,   # model id
    "prompt_id":         int,   # stable question identifier
    "prompt":            str,   # raw question text (no system prompt)
    "score":             int,   # 1 = correct, 0 = incorrect
    "capability_string": str,   # MODEL-SAT cₘ from capability_representations.json
    "stage":             int    # 1 = in-domain (MMLU-aligned), 2 = out-of-domain
  }

Stage labelling follows the paper's two-stage learning strategy
(arXiv:2502.17282, "Learning Strategy"):

  Stage 1 — in-domain training instructions "primarily sourced from the
            same category as the MMLU dataset". Used to fit only the
            connector and establish capability→instruction alignment.
  Stage 2 — everything else (out-of-domain). Used when all parameters
            are unfrozen.

Output: modelsat_baseline/modelsat_data/training_pairs.jsonl
"""

import json
import os
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_DIR, "embedllm-eval", "results")
DATA_DIR     = os.path.join(SCRIPT_DIR, "modelsat_data")
CAP_REP_FILE = os.path.join(DATA_DIR, "capability_representations.json")
OUT_FILE     = os.path.join(DATA_DIR, "training_pairs.jsonl")

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


def load_capability_strings() -> dict[str, str]:
    with open(CAP_REP_FILE) as f:
        data = json.load(f)
    return {model: data[model]["capability_string"] for model in MODELS}


def stage_for(category: str) -> int:
    """MMLU questions are in-domain (Stage 1); the rest is out-of-domain (Stage 2)."""
    return 1 if category.startswith("mmlu_") else 2


def load_records_with_stage(model: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, model, "all_results.json")
    with open(path) as f:
        records = json.load(f)
    for r in records:
        r["_stage"] = stage_for(r["category"])
    return records


def main() -> None:
    print("Loading capability strings...")
    cap_strings = load_capability_strings()

    # per_model_stats[model][stage] = {total, positives, negatives, pos_rate}
    per_model_stats: dict[str, dict[int, dict]] = {}
    per_model_prompt_ids: dict[str, set] = {}
    total_written = 0

    print(f"\nReading results from {RESULTS_DIR}")
    print(f"Output → {OUT_FILE}\n")

    with open(OUT_FILE, "w") as out_f:
        for model in MODELS:
            records = load_records_with_stage(model)
            cap_str = cap_strings[model]

            stats_by_stage: dict[int, dict] = {}
            for stage in (1, 2):
                stage_records = [r for r in records if r["_stage"] == stage]
                positives = sum(1 for r in stage_records if r["score"] == 1)
                stats_by_stage[stage] = {
                    "total":     len(stage_records),
                    "positives": positives,
                    "negatives": len(stage_records) - positives,
                    "pos_rate":  positives / len(stage_records) if stage_records else 0.0,
                }
            per_model_stats[model] = stats_by_stage
            per_model_prompt_ids[model] = {r["prompt_id"] for r in records}

            for r in records:
                line = {
                    "model_name":        model,
                    "prompt_id":         r["prompt_id"],
                    "prompt":            r["prompt"],
                    "score":             r["score"],
                    "capability_string": cap_str,
                    "stage":             r["_stage"],
                }
                out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                total_written += 1

    # ── Per-model stats (per stage) ─────────────────────────────────────────
    header = (
        f"{'Model':<40s}  "
        f"{'S1 Tot':>7}  {'S1 Pos':>7}  {'S1 Pos%':>7}  "
        f"{'S2 Tot':>7}  {'S2 Pos':>7}  {'S2 Pos%':>7}"
    )
    print(header)
    print("-" * len(header))
    for model in MODELS:
        s1 = per_model_stats[model][1]
        s2 = per_model_stats[model][2]
        print(
            f"{model:<40s}  "
            f"{s1['total']:>7}  {s1['positives']:>7}  {s1['pos_rate']:>7.1%}  "
            f"{s2['total']:>7}  {s2['positives']:>7}  {s2['pos_rate']:>7.1%}"
        )
    print("-" * len(header))
    print(f"{'TOTAL records written':<40s}  {total_written:>7}")

    # ── Prompt-ID overlap check ──────────────────────────────────────────────
    print("\nPrompt-ID overlap check across all 10 models:")
    reference_model = MODELS[0]
    reference_ids   = per_model_prompt_ids[reference_model]
    all_match = True
    for model in MODELS[1:]:
        ids = per_model_prompt_ids[model]
        if ids != reference_ids:
            extra   = ids - reference_ids
            missing = reference_ids - ids
            print(f"  MISMATCH — {model}: +{len(extra)} extra, -{len(missing)} missing IDs")
            all_match = False

    if all_match:
        print(f"  All 10 models share exactly the same {len(reference_ids)} prompt IDs. ✓")
    else:
        print("  WARNING: prompt-ID sets differ — check above.")
        sys.exit(1)

    print(f"\nDone. Wrote {total_written} training pairs to:\n  {OUT_FILE}")


if __name__ == "__main__":
    main()
