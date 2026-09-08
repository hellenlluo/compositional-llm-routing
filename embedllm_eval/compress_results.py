#!/usr/bin/env python3
"""
For each model folder in results/, produce two JSON files:
  - all_results.json  : flat list of every record from every category file
  - summary.json      : accuracy per category and per family
"""

import json
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def process_model(model_dir: str) -> None:
    model_name = os.path.basename(model_dir)
    print(f"Processing {model_name} ...", flush=True)

    all_records = []

    # Accumulators: {key: [correct, total]}
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(model_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  WARNING: could not parse {fname}: {e}")
                continue

        if not isinstance(records, list):
            print(f"  WARNING: {fname} is not a list, skipping")
            continue

        for rec in records:
            all_records.append(rec)
            score = rec.get("score", 0)
            cat = rec.get("category", "unknown")
            fam = rec.get("family", "unknown")
            by_category[cat][0] += int(score)
            by_category[cat][1] += 1
            by_family[fam][0] += int(score)
            by_family[fam][1] += 1

    # --- 1. Write concatenated JSON ---
    concat_path = os.path.join(model_dir, "all_results.json")
    with open(concat_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f)
    print(f"  Wrote {len(all_records)} records -> all_results.json")

    # --- 2. Build summary ---
    def make_accuracy_dict(acc_map):
        out = {}
        for key, (correct, total) in sorted(acc_map.items()):
            out[key] = {
                "correct": correct,
                "total": total,
                "accuracy": round(correct / total, 6) if total > 0 else None,
            }
        return out

    summary = {
        "model_name": model_name,
        "total_questions": len(all_records),
        "total_correct": sum(int(r.get("score", 0)) for r in all_records),
        "overall_accuracy": round(
            sum(int(r.get("score", 0)) for r in all_records) / len(all_records), 6
        ) if all_records else None,
        "accuracy_by_category": make_accuracy_dict(by_category),
        "accuracy_by_family": make_accuracy_dict(by_family),
    }

    summary_path = os.path.join(model_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote summary.json  (overall accuracy: {summary['overall_accuracy']})")


def main():
    model_dirs = sorted(
        os.path.join(RESULTS_DIR, d)
        for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    )
    for model_dir in model_dirs:
        process_model(model_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
