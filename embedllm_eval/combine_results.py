"""Combine per-model per-category result JSONs into unified outputs.

Produces:
  embedllm_eval/results/all_results.json
    All individual entries sorted by prompt_id, one list.

  embedllm_eval/results/accuracy_summary.json
    Per-model per-category accuracy table, plus per-model per-family aggregates
    and overall per-model accuracy.

Usage:
  python embedllm_eval/combine_results.py
  python embedllm_eval/combine_results.py --results-dir embedllm_eval/results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "embedllm_eval" / "results"


def collect_all_results(results_dir: Path) -> list[dict]:
    """Walk results_dir/<model_id>/<category>.json and collect all entries."""
    all_entries = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for cat_file in sorted(model_dir.glob("*.json")):
            if cat_file.name in ("all_results.json", "accuracy_summary.json"):
                continue
            entries = json.loads(cat_file.read_text())
            all_entries.extend(entries)
    return all_entries


def build_accuracy_summary(all_entries: list[dict]) -> dict:
    """Build a nested accuracy summary from all result entries."""
    by_model_cat: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    cat_to_family: dict[str, str] = {}

    for e in all_entries:
        by_model_cat[e["model_id"]][e["category"]].append(e["correct"])
        cat_to_family[e["category"]] = e.get("family", "unknown")

    summary = {
        "per_model_per_category": {},
        "per_model_per_family": {},
        "per_model_overall": {},
    }

    for model_id in sorted(by_model_cat):
        cat_accs = {}
        family_agg: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

        for cat in sorted(by_model_cat[model_id]):
            scores = by_model_cat[model_id][cat]
            n_correct = sum(scores)
            n_total = len(scores)
            acc = n_correct / n_total if n_total > 0 else 0.0
            cat_accs[cat] = {
                "n_correct": n_correct,
                "n_total": n_total,
                "accuracy": round(acc, 4),
            }
            family = cat_to_family.get(cat, "unknown")
            family_agg[family]["correct"] += n_correct
            family_agg[family]["total"] += n_total

        summary["per_model_per_category"][model_id] = cat_accs

        family_accs = {}
        for family in sorted(family_agg):
            c = family_agg[family]["correct"]
            t = family_agg[family]["total"]
            family_accs[family] = {
                "n_correct": c,
                "n_total": t,
                "accuracy": round(c / t, 4) if t > 0 else 0.0,
            }
        summary["per_model_per_family"][model_id] = family_accs

        total_correct = sum(s for scores in by_model_cat[model_id].values() for s in scores)
        total_qs = sum(len(scores) for scores in by_model_cat[model_id].values())
        summary["per_model_overall"][model_id] = {
            "n_correct": total_correct,
            "n_total": total_qs,
            "accuracy": round(total_correct / total_qs, 4) if total_qs > 0 else 0.0,
        }

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine EmbedLLM evaluation results")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    all_entries = collect_all_results(results_dir)
    if not all_entries:
        print("No result files found.")
        return

    print(f"Collected {len(all_entries)} entries")

    all_sorted = sorted(all_entries, key=lambda e: (e["prompt_id"], e["model_id"]))
    all_out = results_dir / "all_results.json"
    all_out.write_text(json.dumps(all_sorted, indent=2, ensure_ascii=False))
    print(f"Wrote {all_out.relative_to(REPO_ROOT)}")

    summary = build_accuracy_summary(all_entries)
    summary_out = results_dir / "accuracy_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_out.relative_to(REPO_ROOT)}")

    print("\n=== Per-Model Overall Accuracy ===")
    for model_id, stats in sorted(summary["per_model_overall"].items()):
        print(f"  {model_id:40s}  {stats['n_correct']:4d}/{stats['n_total']:4d}  = {stats['accuracy']:.4f}")

    print("\n=== Per-Model Per-Family Accuracy ===")
    families = sorted({f for fam_dict in summary["per_model_per_family"].values() for f in fam_dict})
    header = f"  {'Model':<40s}" + "".join(f"  {f:>10s}" for f in families)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model_id in sorted(summary["per_model_per_family"]):
        row = f"  {model_id:<40s}"
        for f in families:
            stats = summary["per_model_per_family"][model_id].get(f)
            if stats:
                row += f"  {stats['accuracy']:>10.4f}"
            else:
                row += f"  {'N/A':>10s}"
        print(row)


if __name__ == "__main__":
    main()
