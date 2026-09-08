"""Stitch per-category result JSONs into per-model complete performance files.

Reads:  embedllm_eval/results/<model_id>/<category>.json  (one per job)
Writes:
  embedllm_eval/results/<model_id>/all_results.json    — flat list of all entries
  embedllm_eval/results/summary.json                   — accuracy per model × category × family

Usage:
  python embedllm_eval/stitch_results.py
  python embedllm_eval/stitch_results.py --results-dir /path/to/results
  python embedllm_eval/stitch_results.py --model qwen3-4b-thinking-2507   # one model only
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
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

SKIP_FILENAMES = {"all_results.json", "summary.json"}


def collect_model_results(model_dir: Path) -> list[dict]:
    """Load all per-category JSONs for one model, skip stitched outputs."""
    records: list[dict] = []
    for path in sorted(model_dir.glob("*.json")):
        if path.name in SKIP_FILENAMES:
            continue
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                records.extend(data)
            else:
                print(f"  [WARN] unexpected format in {path.name}, skipping")
        except Exception as e:
            print(f"  [WARN] could not read {path}: {e}")
    return records


def build_summary(all_model_records: dict[str, list[dict]]) -> dict:
    """
    Returns nested accuracy summary:
      {
        model_id: {
          "overall":   {"n_correct": int, "n_total": int, "accuracy": float},
          "by_category": {category: {...}},
          "by_family":   {family:   {...}},
        }
      }
    """
    summary: dict = {}

    for model_id, records in all_model_records.items():
        by_cat: dict[str, list[int]]    = defaultdict(list)
        by_fam: dict[str, list[int]]    = defaultdict(list)
        all_scores: list[int]           = []

        for r in records:
            score = r.get("score", 0)
            by_cat[r["category"]].append(score)
            by_fam[r["family"]].append(score)
            all_scores.append(score)

        def _agg(scores: list[int]) -> dict:
            n   = len(scores)
            c   = sum(scores)
            return {"n_correct": c, "n_total": n, "accuracy": round(c / n, 4) if n else 0.0}

        summary[model_id] = {
            "overall":     _agg(all_scores),
            "by_category": {cat: _agg(sc) for cat, sc in sorted(by_cat.items())},
            "by_family":   {fam: _agg(sc) for fam, sc in sorted(by_fam.items())},
        }

    return summary


def print_summary_table(summary: dict) -> None:
    print("\n=== Overall accuracy per model ===")
    rows = [
        (mid, v["overall"]["n_correct"], v["overall"]["n_total"], v["overall"]["accuracy"])
        for mid, v in summary.items()
    ]
    rows.sort(key=lambda x: -x[3])
    col_w = max(len(r[0]) for r in rows) + 2
    print(f"  {'Model':<{col_w}}  Correct   Total   Accuracy")
    print(f"  {'-' * col_w}  -------   -----   --------")
    for mid, nc, nt, acc in rows:
        print(f"  {mid:<{col_w}}  {nc:>7}   {nt:>5}   {acc:.4f}")

    # Per-family table
    all_families = sorted({fam for v in summary.values() for fam in v["by_family"]})
    if all_families:
        print("\n=== Accuracy by family ===")
        fam_w = max(len(f) for f in all_families) + 2
        header = f"  {'Model':<{col_w}}" + "".join(f"  {f:<{fam_w}}" for f in all_families)
        print(header)
        print("  " + "-" * (col_w + len(all_families) * (fam_w + 2)))
        for mid, v in sorted(summary.items()):
            row = f"  {mid:<{col_w}}"
            for fam in all_families:
                acc = v["by_family"].get(fam, {}).get("accuracy", "-")
                cell = f"{acc:.3f}" if isinstance(acc, float) else acc
                row += f"  {cell:<{fam_w}}"
            print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stitch per-category result JSONs per model.")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="Root results directory (default: embedllm_eval/results)")
    ap.add_argument("--model", default=None,
                    help="Process only this model id (default: all 8 models)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    models_to_process = [args.model] if args.model else MODEL_IDS

    all_model_records: dict[str, list[dict]] = {}
    missing: list[str] = []

    for model_id in models_to_process:
        model_dir = results_dir / model_id
        if not model_dir.is_dir():
            print(f"[MISSING] {model_id} — no results directory found, skipping")
            missing.append(model_id)
            continue

        records = collect_model_results(model_dir)
        if not records:
            print(f"[EMPTY]   {model_id} — no category JSONs found yet")
            missing.append(model_id)
            continue

        # Sort by (category, prompt_id) for a deterministic output
        records.sort(key=lambda r: (r["category"], r["prompt_id"]))
        all_model_records[model_id] = records

        # Write all_results.json for this model
        out_path = model_dir / "all_results.json"
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(f"[OK]      {model_id} — {len(records)} entries → {out_path.relative_to(REPO_ROOT)}")

    if not all_model_records:
        print("\nNo results to stitch yet. Run jobs first.")
        return

    # Build and write cross-model summary
    summary = build_summary(all_model_records)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary written → {summary_path.relative_to(REPO_ROOT)}")

    print_summary_table(summary)

    if missing:
        print(f"\n[NOTE] {len(missing)} model(s) had no results yet: {missing}")


if __name__ == "__main__":
    main()
