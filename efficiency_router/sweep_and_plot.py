#!/usr/bin/env python3
"""
Run eval.py for every offset checkpoint (1-10) across model sets
(no_qwen3 = 8-model, small4 = 4-model), collect cost & accuracy metrics,
and produce a grid of cost-vs-accuracy scatter plots.

Plots produced (saved to outputs/effrouter_offset_plots/):
  One figure with a 2x2 grid of subplots:
    rows:    All questions | R+C questions
    columns: MoreHopQA     | MuSiQue
  Repeated for each model set (4-model, 8-model).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CKPT_DIR    = SCRIPT_DIR / "checkpoints"
TMP_DIR     = SCRIPT_DIR / "results" / "offset_sweep"
PLOT_DIR    = PROJECT_DIR / "outputs" / "effrouter_offset_plots"
RC_QIDS     = PROJECT_DIR / "outputs" / "rc_qids.json"
EVAL_PY     = SCRIPT_DIR / "eval.py"
EVAL_FROM_PREDS = SCRIPT_DIR / "eval_from_preds.py"
PYTHON      = sys.executable

DATASETS    = ["morehopqa", "musique"]
OFFSETS     = list(range(1, 11))

MODEL_SETS = {
    "8-model": "no_qwen3",
    "4-model": "small4",
}

# ---------------------------------------------------------------------------
# Step 1 — run eval for every checkpoint
# ---------------------------------------------------------------------------

def ckpt_path(variant: str, offset: int) -> Path:
    if variant == "no_qwen3":
        return CKPT_DIR / f"effrouter_no_qwen3_offset_{offset}.pt"
    elif variant == "small4":
        return CKPT_DIR / f"effrouter_small4_offset_{offset}.pt"
    raise ValueError(variant)


def run_eval(variant: str, offset: int) -> dict:
    """Run eval.py for one checkpoint; return parsed eval_results as dict keyed by dataset."""
    ckpt = ckpt_path(variant, offset)
    out_dir = TMP_DIR / variant / f"offset_{offset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_out  = out_dir / "eval_results.json"
    preds_out = out_dir / "predictions.json"

    if not eval_out.exists():
        print(f"  Running eval: {variant} offset={offset} ...")
        cmd = [
            PYTHON, str(EVAL_PY),
            "--ckpt",             str(ckpt),
            "--datasets",         *DATASETS,
            "--out",              str(eval_out),
            "--predictions-out",  str(preds_out),
        ]
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    FAILED:\n{result.stderr[-1000:]}")
            return {}
    else:
        print(f"  Cached: {variant} offset={offset}")

    return {r["dataset"]: r for r in json.loads(eval_out.read_text())}


def run_eval_rc(variant: str, offset: int) -> dict:
    """Run eval_from_preds.py with qid-filter for R+C; return dict keyed by dataset."""
    out_dir   = TMP_DIR / variant / f"offset_{offset}"
    preds_out = out_dir / "predictions.json"
    rc_out    = out_dir / "eval_results_rc.json"

    if not preds_out.exists():
        print(f"  No predictions for {variant} offset={offset}, skipping R+C")
        return {}

    if not rc_out.exists():
        print(f"  Running R+C eval: {variant} offset={offset} ...")
        rc_tmp_dir = out_dir / "rc_tmp"
        rc_tmp_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON, str(EVAL_FROM_PREDS),
            "--preds",      str(preds_out),
            "--out-dir",    str(rc_tmp_dir),
            "--qid-filter", str(RC_QIDS),
        ]
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    FAILED:\n{result.stderr[-1000:]}")
            return {}
        # eval_from_preds writes eval_results.json into out-dir
        src = rc_tmp_dir / "eval_results.json"
        if src.exists():
            rc_out.write_text(src.read_text())

    if not rc_out.exists():
        return {}
    return {r["dataset"]: r for r in json.loads(rc_out.read_text())}


# ---------------------------------------------------------------------------
# Step 2 — collect all results
# ---------------------------------------------------------------------------

def collect_results():
    """Returns nested dict: results[model_set_label][filter][dataset][offset] = (acc, cost_savings)"""
    results = {}
    for label, variant in MODEL_SETS.items():
        results[label] = {"all": {d: {} for d in DATASETS},
                          "rc":  {d: {} for d in DATASETS}}
        for offset in OFFSETS:
            all_r = run_eval(variant, offset)
            rc_r  = run_eval_rc(variant, offset)
            for ds in DATASETS:
                if ds in all_r:
                    r = all_r[ds]
                    results[label]["all"][ds][offset] = (
                        r["routing_accuracy"] * 100,
                        r["cost_savings_pct"],
                    )
                if ds in rc_r:
                    r = rc_r[ds]
                    results[label]["rc"][ds][offset] = (
                        r["routing_accuracy"] * 100,
                        r["cost_savings_pct"],
                    )
    return results


# ---------------------------------------------------------------------------
# Step 3 — plot
# ---------------------------------------------------------------------------

DS_LABELS  = {"morehopqa": "MoreHopQA", "musique": "MuSiQue"}
DS_COLORS  = {"morehopqa": "#2196F3",   "musique": "#FF5722"}
DS_MARKERS = {"morehopqa": "o",         "musique": "s"}
FILTER_LABELS = {"all": "All questions", "rc": "R+C questions"}


def make_figure(label: str, data: dict, plot_dir: Path):
    """One 2×2 figure for a single model set."""
    filters  = ["all", "rc"]
    datasets = DATASETS

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f"EffiRoute offset sweep — {label}", fontsize=13, fontweight="bold")

    for row_i, filt in enumerate(filters):
        for col_i, ds in enumerate(datasets):
            ax = axes[row_i][col_i]
            ax.set_title(f"{DS_LABELS[ds]}  ·  {FILTER_LABELS[filt]}", fontsize=10)
            ax.set_xlabel("Cost savings vs. random (%)", fontsize=9)
            ax.set_ylabel("Full accuracy (%)", fontsize=9)

            pts = data[filt][ds]  # offset -> (acc, cost)
            if not pts:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                continue

            offsets_sorted = sorted(pts.keys())
            accs   = [pts[o][0] for o in offsets_sorted]
            costs  = [pts[o][1] for o in offsets_sorted]

            sc = ax.scatter(
                costs, accs,
                c=offsets_sorted,
                cmap="plasma",
                s=80,
                marker=DS_MARKERS[ds],
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
            )
            # connect points with a light line in offset order
            ax.plot(costs, accs, color="grey", linewidth=0.8, alpha=0.5, zorder=2)

            # label each point with its offset
            for o, c, a in zip(offsets_sorted, costs, accs):
                ax.annotate(str(o), (c, a), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color="#333333")

            ax.grid(True, linestyle="--", alpha=0.4)

        # shared colorbar per row
        cbar = fig.colorbar(sc, ax=axes[row_i, :], shrink=0.7, pad=0.02)
        cbar.set_label("Offset", fontsize=8)
        cbar.set_ticks(OFFSETS)

    slug = label.replace(" ", "_").replace("-", "")
    out_path = plot_dir / f"effrouter_offset_sweep_{slug}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_combined_figure(all_results: dict, plot_dir: Path):
    """4×2 combined figure: rows=model_set×filter, cols=dataset."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 14), constrained_layout=True)
    fig.suptitle("EffiRoute offset sweep — all conditions", fontsize=13, fontweight="bold")

    row_configs = [
        ("8-model", "all"),
        ("8-model", "rc"),
        ("4-model", "all"),
        ("4-model", "rc"),
    ]
    row_titles = [
        "8-model · All questions",
        "8-model · R+C questions",
        "4-model · All questions",
        "4-model · R+C questions",
    ]

    for row_i, ((label, filt), row_title) in enumerate(zip(row_configs, row_titles)):
        for col_i, ds in enumerate(DATASETS):
            ax = axes[row_i][col_i]
            ax.set_title(f"{row_title}  ·  {DS_LABELS[ds]}", fontsize=9)
            ax.set_xlabel("Cost savings vs. random (%)", fontsize=8)
            ax.set_ylabel("Full accuracy (%)", fontsize=8)

            pts = all_results[label][filt][ds]
            if not pts:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                continue

            offsets_sorted = sorted(pts.keys())
            accs  = [pts[o][0] for o in offsets_sorted]
            costs = [pts[o][1] for o in offsets_sorted]

            sc = ax.scatter(costs, accs, c=offsets_sorted, cmap="plasma",
                            s=70, edgecolors="k", linewidths=0.5, zorder=3)
            ax.plot(costs, accs, color="grey", linewidth=0.8, alpha=0.5, zorder=2)
            for o, c, a in zip(offsets_sorted, costs, accs):
                ax.annotate(str(o), (c, a), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color="#333333")
            ax.grid(True, linestyle="--", alpha=0.4)

        cbar = fig.colorbar(sc, ax=axes[row_i, :], shrink=0.7, pad=0.02)
        cbar.set_label("Offset", fontsize=8)
        cbar.set_ticks(OFFSETS)

    out_path = plot_dir / "effrouter_offset_sweep_combined.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Collecting eval results ===")
    all_results = collect_results()

    print("\n=== Generating plots ===")
    for label in MODEL_SETS:
        make_figure(label, all_results[label], PLOT_DIR)

    make_combined_figure(all_results, PLOT_DIR)

    # also dump collected numbers to json for reference
    out_json = PLOT_DIR / "offset_sweep_data.json"
    serializable = {}
    for label, filters in all_results.items():
        serializable[label] = {}
        for filt, datasets in filters.items():
            serializable[label][filt] = {}
            for ds, offsets in datasets.items():
                serializable[label][filt][ds] = {
                    str(o): {"accuracy_pct": v[0], "cost_savings_pct": v[1]}
                    for o, v in offsets.items()
                }
    out_json.write_text(json.dumps(serializable, indent=2))
    print(f"Saved data: {out_json}")
    print("\nDone.")
