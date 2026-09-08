"""
Oracle router analysis for multi-model-required questions.

For each dataset this script:
  1. Identifies questions where the oracle chain is unbroken (every subtask
     has ≥1 correct model) but no single model answers all subtasks correctly.
  2. Saves the full question-ID lists to router_analysis/multi_model_required_<dataset>.json
  3. Plots per-question subtask grids into router_analysis/grids/<dataset>/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT   = Path(__file__).resolve().parent.parent
MATRICES    = REPO_ROOT / "outputs" / "updated_matrices"
OUT_DIR     = REPO_ROOT / "router_analysis"
GRIDS_DIR   = OUT_DIR / "grids"
DATASETS    = ["morehopqa", "musique", "stepcot"]

# Cap on grid plots for large datasets (musique has ~2k qualifying questions)
MAX_GRIDS   = {"morehopqa": 999, "musique": 50, "stepcot": 999}

CORRECT_COLOR  = "#2ecc71"
INCORRECT_COLOR = "#f0f0f0"
MISSING_COLOR  = "#e8d5f5"
GRID_COLOR     = "#cccccc"

MODEL_SHORT = {
    "deepseek-r1-distill-llama-8b": "DeepSeek-R1\n8B",
    "llama-3.1-8b-instruct":        "LLaMA-3.1\n8B",
    "llama-3.1-nemotron-nano-8b":   "Nemotron\nNano-8B",
    "mathstral-7b":                 "Mathstral\n7B",
    "medgemma-4b-it":               "MedGemma\n4B",
    "mistral-7b-instruct-v0.3":     "Mistral-7B\nv0.3",
    "phi-4-mini-instruct":          "Phi-4-mini\n3.8B",
    "qwen1.5-0.5b-chat":            "Qwen1.5\n0.5B",
    "qwen3-30b-a3b":                "Qwen3\n30B-A3B",
    "qwen3-4b-thinking-2507":       "Qwen3-4B\nThinking",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_multi_model_required(entry: dict) -> bool:
    binary = np.array(entry["binary"], dtype=float)
    if binary.size == 0:
        return False
    n_subtasks = len(entry["rows"])
    n_models   = len(entry["cols"])
    chain_ok = all(np.nansum(binary[si]) > 0 for si in range(n_subtasks))
    no_solo  = not any(np.all(binary[:, mi] == 1) for mi in range(n_models))
    return chain_ok and no_solo


def plot_grid(question_id: str, entry: dict, dataset: str, out_path: Path) -> None:
    models   = entry["cols"]
    subtasks = entry["rows"]
    binary   = np.array(entry["binary"], dtype=float)
    grid     = binary.T  # (n_models, n_subtasks)
    n_models, n_subtasks = grid.shape

    cell_w, cell_h = 0.9, 0.6
    fig_w = max(4, n_subtasks * cell_w + 2.5)
    fig_h = max(2, n_models  * cell_h + 1.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_subtasks)
    ax.set_ylim(0, n_models)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for mi in range(n_models):
        for si in range(n_subtasks):
            val = grid[mi, si]
            if np.isnan(val):
                color, label, tc = MISSING_COLOR, "–", "#aaaaaa"
            elif val:
                color, label, tc = CORRECT_COLOR, "1", "#2c3e50"
            else:
                color, label, tc = INCORRECT_COLOR, "0", "#aaaaaa"
            ax.add_patch(mpatches.FancyBboxPatch(
                (si + 0.05, mi + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.05", linewidth=0.8,
                edgecolor=GRID_COLOR, facecolor=color,
            ))
            ax.text(si + 0.5, mi + 0.5, label,
                    ha="center", va="center", fontsize=9, fontweight="bold", color=tc)

    ax.set_yticks(np.arange(n_models) + 0.5)
    ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=8)
    ax.set_xticks(np.arange(n_subtasks) + 0.5)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_subtasks)], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        mpatches.Patch(color=CORRECT_COLOR,  label="Correct"),
        mpatches.Patch(facecolor=INCORRECT_COLOR, edgecolor=GRID_COLOR, label="Incorrect"),
        mpatches.Patch(color=MISSING_COLOR,  label="Missing"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.8, bbox_to_anchor=(1.0, -0.12))

    fig.suptitle(f"{dataset}  ·  {question_id[:8]}…", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[str]] = {}

    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        matrix_path = MATRICES / f"{dataset}_subtasks.json"
        data = json.loads(matrix_path.read_text())

        qualifying = [qid for qid, e in data.items() if is_multi_model_required(e)]
        all_results[dataset] = qualifying
        print(f"  qualifying: {len(qualifying)}/{len(data)}")

        # --- 1. Save ID list ---
        list_path = OUT_DIR / f"multi_model_required_{dataset}.json"
        list_path.write_text(json.dumps(qualifying, indent=2))
        print(f"  list → {list_path.relative_to(REPO_ROOT)}")

        # --- 2. Plot grids ---
        ds_grid_dir = GRIDS_DIR / dataset
        ds_grid_dir.mkdir(exist_ok=True)
        to_plot = qualifying[:MAX_GRIDS[dataset]]
        print(f"  plotting {len(to_plot)} grids…")
        for qid in to_plot:
            plot_grid(qid, data[qid], dataset,
                      ds_grid_dir / f"{dataset}_{qid[:8]}.png")
        if len(qualifying) > MAX_GRIDS[dataset]:
            print(f"  (skipped {len(qualifying) - MAX_GRIDS[dataset]} grids — set MAX_GRIDS higher to plot all)")

    print(f"\n\nAll done. Results in {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
