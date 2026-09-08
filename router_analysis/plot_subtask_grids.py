"""Plot per-question subtask correctness grids for the oracle router analysis.

For each dataset, picks a few representative questions and saves a grid image:
  - rows   = models
  - columns = subtasks (1-indexed)
  - filled cell = correct (1), empty cell = incorrect (0)

Output: router_analysis/<dataset>_<question_id_prefix>.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
MATRICES    = REPO_ROOT / "outputs" / "matrices"
OUT_DIR     = REPO_ROOT / "router_analysis"
OUT_DIR.mkdir(exist_ok=True)

DATASETS    = ["morehopqa", "musique", "stepcot"]
N_QUESTIONS = 5   # how many questions to plot per dataset

# Short display names for models (keep labels readable)
MODEL_SHORT = {
    "deepseek-r1-distill-llama-8b": "DeepSeek-R1\n8B",
    "llama-3.1-8b-instruct":        "LLaMA-3.1\n8B",
    "medgemma-4b-it":               "MedGemma\n4B",
    "mistral-7b-instruct-v0.3":     "Mistral-7B\nv0.3",
    "phi-4-mini-instruct":          "Phi-4-mini\n3.8B",
    "qwen1.5-0.5b-chat":            "Qwen1.5\n0.5B",
    "qwen3-30b-a3b":                "Qwen3\n30B-A3B",
    "qwen3-4b-thinking-2507":       "Qwen3-4B\nThinking",
}

CORRECT_COLOR   = "#2ecc71"   # green fill for correct
INCORRECT_COLOR = "#f0f0f0"   # light grey for incorrect
MISSING_COLOR   = "#e8d5f5"   # soft purple for missing data
GRID_COLOR      = "#cccccc"


def plot_question(question_id: str, entry: dict, dataset: str, out_path: Path) -> None:
    """Render one question's subtask grid and save as PNG."""
    models    = entry["cols"]          # list of model ids
    subtasks  = entry["rows"]          # list of subtask indices (0-based)
    binary    = np.array(entry["binary"], dtype=float)  # shape: (n_subtasks, n_models)

    # Transpose so rows=models, cols=subtasks
    grid = binary.T   # shape: (n_models, n_subtasks)

    n_models   = len(models)
    n_subtasks = len(subtasks)

    cell_w, cell_h = 0.9, 0.6
    fig_w = max(4, n_subtasks * cell_w + 2.5)
    fig_h = max(2, n_models  * cell_h + 1.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_subtasks)
    ax.set_ylim(0, n_models)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for mi, model in enumerate(models):
        for si in range(n_subtasks):
            val = grid[mi, si]
            if np.isnan(val):
                color = MISSING_COLOR
                label = "–"
                text_color = "#aaaaaa"
            elif val:
                color = CORRECT_COLOR
                label = "1"
                text_color = "#2c3e50"
            else:
                color = INCORRECT_COLOR
                label = "0"
                text_color = "#aaaaaa"
            rect = mpatches.FancyBboxPatch(
                (si + 0.05, mi + 0.05),
                0.9, 0.9,
                boxstyle="round,pad=0.05",
                linewidth=0.8,
                edgecolor=GRID_COLOR,
                facecolor=color,
            )
            ax.add_patch(rect)
            ax.text(
                si + 0.5, mi + 0.5,
                label,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=text_color,
            )

    # Y-axis: model labels
    ax.set_yticks(np.arange(n_models) + 0.5)
    ax.set_yticklabels(
        [MODEL_SHORT.get(m, m) for m in models],
        fontsize=8,
    )

    # X-axis: subtask labels (1-indexed for display)
    ax.set_xticks(np.arange(n_subtasks) + 0.5)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_subtasks)], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    correct_patch   = mpatches.Patch(color=CORRECT_COLOR,   label="Correct")
    incorrect_patch = mpatches.Patch(facecolor=INCORRECT_COLOR, edgecolor=GRID_COLOR, label="Incorrect")
    missing_patch   = mpatches.Patch(color=MISSING_COLOR,   label="Missing")
    ax.legend(
        handles=[correct_patch, incorrect_patch, missing_patch],
        loc="lower right", fontsize=8, framealpha=0.8,
        bbox_to_anchor=(1.0, -0.12),
    )

    title = f"{dataset}  ·  {question_id[:8]}…"
    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    for dataset in DATASETS:
        matrix_path = MATRICES / f"{dataset}_subtasks.json"
        if not matrix_path.exists():
            print(f"[skip] {matrix_path} not found")
            continue

        data = json.loads(matrix_path.read_text())
        question_ids = list(data.keys())

        # Pick questions with varying numbers of subtasks for visual variety;
        # fall back to first N_QUESTIONS if not enough variety
        by_n_subtasks: dict[int, list[str]] = {}
        for qid in question_ids:
            n = len(data[qid]["rows"])
            by_n_subtasks.setdefault(n, []).append(qid)

        selected: list[str] = []
        for qids in sorted(by_n_subtasks.values(), key=len):
            for qid in qids:
                if qid not in selected:
                    selected.append(qid)
                    break
            if len(selected) >= N_QUESTIONS:
                break
        # top-up with first unseen ids if needed
        for qid in question_ids:
            if len(selected) >= N_QUESTIONS:
                break
            if qid not in selected:
                selected.append(qid)

        print(f"\n{dataset}: plotting {len(selected)} questions")
        for qid in selected:
            out_path = OUT_DIR / f"{dataset}_{qid[:8]}.png"
            plot_question(qid, data[qid], dataset, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
