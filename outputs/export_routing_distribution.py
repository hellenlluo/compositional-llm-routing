#!/usr/bin/env python3
"""
Export routing distribution bar charts as PNG files.
Generates one PNG per (model_set × dataset) combination.
Output: outputs/routing_distribution_plots/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent / "routing_distribution_plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
ROUTERS = ["SimRoute", "EmbedLLM", "ExpertRoute", "ModelSAT", "EffiRoute"]

M8 = ["DeepSeek-R1", "Llama-3.1-8B", "Nemotron-Nano", "Mathstral",
      "MedGemma-4B", "Mistral-7B", "Phi-4-Mini", "Qwen1.5-0.5B"]
M4 = ["Nemotron-Nano", "Mistral-7B", "Phi-4-Mini", "Qwen1.5-0.5B"]

# Rows = models, Cols = routers [SimRoute, EmbedLLM, ExpertRoute, ModelSAT, EffiRoute]
DATA = {
    ("8-model", "MoreHopQA"): {
        "models": M8,
        "dist": [
            [16.5, 21.6, 22.0, 50.9, 16.8],   # DeepSeek-R1
            [ 7.6, 38.6, 62.8, 45.5, 42.2],   # Llama-3.1-8B
            [17.4,  0.3,  1.8,  0.0,  6.2],   # Nemotron-Nano
            [24.1,  1.0,  0.1,  0.0,  1.8],   # Mathstral
            [11.2, 23.2,  4.1,  0.0,  4.6],   # MedGemma-4B
            [ 4.5,  5.1,  0.4,  0.0,  3.8],   # Mistral-7B
            [ 8.9, 10.2,  8.8,  3.6, 23.1],   # Phi-4-Mini
            [ 9.8,  0.0,  0.0,  0.0,  1.6],   # Qwen1.5-0.5B
        ],
        "accuracy": [49.2, 39.5, 49.7, 48.5, 42.4],
    },
    ("8-model", "MuSiQue"): {
        "models": M8,
        "dist": [
            [28.8, 16.6, 42.5, 30.5, 37.5],
            [10.4, 16.2, 24.6, 51.5, 13.3],
            [11.6,  9.8,  5.1,  0.0,  7.7],
            [14.4, 12.9,  2.8,  0.0,  3.0],
            [13.8, 16.0,  7.7,  0.3,  5.8],
            [ 7.1, 14.2,  6.4,  0.0, 18.6],
            [ 9.4, 13.4,  8.8, 17.6,  9.9],
            [ 4.4,  0.9,  2.2,  0.0,  4.2],
        ],
        "accuracy": [46.1, 42.9, 50.6, 46.9, 40.7],
    },
    ("4-model", "MoreHopQA"): {
        "models": M4,
        "dist": [
            [20.1,  3.5, 10.5,  0.0, 15.9],   # Nemotron-Nano
            [29.5, 12.5,  4.8,  0.2, 16.6],   # Mistral-7B
            [33.9, 83.9, 84.7, 99.8, 60.5],   # Phi-4-Mini
            [16.5,  0.1,  0.0,  0.0,  7.0],   # Qwen1.5-0.5B
        ],
        "accuracy": [28.2, 29.4, 30.9, 20.2, 29.2],
    },
    ("4-model", "MuSiQue"): {
        "models": M4,
        "dist": [
            [37.5,  4.5, 12.9,  0.7, 23.9],
            [22.2, 10.3, 24.4,  1.9, 33.5],
            [25.0, 85.0, 62.1, 97.4, 33.0],
            [15.3,  0.2,  0.6,  0.0,  9.6],
        ],
        "accuracy": [25.8, 35.8, 37.0, 24.6, 28.2],
    },
}

# Color palette anchored to #B9D8D8 (sage teal)
PALETTE = [
    "#1F4646",  # DeepSeek-R1     very dark teal
    "#2E7070",  # Llama-3.1-8B    dark teal
    "#5AABAB",  # Nemotron-Nano   medium teal
    "#B9D8D8",  # Mathstral       anchor — light sage teal
    "#3A5A72",  # MedGemma-4B     dark slate blue
    "#7BA8C4",  # Mistral-7B      pale slate blue
    "#C49A6C",  # Phi-4-Mini      warm amber (complementary)
    "#A0A0A8",  # Qwen1.5-0.5B   neutral grey-blue
]


def make_chart(model_set: str, dataset: str, info: dict, ax: plt.Axes, show_legend: bool):
    models = info["models"]
    dist = np.array(info["dist"], dtype=float)   # shape: (n_models, n_routers)
    accuracy = info["accuracy"]
    n_routers = len(ROUTERS)

    # Normalize each router column to 100%
    col_sums = dist.sum(axis=0)
    col_sums[col_sums == 0] = 1
    dist_norm = dist / col_sums * 100

    y = np.arange(n_routers)
    bar_h = 0.55

    lefts = np.zeros(n_routers)
    bars = []
    for i, model in enumerate(models):
        vals = dist_norm[i]
        b = ax.barh(y, vals, left=lefts, height=bar_h,
                    color=PALETTE[i % len(PALETTE)], label=model)
        bars.append(b)
        # Label segments ≥ 8%
        for j, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 8:
                ax.text(l + v / 2, j, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        lefts += vals

    # Accuracy annotations to the right of each bar
    for j, acc in enumerate(accuracy):
        ax.text(101, j, f"{acc:.1f}%", va="center", fontsize=8,
                color="#333333", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(ROUTERS, fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Routing share (%)", fontsize=8)
    ax.set_title(f"{model_set}  ·  {dataset}", fontsize=10, fontweight="bold", pad=8)
    ax.axvline(100, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.text(101, n_routers - 0.05, "Acc.", fontsize=7, color="#888888", va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    if show_legend:
        handles = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=m)
                   for i, m in enumerate(models)]
        return handles
    return []


# ── Individual PNGs ───────────────────────────────────────────────────────────
for (model_set, dataset), info in DATA.items():
    n_models = len(info["models"])
    # Taller figure for 8-model charts to give legend room below
    fig_h = 4.4 if n_models > 4 else 3.5
    fig, ax = plt.subplots(figsize=(7, fig_h))
    bottom_margin = 0.28 if n_models > 4 else 0.18
    fig.subplots_adjust(left=0.15, right=0.88, top=0.90, bottom=bottom_margin)
    handles = make_chart(model_set, dataset, info, ax, show_legend=True)
    ncols = 4 if n_models > 4 else n_models
    fig.legend(handles=handles, loc="lower center", ncol=ncols,
               fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    slug = f"{model_set.replace('-','').replace(' ','_')}_{dataset.lower()}"
    path = OUT_DIR / f"routing_dist_{slug}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── Combined 2×2 PNG ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
fig.suptitle("Routing Distribution by Router", fontsize=13, fontweight="bold", y=1.01)

configs = [
    (("8-model", "MoreHopQA"), axes[0][0]),
    (("8-model", "MuSiQue"),   axes[0][1]),
    (("4-model", "MoreHopQA"), axes[1][0]),
    (("4-model", "MuSiQue"),   axes[1][1]),
]

all_handles = []
for (key, ax) in configs:
    show = (key == ("8-model", "MoreHopQA"))
    h = make_chart(key[0], key[1], DATA[key], ax, show_legend=show)
    if h:
        all_handles = h

# Build full legend from 8-model models (superset)
all_handles = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=m)
               for i, m in enumerate(M8)]
fig.legend(handles=all_handles, loc="lower center", ncol=4,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))

fig.tight_layout()
combined_path = OUT_DIR / "routing_dist_combined.png"
fig.savefig(combined_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {combined_path}")
print("\nAll done.")
