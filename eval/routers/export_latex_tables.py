#!/usr/bin/env python3
"""
Generate LaTeX tables for the router comparison paper section.

Produces 4 tables:
  no_qwen3 × all questions
  no_qwen3 × R+C questions
  small4   × all questions
  small4   × R+C questions

Each table has three dataset sub-groups (MoreHopQA / MuSiQue / StepCoT)
and three metrics: Full-task | Subtask (micro) | Chained subtask.

Usage:
    python -m eval.routers.export_latex_tables
    python -m eval.routers.export_latex_tables --out results_tables.tex
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ── Router display names ──────────────────────────────────────────────────────
ROUTER_LABELS = {
    "cosine":    r"\textsc{SimRoute}",
    "efficiency":"Efficiency Router",
    "embedllm":  "EmbedLLM",
    "experiment":r"\textsc{ExpertRoute}",
    "modelsat":  "ModelSAT",
}

# ── Data ─────────────────────────────────────────────────────────────────────
# (full, subtask_micro, chained)  — None = not available
DATA = {
    # ── ALL QUESTIONS ─────────────────────────────────────────────────────────
    ("no_qwen3", "all", "morehopqa"): {
        "cosine":    (0.4919, 0.7823, 0.3980),
        "efficiency":(0.4240, 0.7706, 0.3202),
        "embedllm":  (0.3953, 0.7553, 0.3274),
        "experiment":(0.4973, 0.8106, 0.4517),
        "modelsat":  (0.4848, 0.7865, 0.4061),
        "oracle_full": 0.7755, "oracle_sub": 0.8979, "oracle_chain": 0.7352,
        "best_single": 0.5474,
    },
    ("no_qwen3", "all", "musique"): {
        "cosine":    (0.4610, 0.5527, 0.1769),
        "efficiency":(0.4071, 0.4990, 0.1620),
        "embedllm":  (0.4291, 0.5060, 0.1672),
        "experiment":(0.5059, 0.5492, 0.2202),
        "modelsat":  (0.4690, 0.5095, 0.1713),
        "oracle_full": 0.7517, "oracle_sub": 0.7447, "oracle_chain": 0.5432,
        "best_single": 0.5388,
    },
    ("no_qwen3", "all", "stepcot"): {
        "cosine":    (None,   None,   None  ),
        "efficiency":(0.6214, 0.2790, 0.0000),
        "embedllm":  (0.7900, 0.3731, 0.0067),
        "experiment":(0.7940, 0.3582, 0.0134),
        "modelsat":  (0.7739, 0.3582, 0.0067),
        "oracle_full": 0.9095, "oracle_sub": 0.4283, "oracle_chain": 0.0536,
        "best_single": 0.7940,
    },
    ("small4", "all", "morehopqa"): {
        "cosine":    (0.2818, 0.6563, 0.1852),
        "efficiency":(0.2916, 0.6837, 0.2236),
        "embedllm":  (0.2943, 0.7169, 0.2773),
        "experiment":(0.3086, 0.7122, 0.2818),
        "modelsat":  (0.2021, 0.6548, 0.1968),
        "oracle_full": 0.5170, "oracle_sub": 0.8231, "oracle_chain": 0.5564,
        "best_single": 0.3086,
    },
    ("small4", "all", "musique"): {
        "cosine":    (0.2582, 0.3589, 0.0684),
        "efficiency":(0.2822, 0.4226, 0.1212),
        "embedllm":  (0.3575, 0.4150, 0.0962),
        "experiment":(0.3700, 0.4081, 0.0890),
        "modelsat":  (0.2456, 0.4460, 0.1338),
        "oracle_full": 0.5397, "oracle_sub": 0.6219, "oracle_chain": 0.3326,
        "best_single": 0.3722,
    },
    ("small4", "all", "stepcot"): {
        "cosine":    (None,   None,   None  ),
        "efficiency":(0.7152, 0.2927, 0.0000),
        "embedllm":  (0.7683, 0.3417, 0.0168),
        "experiment":(0.7554, 0.3381, 0.0151),
        "modelsat":  (0.7722, 0.3429, 0.0000),
        "oracle_full": 0.8727, "oracle_sub": 0.3912, "oracle_chain": 0.0285,
        "best_single": 0.7722,
    },
    # ── R+C QUESTIONS ─────────────────────────────────────────────────────────
    ("no_qwen3", "rc", "morehopqa"): {
        "cosine":    (0.4879, 0.7864, 0.3977),
        "efficiency":(0.4142, 0.7466, 0.3162),
        "embedllm":  (0.4316, 0.7642, 0.3249),
        "experiment":(0.4918, 0.8136, 0.4568),
        "modelsat":  (0.4782, 0.7893, 0.3919),
        "oracle_full": 0.7682, "oracle_sub": 0.8980, "oracle_chain": 0.7362,
        "best_single": 0.5441,
    },
    ("no_qwen3", "rc", "musique"): {
        "cosine":    (0.6000, 0.4127, 0.0000),
        "efficiency":(0.4000, 0.4444, 0.0500),
        "embedllm":  (0.5000, 0.4603, 0.0500),
        "experiment":(0.6500, 0.4603, 0.0000),
        "modelsat":  (0.6000, 0.5079, 0.1000),
        "oracle_full": 0.9500, "oracle_sub": 0.7619, "oracle_chain": 0.5000,
        "best_single": 0.7000,
        "note": "$n=20$",
    },
    ("small4", "rc", "morehopqa"): {
        "cosine":    (0.2706, 0.6553, 0.1814),
        "efficiency":(0.2512, 0.6736, 0.2270),
        "embedllm":  (0.2774, 0.7165, 0.2735),
        "experiment":(0.2900, 0.7105, 0.2745),
        "modelsat":  (0.1959, 0.6553, 0.2027),
        "oracle_full": 0.5044, "oracle_sub": 0.8223, "oracle_chain": 0.5490,
        "best_single": 0.2900,
    },
    ("small4", "rc", "musique"): {
        "cosine":    (0.2500, 0.3651, 0.1000),
        "efficiency":(0.2000, 0.3651, 0.0000),
        "embedllm":  (0.3500, 0.3333, 0.1000),
        "experiment":(0.4500, 0.3492, 0.1000),
        "modelsat":  (0.0500, 0.5079, 0.1500),
        "oracle_full": 0.6000, "oracle_sub": 0.6508, "oracle_chain": 0.4000,
        "best_single": 0.4500,
        "note": "$n=20$",
    },
}

ROUTERS   = ["cosine", "embedllm", "experiment", "modelsat"]
DATASETS  = ["morehopqa", "musique"]
DS_LABELS = {"morehopqa": "MoreHopQA", "musique": "MuSiQue", "stepcot": "StepCoT"}

# ── Helpers ──────────────────────────────────────────────────────────────────

def pct(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "---"
    s = f"{v*100:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def best_among(cell: dict, key: int) -> float:
    vals = [cell[r][key] for r in ROUTERS if cell[r][key] is not None]
    return max(vals) if vals else float("inf")


MODEL_SETS  = [("small4", "4-model"), ("no_qwen3", "8-model")]
SPLITS      = [("all", "All questions"), ("rc", "R+C questions")]
SPLIT_DS    = {"all": DATASETS, "rc": ["morehopqa", "musique"]}


def make_combined_table(caption: str, label: str) -> str:
    """One table* with all conditions: splits and model-sets as row sections,
    datasets as column groups. Best single row omitted for compactness."""
    datasets = DATASETS  # both datasets in columns for all splits
    n_ds  = len(datasets)
    n_cols = 1 + 3 * n_ds
    col_spec = "l" + "".join([" rrr"] * n_ds)

    L: list[str] = []
    L.append(r"\begin{table}[!t]")
    L.append(r"  \centering")
    L.append(r"  \renewcommand{\arraystretch}{0.88}")
    L.append(f"  \\caption{{{caption}}}")
    L.append(f"  \\label{{{label}}}")
    L.append(r"  \resizebox{\linewidth}{!}{%")
    L.append(r"  \small")
    L.append(r"  \setlength{\tabcolsep}{4pt}")
    L.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    L.append(r"    \toprule")

    # Dataset column-group headers (use "all" split to get notes)
    ds_headers = [""]
    cmidrule_parts = []
    for i, ds in enumerate(datasets):
        note = next(
            (DATA[(ms, "rc", ds)].get("note", "") for ms, _ in MODEL_SETS if (ms, "rc", ds) in DATA),
            "",
        )
        ds_label = DS_LABELS[ds]
        col_start = 2 + 3 * i
        col_end   = col_start + 2
        ds_headers.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{ds_label}}}}}")
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
    L.append("    " + " & ".join(ds_headers) + " \\\\")
    L.append("    " + " ".join(cmidrule_parts))

    metric_hdr = [""]
    for _ in datasets:
        metric_hdr += [r"\textbf{Full}", r"\textbf{Sub}", r"\textbf{Chained}"]
    L.append("    " + " & ".join(metric_hdr) + " \\\\")

    first_split = True
    for split, split_label in SPLITS:
        L.append(r"    \midrule")
        # Split section header spanning all columns
        L.append(f"    \\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{split_label}}}}} \\\\")
        L.append(f"    \\cmidrule(l){{1-{n_cols}}}")

        first_ms = True
        for model_set, ms_label in MODEL_SETS:
            if not first_ms:
                L.append(f"    \\cmidrule(l){{1-{n_cols}}}")
            first_ms = False

            # Model-set sub-label
            L.append(f"    \\multicolumn{{{n_cols}}}{{l}}{{\\hspace{{0.5em}}\\textit{{{ms_label}}}}} \\\\")

            bests: dict[str, tuple] = {}
            for ds in datasets:
                key = (model_set, split, ds)
                if key in DATA:
                    bests[ds] = (best_among(cell := DATA[key], 0),
                                 best_among(cell, 1),
                                 best_among(cell, 2))

            for r in ROUTERS:
                row = [r"  \hspace{1em}" + ROUTER_LABELS[r]]
                for ds in datasets:
                    key = (model_set, split, ds)
                    if key not in DATA:
                        row += ["---", "---", "---"]
                        continue
                    cell = DATA[key]
                    triple = cell[r]
                    b = bests.get(ds, (None, None, None))
                    row.append(pct(triple[0], triple[0] is not None and triple[0] == b[0]))
                    row.append(pct(triple[1], triple[1] is not None and triple[1] == b[1]))
                    row.append(pct(triple[2], triple[2] is not None and triple[2] == b[2]))
                L.append("    " + " & ".join(row) + " \\\\")

            # Oracle row
            oracle_row = [r"  \hspace{1em}\textit{Oracle}"]
            for ds in datasets:
                key = (model_set, split, ds)
                if key not in DATA:
                    oracle_row += ["---", "---", "---"]
                    continue
                cell = DATA[key]
                oracle_row += [
                    f"\\textit{{{pct(cell['oracle_full'])}}}",
                    f"\\textit{{{pct(cell['oracle_sub'])}}}",
                    f"\\textit{{{pct(cell['oracle_chain'])}}}"
                ]
            L.append("    " + " & ".join(oracle_row) + " \\\\")

    L.append(r"    \bottomrule")
    L.append(r"  \end{tabular}}%")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "outputs" / "router_comparison_tables.tex"))
    args = ap.parse_args()

    caption = (
        "Router comparison. "
        "Full = full-task accuracy; Sub = micro per-hop accuracy under full-task routing; "
        "Chained = all hops independently routed correctly. "
        "R+C = retrieval$+$computation questions only; "
        "MuSiQue R+C has $n{=}20$ questions (directional). "
        "Bold = best router per column."
    )

    preamble = (
        "% Auto-generated by eval/routers/export_latex_tables.py\n"
        "% Requires: \\usepackage{booktabs}\n\n"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(preamble + make_combined_table(caption, "tab:results") + "\n")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
