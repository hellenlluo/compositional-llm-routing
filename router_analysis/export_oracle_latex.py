#!/usr/bin/env python3
"""
Generate LaTeX tables for oracle matrix analysis.
Reads outputs/oracle_analysis_{no_qwen3,small4}.json (produced by dump_oracle_analysis.py).

Usage:
    python -m router_analysis.export_oracle_latex
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Full row list — used for the appendix table.
ROWS_FULL = [
    # (key, display label, indent)
    # ── Overview ──────────────────────────────────────────────────────────────
    ("all",              "All questions",                          0),
    ("full_fail",        r"\hspace{1em}Full-task failures",              1),
    ("all_het_fail",     r"\hspace{2em}Heterogeneous failures",           2),
    ("all_hom_fail",     r"\hspace{2em}Homogeneous failures",             2),
    ("complex",          r"Complex ($\geq$3 hops)",                       0),
    ("complex_fail",     r"\hspace{1em}Full-task failures (complex)",     1),
    ("het_full_fail",    r"\hspace{2em}Heterogeneous failures",           2),
    ("hom_full_fail",    r"\hspace{2em}Homogeneous failures",             2),
    (None,               r"\midrule",                                    -1),
    # ── Homogeneous ───────────────────────────────────────────────────────────
    ("hom_all",          "Homogeneous",                                   0),
    ("hom_R",            r"\hspace{1em}R only",                           1),
    ("hom_Rsn",          r"\hspace{1em}Reasoning only",                   1),
    (None,               r"\midrule",                                    -1),
    # ── Heterogeneous ─────────────────────────────────────────────────────────
    ("het_all",          "Heterogeneous",                                 0),
    ("het_mmr",          r"\hspace{1em}MMR",                              1),
    ("het_mmr_fail",     r"\hspace{2em}MMR failures",                     2),
    ("het_RC",           r"\hspace{1em}R + C",                            1),
    ("het_RC_mmr",       r"\hspace{2em}MMR",                              2),
    ("het_RC_mmr_fail",  r"\hspace{3em}MMR failures",                     3),
    ("het_RRsn",         r"\hspace{1em}R + Reasoning",                    1),
    ("het_RRsn_mmr",     r"\hspace{2em}MMR",                              2),
    ("het_all3",         r"\hspace{1em}R + C + Reasoning",                1),
    ("het_all3_mmr",     r"\hspace{2em}MMR",                              2),
]

# Short row list — used for the main-paper table (no MMR breakdowns).
ROWS_SHORT = [
    ("all",           "All questions",                                  0),
    ("full_fail",     r"\hspace{1em}Full-task failures",               1),
    ("complex",       r"Complex ($\geq$3 hops)",                       0),
    ("complex_fail",  r"\hspace{1em}Full-task failures (complex)",     1),
    ("het_full_fail", r"\hspace{2em}Heterogeneous failures",           2),
    ("hom_full_fail", r"\hspace{2em}Homogeneous failures",             2),
    ("hom_all",       r"\hspace{1em}Homogeneous",                      1),
    ("het_all",       r"\hspace{1em}Heterogeneous",                    1),
    ("het_RC",        r"\hspace{2em}R + C",                            2),
    ("het_RRsn",      r"\hspace{2em}R + Reasoning",                    2),
]

ROWS = ROWS_FULL  # default (backward-compat)

SCOPES = [
    ("morehopqa", "MoreHopQA"),
    ("musique",   "MuSiQue"),
]


def pct(v):
    if v is None:
        return "---"
    return f"{v*100:.1f}"


def gain(d):
    if d["full"] is None or d["strict"] is None:
        return "---"
    g = d["strict"] - d["full"]
    return ("+" if g >= 0 else "") + f"{g*100:.1f}"


MODEL_SETS = [
    ("small4",   "4-model set"),
    ("no_qwen3", "8-model set"),
]


def make_section(data: dict, rows: list) -> list[str]:
    """Return table body lines for one model-set × all scopes."""
    lines = []
    for scope, scope_lbl in SCOPES:
        lines.append(r"    \midrule")
        lines.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{scope_lbl}}}}} \\\\")
        pending_midrule = False
        for key, label_str, indent in rows:
            if key is None:
                pending_midrule = True
                continue
            d = data[scope].get(key, {})
            n = d.get("n", 0)
            if not n:
                continue
            if pending_midrule:
                lines.append(r"    \midrule")
                pending_midrule = False
            cell = f"{n:,} & {pct(d.get('full'))} & {pct(d.get('strict'))} & {gain(d)}"
            bold = indent == 0
            lbl = f"\\textbf{{{label_str}}}" if bold else label_str
            lines.append(f"    {lbl} & {cell} \\\\")
    return lines


def make_single_table(ms: str, ms_label: str, raw_data: dict, caption: str, label: str, rows: list) -> str:
    data = dict(raw_data)
    data["rc"] = {"all": data["cross"]["het_RC"]}

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"  \centering")
    lines.append(r"  \renewcommand{\arraystretch}{0.88}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l rrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Group} & \textbf{N} & \textbf{Full} & \textbf{Chained} & \textbf{Gain} \\")
    lines += make_section(data, rows)
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}}%")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_combined_table(all_data: dict, caption: str, label: str, rows: list) -> str:
    """Generate one table per model set, returned as a single string."""
    tables = []
    for ms, ms_label in MODEL_SETS:
        ms_caption = caption.replace("Oracle accuracy analysis.", f"Oracle accuracy analysis ({ms_label}).")
        tables.append(make_single_table(ms, ms_label, all_data[ms], ms_caption, f"{label}_{ms}", rows))
    return "\n\n".join(tables)


def main():
    header = "% Auto-generated by router_analysis/export_oracle_latex.py\n% Requires: \\usepackage{booktabs}\n\n"

    all_data = {}
    for ms, _ in MODEL_SETS:
        path = REPO / "outputs" / f"oracle_analysis_{ms}.json"
        all_data[ms] = json.loads(path.read_text())

    # ── Full table (appendix) ──────────────────────────────────────────────────
    cap_full = (
        r"Oracle accuracy analysis. "
        r"Full = fraction of questions where $\geq 1$ model answers the full question correctly. "
        r"Chained = fraction of questions where every subtask hop has $\geq 1$ correct model. "
        r"Gain = Chained $-$ Full. "
        r"MMR = Multi-Model Required: every hop has $\geq 1$ correct model but no single model "
        r"covers all hops. Subtask labels (R/C/Reasoning) assigned by LLM classifier."
    )
    out_full = REPO / "outputs" / "oracle_analysis_tables.tex"
    out_full.write_text(header + make_combined_table(all_data, cap_full, "tab:oracle_full", ROWS_FULL) + "\n")
    print(f"Saved (full/appendix) → {out_full}")

    # ── Short table (main paper, no MMR) ──────────────────────────────────────
    cap_short_first = (
        r"Oracle accuracy upper bounds by question group (4-model set). "
        r"\textbf{Full}: $\geq 1$ model answers the full question correctly. "
        r"\textbf{Chained}: $\geq 1$ model correct on every subtask hop. "
        r"\textbf{Gain} = Chained $-$ Full. "
        r"Subtask labels (R/C/Reasoning) from LLM classifier."
    )
    cap_short_second = (
        r"Oracle accuracy upper bounds by question group (8-model set). "
        r"Metrics defined as in the preceding table."
    )
    short_tables = []
    for i, (ms, ms_label) in enumerate(MODEL_SETS):
        cap = cap_short_first if i == 0 else cap_short_second
        short_tables.append(make_single_table(ms, ms_label, all_data[ms], cap, f"tab:oracle_short_{ms}", ROWS_SHORT))
    out_short = REPO / "outputs" / "oracle_analysis_tables_short.tex"
    out_short.write_text(header + "\n\n".join(short_tables) + "\n")
    print(f"Saved (short/main paper) → {out_short}")


if __name__ == "__main__":
    main()
