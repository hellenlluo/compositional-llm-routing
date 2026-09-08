"""
Combined accuracy metrics across all questions, with breakdown by subtask-type
composition (using LLM-assigned labels) and sequence order.

Universe: all questions in each dataset that have both subtask and full-question
judgments AND whose subtasks have been labelled by classify_subtasks.py.

Subtask labels (written into the cleaned data files by classify_subtasks.py):
  RETRIEVAL  — fact lookup from world knowledge / provided context
  COMPUTATION — deterministic procedure applied to a known input
  REASONING  — logical inference or comparison

A question's TYPE-SET is the set of distinct labels across its subtask chain.
  Homogeneous  — all subtasks share one label (e.g. {R}, {C}, {Reasoning})
  Heterogeneous — subtasks span 2+ distinct labels

For each group the script prints:
  A. Full-task router accuracy — fraction of questions where ≥1 model answers
                                 the full (undecomposed) question correctly
  B. Subtask router accuracy   — defined by the chosen --oracle (see below)
  Gain = B − A

Subtask oracle variants (--oracle):

  strict          (default)
      B(q) = 1 iff every subtask hop has ≥1 model that answers it correctly in
             isolation; otherwise 0.
      • Apples-to-apples with A: both metrics are question-level binary scores
        ("did the strategy produce a correct answer for this question?").
      • Penalises compounding: with mean per-hop coverage p over chains of
        length L, the strict oracle requires p^L for B to match a question-
        level metric, so B < A is expected on long chains even when models
        are individually strong on each hop.

  soft
      B(q) = mean over hops of [hop covered by ≥1 model in isolation].
      • Continuous score in [0, 1] giving partial credit proportional to the
        fraction of the chain that subtask routing successfully handles.
      • Equal weighting across hop positions: an early-hop failure and a
        last-hop failure receive identical penalty, matching the fact that
        any uncovered hop can break the final answer.
      • Removes the compounding artefact but is no longer apples-to-apples
        with A: it reports "fraction of hops covered" rather than "fraction
        of questions correctly answered". Useful as a supplementary metric.

There is no single oracle that is simultaneously (i) question-level binary
like A, (ii) free of the compounding penalty, and (iii) defined independently
of A. Reporting BOTH `strict` (honest question-level comparison) and `soft`
(partial credit, no compounding) gives a complete picture; each clarifies a
limitation of the other.

MMR (Multi-Model Required) — no single model answers all subtasks simultaneously,
even though every subtask has at least one correct model.

Sequence analysis:
  Within each heterogeneous type-set group the script also prints the most
  common subtask-label sequences and their router metrics.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRICES  = REPO_ROOT / "outputs" / "updated_matrices"
OUT_DIR   = REPO_ROOT / "router_analysis"

CLEANED_PATHS = {
    "morehopqa": REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data" / "morehopqa_cleaned.json",
    "musique":   REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data" / "musique_cleaned.jsonl",
    "stepcot":   REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data" / "stepcot_cleaned.json",
}

SUPPORTS_LABELS = {"morehopqa", "musique"}
DATASETS = ["morehopqa", "musique", "stepcot"]

# Short display names for type-sets
_TYPESET_NAMES = {
    frozenset({"RETRIEVAL"}):                          "R only",
    frozenset({"COMPUTATION"}):                        "C only",
    frozenset({"REASONING"}):                          "Reasoning only",
    frozenset({"RETRIEVAL", "COMPUTATION"}):           "R + C",
    frozenset({"RETRIEVAL", "REASONING"}):             "R + Reasoning",
    frozenset({"COMPUTATION", "REASONING"}):           "C + Reasoning",
    frozenset({"RETRIEVAL", "COMPUTATION", "REASONING"}): "R + C + Reasoning",
}

def typeset_name(ts: frozenset) -> str:
    return _TYPESET_NAMES.get(ts, "+".join(sorted(ts)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_sequence(entry: dict) -> tuple[str, ...]:
    """Return the ordered tuple of subtask labels for a question."""
    return tuple(s.get("label", "UNLABELLED") for s in entry["question_decomposition"])


def label_typeset(entry: dict) -> frozenset[str]:
    return frozenset(label_sequence(entry))


def is_heterogeneous(entry: dict) -> bool:
    return len(label_typeset(entry)) > 1


def is_complex(entry: dict, min_hops: int = 3) -> bool:
    return len(entry["question_decomposition"]) >= min_hops


def load_cleaned(dataset: str) -> dict:
    path = CLEANED_PATHS[dataset]
    if path.suffix == ".jsonl":
        with path.open() as f:
            raw = [json.loads(line) for line in f]
    else:
        raw = json.loads(path.read_text())
    return {e["id"]: e for e in raw}


# ---------------------------------------------------------------------------
# Model filtering
# ---------------------------------------------------------------------------

def filter_full_matrix(full_d: dict, keep: set[str] | None) -> dict:
    """Return a copy of full_d with columns restricted to *keep* (None = all)."""
    if keep is None:
        return full_d
    idx = [i for i, c in enumerate(full_d["cols"]) if c in keep]
    return {
        **full_d,
        "cols":   [full_d["cols"][i] for i in idx],
        "values": [[row[i] for i in idx] for row in full_d["values"]],
    }


def filter_subtask_matrix(matrix: dict, keep: set[str] | None) -> dict:
    """Return a copy of matrix with model columns restricted to *keep* (None = all)."""
    if keep is None:
        return matrix
    result = {}
    for qid, entry in matrix.items():
        idx = [i for i, c in enumerate(entry["cols"]) if c in keep]
        result[qid] = {
            **entry,
            "cols":   [entry["cols"][i] for i in idx],
            "binary": [[row[i] for i in idx] for row in entry["binary"]],
        }
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

ORACLES = ("strict", "soft")


def question_score(qid: str, matrix: dict, full_d: dict, row_idx: dict,
                   oracle: str) -> tuple[int, float, int]:
    """Compute (A, B, C) for a single question under the chosen oracle.

    Returns
    -------
    a : 0 or 1
        Full-task oracle: 1 iff at least one model answers the full
        (undecomposed) question correctly.
    b : float in [0, 1]
        Subtask oracle, per the requested variant. Binary (0/1) for `strict`,
        continuous in [0, 1] for `soft`. See module docstring for definitions.
    c : 0 or 1
        Single-model subtask oracle: 1 iff a SINGLE model answers every hop
        correctly. Used only by callers that distinguish "any-model-per-hop"
        from "single-model-handles-everything"; not affected by `oracle`.
    """
    binary = np.array(matrix[qid]["binary"], dtype=float)
    n_sub, n_mod = binary.shape

    a = int(any(v == 1 for v in full_d["values"][row_idx[qid]]))

    # Per-hop strict coverage: hop k is covered iff some model answers it
    # correctly in isolation.
    hop_covered_strict = np.array(
        [int(np.any(binary[si] == 1)) for si in range(n_sub)],
        dtype=float,
    )

    if oracle == "strict":
        # Question-level binary: every hop must be covered.
        b = float(np.all(hop_covered_strict == 1))
    elif oracle == "soft":
        # Equal-weighted partial credit: mean fraction of hops covered.
        b = float(hop_covered_strict.mean()) if n_sub else 0.0
    else:
        raise ValueError(f"Unknown oracle '{oracle}'. Choose from {ORACLES}.")

    c = int(any(np.all(binary[:, mi] == 1) for mi in range(n_mod)))
    return a, b, c


def compute_metrics(group: list[str], matrix: dict, full_d: dict, row_idx: dict,
                    oracle: str = "strict") -> tuple:
    """Return (n, A, B, C) means for a list of question IDs.

    A and C are always binary fractions. B is a binary fraction under the
    `strict` oracle and a continuous mean in [0, 1] under the soft oracles.
    """
    if not group:
        return 0, 0.0, 0.0, 0.0
    A = C = 0
    B = 0.0
    for qid in group:
        a, b, c = question_score(qid, matrix, full_d, row_idx, oracle)
        A += a
        B += b
        C += c
    n = len(group)
    return n, A / n, B / n, C / n


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_row(label: str, group: list, matrix: dict, full_d: dict, row_idx: dict,
              indent: int = 0, oracle: str = "strict") -> None:
    pad = "  " * indent
    if not group:
        print(f"{pad}{label:<38} {'0':>6}   {'—':>21}   {'—':>20}   {'—':>12}")
        return
    n, A, B, _ = compute_metrics(group, matrix, full_d, row_idx, oracle=oracle)
    print(f"{pad}{label:<38} {n:>6}   {A:>21.1%}   {B:>20.1%}   {B-A:>+11.1%}")


def print_sequence_breakdown(
    group: list[str], cleaned: dict, matrix: dict, full_d: dict,
    row_idx: dict, top_n: int = 6, oracle: str = "strict",
) -> None:
    """Print the most common subtask-label sequences within a group."""
    seq_map: dict[tuple, list[str]] = {}
    for qid in group:
        seq = label_sequence(cleaned[qid])
        seq_map.setdefault(seq, []).append(qid)
    counts = Counter({seq: len(ids) for seq, ids in seq_map.items()})
    print(f"      {'Sequence':<42} {'N':>6}   {'Full-task (A)':>21}   {'Subtask (B)':>20}   {'Gain':>11}")
    print("      " + "-" * 106)
    for seq, _ in counts.most_common(top_n):
        ids   = seq_map[seq]
        _abbrev = {"RETRIEVAL": "Ret", "COMPUTATION": "Cmp", "REASONING": "Rsn"}
        label = "→".join(_abbrev.get(s, s[:3]) for s in seq)  # e.g. Ret→Ret→Cmp
        n, A, B, _ = compute_metrics(ids, matrix, full_d, row_idx, oracle=oracle)
        print(f"      {label:<42} {n:>6}   {A:>21.1%}   {B:>20.1%}   {B-A:>+11.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Heterogeneous accuracy analysis.")
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL_ID", default=None,
        help="Restrict analysis to these model IDs only (default: all models).",
    )
    parser.add_argument(
        "--oracle", choices=ORACLES, default="strict",
        help=(
            "Subtask oracle B definition (default: strict). "
            "'strict' = all hops covered, question-level binary (apples-to-"
            "apples with A but penalises compounding on long chains). "
            "'soft' = mean fraction of hops covered (equal-weight partial "
            "credit; removes compounding penalty but reports hop-level rather "
            "than question-level coverage)."
        ),
    )
    args = parser.parse_args()
    keep_models: set[str] | None = set(args.models) if args.models else None
    oracle = args.oracle

    if keep_models:
        print(f"Model filter: {sorted(keep_models)}")
    print(f"Subtask oracle: {oracle}\n")

    hdr = f"  {'Group':<38} {'N':>6}   {'Full-task router (A)':>22}   {'Subtask router (B)':>20}   {'Gain (B−A)':>12}"

    summary: dict[str, list] = {
        "all": [], "complex": [],
        "full_fail": [], "het_full_fail": [], "hom_full_fail": [],
        "hom_R": [], "hom_C": [], "hom_Reasoning": [],
        "het_all": [], "het_mmr": [], "het_mmr_uns": [], "het_non_mmr": [],
        "het_RC": [], "het_RC_mmr": [], "het_RC_mmr_uns": [],
        "het_RReasoning": [], "het_RReasoning_mmr": [],
        "het_all3": [], "het_all3_mmr": [],
    }
    summary_per_ds: dict[str, dict[str, list]] = {}
    # Per-question caches keyed by qid. _full_correct_cache stores the binary
    # full-task oracle A, and _oracle_sub_cache stores the chosen-oracle B
    # score (binary for `strict`, continuous in [0, 1] for soft variants).
    _full_correct_cache: dict[str, bool] = {}
    _oracle_sub_cache:   dict[str, float] = {}

    for dataset in DATASETS:
        cleaned = load_cleaned(dataset)
        matrix  = filter_subtask_matrix(
            json.loads((MATRICES / f"{dataset}_subtasks.json").read_text()), keep_models)
        full_d  = filter_full_matrix(
            json.loads((MATRICES / f"{dataset}_full_binary.json").read_text()), keep_models)
        row_idx = {qid: i for i, qid in enumerate(full_d["rows"])}

        # Compute MMR IDs from the matrices
        mmr_ids: set[str] = set()
        for qid, qdata in matrix.items():
            binary = qdata["binary"]
            if not binary: continue
            n_sub, n_mod = len(binary), len(binary[0])
            if (all(any(binary[si][mi] == 1 for mi in range(n_mod)) for si in range(n_sub))
                    and not any(all(binary[si][mi] == 1 for si in range(n_sub)) for mi in range(n_mod))):
                mmr_ids.add(qid)

        all_qs = [q for q in row_idx if q in matrix and q in cleaned]

        def full_correct(qid: str) -> bool:
            return any(v == 1 for v in full_d["values"][row_idx[qid]])

        def oracle_sub_score(qid: str) -> float:
            """B-score for `qid` under the active --oracle (binary or [0,1])."""
            binary = matrix.get(qid, {}).get("binary", [])
            if not binary:
                return 0.0
            _, b, _ = question_score(qid, matrix, full_d, row_idx, oracle)
            return b

        # B-description for the dataset header — varies with the chosen oracle.
        b_desc = {
            "strict": "fraction of questions where every hop has ≥1 correct model",
            "soft":   "mean fraction of hops covered (equal-weight partial credit)",
        }[oracle]
        print(f"\n{'='*110}")
        print(f"Dataset: {dataset}  (total evaluated: {len(all_qs)})  oracle={oracle}")
        print(f"{'='*110}")
        print(f"  Full-task router  = oracle accuracy if one model is chosen per question (metric A)")
        print(f"  Subtask router    = {b_desc} (metric B)")
        print(f"  Gain              = B − A")
        print()
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        if dataset not in SUPPORTS_LABELS:
            mmr_all = [q for q in all_qs if q in mmr_ids]
            for label, grp in [
                ("All questions",      all_qs),
                ("Full-task failures", [q for q in all_qs if not full_correct(q)]),
                ("MMR",                mmr_all),
                ("Non-MMR",            [q for q in all_qs if q not in mmr_ids]),
            ]:
                print_row(label, grp, matrix, full_d, row_idx, oracle=oracle)
            print(f"  (Subtask label taxonomy not applicable: answers are option letters)")
            continue

        for qid in all_qs:
            _full_correct_cache[qid] = full_correct(qid)
            _oracle_sub_cache[qid]   = oracle_sub_score(qid)

        complex_qs = [q for q in all_qs if is_complex(cleaned[q])]

        # Partition by type-set
        def by_typeset(pool: list[str], ts: frozenset) -> list[str]:
            return [q for q in pool if label_typeset(cleaned[q]) == ts]

        hom_R         = by_typeset(complex_qs, frozenset({"RETRIEVAL"}))
        hom_C         = by_typeset(complex_qs, frozenset({"COMPUTATION"}))
        hom_Reasoning = by_typeset(complex_qs, frozenset({"REASONING"}))
        het_RC        = by_typeset(complex_qs, frozenset({"RETRIEVAL", "COMPUTATION"}))
        het_RReasoning= by_typeset(complex_qs, frozenset({"RETRIEVAL", "REASONING"}))
        het_CReasoning= by_typeset(complex_qs, frozenset({"COMPUTATION", "REASONING"}))
        het_all3      = by_typeset(complex_qs, frozenset({"RETRIEVAL", "COMPUTATION", "REASONING"}))
        het_all       = [q for q in complex_qs if is_heterogeneous(cleaned[q])]
        hom_all       = [q for q in complex_qs if not is_heterogeneous(cleaned[q])]

        def mmr_sub(pool): return [q for q in pool if q in mmr_ids]
        def uns_sub(pool): return [q for q in pool if not full_correct(q)]
        def mmr_uns(pool): return [q for q in mmr_sub(pool) if not full_correct(q)]

        full_fail     = [q for q in all_qs if not full_correct(q)]
        het_full_fail = [q for q in het_all if not full_correct(q)]
        hom_full_fail = [q for q in hom_all if not full_correct(q)]

        # Sequence distribution within each main heterogeneous group
        typeset_dist = Counter(typeset_name(label_typeset(cleaned[q])) for q in complex_qs)

        ds_slice = {
            "all": all_qs, "complex": complex_qs,
            "full_fail": full_fail, "het_full_fail": het_full_fail,
            "hom_full_fail": hom_full_fail,
            "hom_R": hom_R, "hom_C": hom_C, "hom_Reasoning": hom_Reasoning,
            "het_all": het_all, "het_mmr": mmr_sub(het_all),
            "het_mmr_uns": mmr_uns(het_all), "het_non_mmr": [q for q in het_all if q not in mmr_ids],
            "het_RC": het_RC, "het_RC_mmr": mmr_sub(het_RC),
            "het_RC_mmr_uns": mmr_uns(het_RC),
            "het_RReasoning": het_RReasoning, "het_RReasoning_mmr": mmr_sub(het_RReasoning),
            "het_all3": het_all3, "het_all3_mmr": mmr_sub(het_all3),
        }
        summary_per_ds[dataset] = ds_slice
        for k, v in ds_slice.items():
            summary[k] += v

        # ── Typeset composition summary ────────────────────────────────────
        print(f"  Subtask type-set composition (complex ≥3 hops, {len(complex_qs)} questions):")
        for name, cnt in sorted(typeset_dist.items(), key=lambda x: -x[1]):
            print(f"    {name}: {cnt}")
        print()

        # ── Main metrics table ─────────────────────────────────────────────
        rows: list[tuple[str, list, int]] = [   # (label, group, indent)
            ("All questions",                   all_qs,              0),
            ("Complex ≥3 hops",                 complex_qs,          0),
            ("Full-task failures",              full_fail,           0),
            ("  Heterogeneous",                 het_full_fail,       0),
            ("  Homogeneous",                   hom_full_fail,       0),
            # ── Homogeneous groups ────────────────────────────
            ("Homogeneous",                     hom_all,             0),
            ("  Retrieval only",                hom_R,               0),
            ("    of which MMR",                mmr_sub(hom_R),      0),
            ("  Computation only",              hom_C,               0),
            ("  Reasoning only",                hom_Reasoning,       0),
            # ── Heterogeneous groups ──────────────────────────
            ("Heterogeneous",                   het_all,             0),
            ("  of which MMR",                  mmr_sub(het_all),    0),
            ("    MMR unsolvable (full-task)",   mmr_uns(het_all),    0),
            ("  non-MMR",                       [q for q in het_all if q not in mmr_ids], 0),
            ("  R + C",                         het_RC,              0),
            ("    of which MMR",                mmr_sub(het_RC),     0),
            ("      MMR unsolvable",            mmr_uns(het_RC),     0),
            ("  R + Reasoning",                 het_RReasoning,      0),
            ("    of which MMR",                mmr_sub(het_RReasoning), 0),
            ("      MMR unsolvable",            mmr_uns(het_RReasoning), 0),
            ("  C + Reasoning",                 het_CReasoning,      0),
            ("  R + C + Reasoning",             het_all3,            0),
            ("    of which MMR",                mmr_sub(het_all3),   0),
        ]

        for label, grp, _ in rows:
            print_row(label, grp, matrix, full_d, row_idx, oracle=oracle)

        # ── Sequence breakdown for non-trivial heterogeneous groups ────────
        for grp_name, grp in [("R + C", het_RC), ("R + Reasoning", het_RReasoning)]:
            if len(grp) < 5:
                continue
            print(f"\n  Sequence breakdown — {grp_name} (Ret=RETRIEVAL, Cmp=COMPUTATION, Rsn=REASONING):")
            print_sequence_breakdown(grp, cleaned, matrix, full_d, row_idx, oracle=oracle)

    # -----------------------------------------------------------------------
    # Cross-dataset summary (morehopqa + musique, micro-averaged)
    # -----------------------------------------------------------------------
    def summary_metrics(key):
        qids = summary[key]
        n = len(qids)
        if not n: return 0, 0.0, 0.0
        A = sum(1 for q in qids if _full_correct_cache[q]) / n
        # _oracle_sub_cache holds the per-question B score (binary 0/1 under
        # `strict`, continuous in [0, 1] under the soft variants). Summing
        # gives the correct mean for either case.
        B = sum(_oracle_sub_cache[q] for q in qids) / n
        return n, A, B

    print(f"\n{'='*110}")
    print(f"SUMMARY: morehopqa + musique  (micro-averaged, oracle={oracle})")
    print(f"{'='*110}")
    print()
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, key in [
        ("All questions",                   "all"),
        ("Complex ≥3 hops",                 "complex"),
        ("Full-task failures",              "full_fail"),
        ("  Heterogeneous",                 "het_full_fail"),
        ("  Homogeneous",                   "hom_full_fail"),
        ("Homogeneous",                     None),
        ("  Retrieval only",                "hom_R"),
        ("  Computation only",              "hom_C"),
        ("  Reasoning only",                "hom_Reasoning"),
        ("Heterogeneous",                   "het_all"),
        ("  of which MMR",                  "het_mmr"),
        ("    MMR unsolvable (full-task)",   "het_mmr_uns"),
        ("  non-MMR",                       "het_non_mmr"),
        ("  R + C",                         "het_RC"),
        ("    of which MMR",                "het_RC_mmr"),
        ("      MMR unsolvable",            "het_RC_mmr_uns"),
        ("  R + Reasoning",                 "het_RReasoning"),
        ("    of which MMR",                "het_RReasoning_mmr"),
        ("  R + C + Reasoning",             "het_all3"),
        ("    of which MMR",                "het_all3_mmr"),
    ]:
        if key is None:
            print(f"  {label}")
            continue
        n, A, B = summary_metrics(key)
        if not n:
            print(f"  {label:<38} {'—':>6}   {'—':>22}   {'—':>20}   {'—':>12}")
            continue
        print(f"  {label:<38} {n:>6}   {A:>21.1%}   {B:>20.1%}   {B-A:>+11.1%}")

    print()


if __name__ == "__main__":
    main()
