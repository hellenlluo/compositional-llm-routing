#!/usr/bin/env python3
"""
Dump oracle matrix analysis to JSON for canvas/LaTeX export.
Runs heterogeneous_accuracy logic for a given model set and outputs
structured results for morehopqa + musique + cross-dataset summary.

Usage:
    python -m router_analysis.dump_oracle_analysis --model-set no_qwen3
    python -m router_analysis.dump_oracle_analysis --model-set small4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRICES  = REPO_ROOT / "outputs" / "updated_matrices"

CLEANED_PATHS = {
    "morehopqa": REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data" / "morehopqa_cleaned.json",
    "musique":   REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data" / "musique_cleaned.jsonl",
}

MODEL_SETS = {
    "no_qwen3": [
        "deepseek-r1-distill-llama-8b", "llama-3.1-8b-instruct",
        "llama-3.1-nemotron-nano-8b", "mathstral-7b", "medgemma-4b-it",
        "mistral-7b-instruct-v0.3", "phi-4-mini-instruct", "qwen1.5-0.5b-chat",
    ],
    "small4": [
        "mistral-7b-instruct-v0.3", "qwen1.5-0.5b-chat",
        "phi-4-mini-instruct", "llama-3.1-nemotron-nano-8b",
    ],
}

_TYPESET_NAMES = {
    frozenset({"RETRIEVAL"}):                              "R only",
    frozenset({"COMPUTATION"}):                            "C only",
    frozenset({"REASONING"}):                              "Reasoning only",
    frozenset({"RETRIEVAL", "COMPUTATION"}):               "R + C",
    frozenset({"RETRIEVAL", "REASONING"}):                 "R + Reasoning",
    frozenset({"COMPUTATION", "REASONING"}):               "C + Reasoning",
    frozenset({"RETRIEVAL", "COMPUTATION", "REASONING"}):  "R + C + Reasoning",
}


def load_cleaned(dataset: str) -> dict:
    path = CLEANED_PATHS[dataset]
    if path.suffix == ".jsonl":
        with path.open() as f:
            raw = [json.loads(line) for line in f]
    else:
        raw = json.loads(path.read_text())
    return {e["id"]: e for e in raw}


def filter_cols(matrix_d: dict, keep: list[str]) -> dict:
    keep_set = set(keep)
    idx = [i for i, c in enumerate(matrix_d["cols"]) if c in keep_set]
    return {
        **matrix_d,
        "cols":   [matrix_d["cols"][i] for i in idx],
        "values": [[row[i] for i in idx] for row in matrix_d["values"]],
    }


def filter_subtask_cols(matrix: dict, keep: list[str]) -> dict:
    keep_set = set(keep)
    result = {}
    for qid, entry in matrix.items():
        idx = [i for i, c in enumerate(entry["cols"]) if c in keep_set]
        result[qid] = {
            **entry,
            "cols":   [entry["cols"][i] for i in idx],
            "binary": [[row[i] for i in idx] for row in entry["binary"]],
        }
    return result


def question_metrics(qid, matrix, full_d, row_idx):
    """Return (full_correct, strict_sub, soft_sub, single_model_sub)."""
    binary = np.array(matrix[qid]["binary"], dtype=float)
    n_sub, n_mod = binary.shape
    full_ok = int(any(v == 1 for v in full_d["values"][row_idx[qid]]))
    if n_sub == 0 or n_mod == 0:
        return full_ok, 0.0, 0.0, 0
    hop_covered = np.array([int(np.any(binary[s] == 1)) for s in range(n_sub)], dtype=float)
    strict = float(np.all(hop_covered == 1))
    soft   = float(hop_covered.mean())
    single = int(any(np.all(binary[:, m] == 1) for m in range(n_mod)))
    return full_ok, strict, soft, single


def compute_group(qids, matrix, full_d, row_idx):
    if not qids:
        return {"n": 0, "full": None, "strict": None, "soft": None, "single": None, "gain_strict": None}
    n = len(qids)
    full = strict = soft = single = 0.0
    for qid in qids:
        f, s_strict, s_soft, s_single = question_metrics(qid, matrix, full_d, row_idx)
        full   += f
        strict += s_strict
        soft   += s_soft
        single += s_single
    return {
        "n":           n,
        "full":        round(full   / n, 6),
        "strict":      round(strict / n, 6),
        "soft":        round(soft   / n, 6),
        "single":      round(single / n, 6),
        "gain_strict": round((strict - full) / n, 6),
    }


def label_typeset(entry):
    return frozenset(s.get("label", "UNLABELLED") for s in entry["question_decomposition"])


def is_heterogeneous(entry):
    return len(label_typeset(entry)) > 1


def is_complex(entry, min_hops=3):
    return len(entry["question_decomposition"]) >= min_hops


def by_typeset(pool, cleaned, ts):
    return [q for q in pool if label_typeset(cleaned[q]) == ts]


def analyse_dataset(dataset, models):
    cleaned = load_cleaned(dataset)
    full_d  = filter_cols(
        json.loads((MATRICES / f"{dataset}_full_binary.json").read_text()), models)
    matrix  = filter_subtask_cols(
        json.loads((MATRICES / f"{dataset}_subtasks.json").read_text()), models)
    row_idx = {qid: i for i, qid in enumerate(full_d["rows"])}

    mmr_ids = set()
    for qid, qdata in matrix.items():
        b = qdata["binary"]
        if not b: continue
        n_sub, n_mod = len(b), len(b[0])
        if (all(any(b[s][m] for m in range(n_mod)) for s in range(n_sub)) and
                not any(all(b[s][m] for s in range(n_sub)) for m in range(n_mod))):
            mmr_ids.add(qid)

    all_qs   = [q for q in row_idx if q in matrix and q in cleaned]
    full_ok  = {q: any(v == 1 for v in full_d["values"][row_idx[q]]) for q in all_qs}
    complex_qs = [q for q in all_qs if is_complex(cleaned[q])]

    TS = frozenset
    hom_R    = by_typeset(complex_qs, cleaned, TS({"RETRIEVAL"}))
    hom_C    = by_typeset(complex_qs, cleaned, TS({"COMPUTATION"}))
    hom_Rsn  = by_typeset(complex_qs, cleaned, TS({"REASONING"}))
    het_RC   = by_typeset(complex_qs, cleaned, TS({"RETRIEVAL", "COMPUTATION"}))
    het_RRsn = by_typeset(complex_qs, cleaned, TS({"RETRIEVAL", "REASONING"}))
    het_CRsn = by_typeset(complex_qs, cleaned, TS({"COMPUTATION", "REASONING"}))
    het_all3 = by_typeset(complex_qs, cleaned, TS({"RETRIEVAL", "COMPUTATION", "REASONING"}))
    het_all  = [q for q in complex_qs if is_heterogeneous(cleaned[q])]
    hom_all  = [q for q in complex_qs if not is_heterogeneous(cleaned[q])]

    # Het/hom split across ALL questions (not just complex)
    all_het = [q for q in all_qs if is_heterogeneous(cleaned[q])]
    all_hom = [q for q in all_qs if not is_heterogeneous(cleaned[q])]

    mmr  = lambda pool: [q for q in pool if q in mmr_ids]
    fail = lambda pool: [q for q in pool if not full_ok[q]]
    mmr_fail = lambda pool: [q for q in pool if q in mmr_ids and not full_ok[q]]

    groups = {
        "all":              all_qs,
        "complex":          complex_qs,
        "full_fail":        fail(all_qs),
        "all_het_fail":     fail(all_het),
        "all_hom_fail":     fail(all_hom),
        "complex_fail":     fail(complex_qs),
        "het_full_fail":    fail(het_all),
        "hom_full_fail":    fail(hom_all),
        "hom_all":          hom_all,
        "hom_R":            hom_R,
        "hom_R_mmr":        mmr(hom_R),
        "hom_C":            hom_C,
        "hom_Rsn":          hom_Rsn,
        "het_all":          het_all,
        "het_mmr":          mmr(het_all),
        "het_mmr_fail":     mmr_fail(het_all),
        "het_non_mmr":      [q for q in het_all if q not in mmr_ids],
        "het_RC":           het_RC,
        "het_RC_mmr":       mmr(het_RC),
        "het_RC_mmr_fail":  mmr_fail(het_RC),
        "het_RRsn":         het_RRsn,
        "het_RRsn_mmr":     mmr(het_RRsn),
        "het_CRsn":         het_CRsn,
        "het_all3":         het_all3,
        "het_all3_mmr":     mmr(het_all3),
    }

    return {
        k: compute_group(v, matrix, full_d, row_idx)
        for k, v in groups.items()
    }, {k: v for k, v in groups.items()}  # also return raw qid lists for cross-ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-set", required=True, choices=list(MODEL_SETS))
    args = ap.parse_args()

    models = MODEL_SETS[args.model_set]
    results = {}

    # Per-dataset cache for cross-dataset summary
    all_qids:  dict[str, list] = {k: [] for k in ["all","complex","full_fail",
        "all_het_fail","all_hom_fail","complex_fail",
        "het_full_fail","hom_full_fail","hom_all","hom_R","hom_R_mmr","hom_C",
        "hom_Rsn","het_all","het_mmr","het_mmr_fail","het_non_mmr",
        "het_RC","het_RC_mmr","het_RC_mmr_fail","het_RRsn","het_RRsn_mmr",
        "het_CRsn","het_all3","het_all3_mmr"]}

    # We need full/subtask matrices across both datasets for cross-ds summary
    cross_full_d: dict = {}  # qid → full_ok
    cross_matrix: dict = {}  # qid → binary data
    cross_full_d_ref: dict = {}
    cross_row_idx: dict = {}

    for ds in ["morehopqa", "musique"]:
        print(f"  Processing {ds}...", flush=True)
        ds_results, ds_qids = analyse_dataset(ds, models)
        results[ds] = ds_results
        for k, v in ds_qids.items():
            all_qids[k].extend(v)

    # Cross-dataset summary: re-run compute_group on pooled qids across both datasets
    # We need unified matrix+full_d for this — rebuild combined lookups
    print("  Building cross-dataset summary...", flush=True)
    combo_matrix: dict = {}
    combo_full_d_vals: dict = {}  # qid → list of correctness values
    combo_row_idx: dict = {}

    for ds in ["morehopqa", "musique"]:
        full_d = filter_cols(
            json.loads((MATRICES / f"{ds}_full_binary.json").read_text()), models)
        matrix = filter_subtask_cols(
            json.loads((MATRICES / f"{ds}_subtasks.json").read_text()), models)
        cleaned = load_cleaned(ds)
        for i, qid in enumerate(full_d["rows"]):
            if qid in matrix and qid in cleaned:
                combo_matrix[qid]    = matrix[qid]
                combo_full_d_vals[qid] = full_d["values"][i]
                combo_row_idx[qid]   = i

    # Build a fake full_d for compute_group compatibility
    combo_full_d = {
        "rows":   list(combo_row_idx.keys()),
        "cols":   models,
        "values": {qid: combo_full_d_vals[qid] for qid in combo_row_idx},
    }
    # Patch question_metrics to work with dict-based values
    def compute_group_cross(qids):
        if not qids:
            return {"n": 0, "full": None, "strict": None, "soft": None, "single": None, "gain_strict": None}
        n = len(qids)
        full = strict = soft = single = 0.0
        for qid in qids:
            binary = np.array(combo_matrix[qid]["binary"], dtype=float)
            n_sub, n_mod = binary.shape
            full_ok = int(any(v == 1 for v in combo_full_d["values"][qid]))
            if n_sub == 0 or n_mod == 0:
                full += full_ok
                continue
            hop_covered = np.array([int(np.any(binary[s] == 1)) for s in range(n_sub)], dtype=float)
            full   += full_ok
            strict += float(np.all(hop_covered == 1))
            soft   += float(hop_covered.mean())
            single += int(any(np.all(binary[:, m] == 1) for m in range(n_mod)))
        return {
            "n":           n,
            "full":        round(full   / n, 6),
            "strict":      round(strict / n, 6),
            "soft":        round(soft   / n, 6),
            "single":      round(single / n, 6),
            "gain_strict": round((strict - full) / n, 6),
        }

    results["cross"] = {k: compute_group_cross(v) for k, v in all_qids.items()}

    out = REPO_ROOT / "outputs" / f"oracle_analysis_{args.model_set}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
