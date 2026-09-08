#!/usr/bin/env python3
"""
Derive EfficiencyRouter metrics from an existing predictions.json — no GPU needed.

Useful for computing metrics on a question-ID subset (e.g. R+C questions)
without re-running full inference.  Chained subtask evaluation requires GPU
and is NOT handled here (run eval.py with --qid-filter for that).

Usage:
  python eval_from_preds.py \
      --preds results/no_qwen3/predictions.json \
      --out-dir results/rc_questions/no_qwen3 \
      --qid-filter ../outputs/rc_qids.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
MATRICES_DIR = Path(PROJECT_DIR) / "outputs" / "updated-matrices"
DATASETS     = ["morehopqa", "musique", "stepcot"]


def load_matrix(dataset: str, variant: str) -> tuple[list, list, np.ndarray]:
    data = json.loads((MATRICES_DIR / f"{dataset}_{variant}.json").read_text())
    return data["rows"], data["cols"], np.array(
        [[np.nan if v is None else float(v) for v in row] for row in data["values"]],
        dtype=np.float64,
    )


def load_subtasks(dataset: str) -> dict:
    path = MATRICES_DIR / f"{dataset}_subtasks.json"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def compute_dataset_metrics(
    dataset: str,
    preds: dict[str, str],
    allowed_qids: set[str] | None,
) -> tuple[dict, dict | None]:
    qids_bin, cols, bin_vals = load_matrix(dataset, "full_binary")
    qids_eff, _,    eff_vals = load_matrix(dataset, "full_per_active_b")
    assert qids_bin == qids_eff, "Row mismatch between binary and efficiency matrices"

    row_idx = {q: i for i, q in enumerate(qids_bin)}
    col_idx = {m: j for j, m in enumerate(cols)}

    eval_qids = [q for q in qids_bin if q in preds]
    if allowed_qids is not None:
        eval_qids = [q for q in eval_qids if q in allowed_qids]
    if not eval_qids:
        print(f"  [{dataset}] 0 questions after filtering — skipping")
        return {}, None
    n = len(eval_qids)
    print(f"\n[{dataset}] {n} questions")

    row_mask = np.array([row_idx[q] for q in eval_qids], dtype=np.int64)
    bin_sub  = bin_vals[row_mask, :]
    eff_sub  = eff_vals[row_mask, :]

    # Derive params-per-model from the FULL efficiency matrix for numerical stability.
    K = len(cols)
    params_per_col = np.zeros(K)
    for j in range(K):
        col = eff_vals[:, j]
        pos = col[col > 0]
        if len(pos):
            params_per_col[j] = 1.0 / float(np.nanmean(pos))
    max_params = (params_per_col[params_per_col > 0].max()
                  if np.any(params_per_col > 0) else 1.0)
    params_per_col[params_per_col == 0] = max_params
    min_params = params_per_col.min()

    routed_j    = [col_idx.get(preds[q], 0) for q in eval_qids]
    valid_col   = [preds[q] in col_idx      for q in eval_qids]
    routed_bin  = np.array([float(np.nan_to_num(bin_sub[i, routed_j[i]])) if valid_col[i] else 0.0
                            for i in range(n)])
    routed_eff  = np.array([float(np.nan_to_num(eff_sub[i, routed_j[i]])) if valid_col[i] else 0.0
                            for i in range(n)])

    routing_acc        = float(routed_bin.mean())
    oracle_binary_mean = float(np.nanmean(np.nanmax(bin_sub, axis=1)))
    best_model_acc     = float(np.nanmax(np.nanmean(bin_sub, axis=0)))
    random_acc         = float(np.nanmean(bin_sub))

    eff_score_mean  = float(routed_eff.mean())
    oracle_eff_mean = float(np.nanmean(np.nanmax(eff_sub, axis=1)))

    nECS_mean   = float((routed_eff * min_params).mean())
    oracle_nECS = float(oracle_eff_mean * min_params)

    params_routed    = np.array([params_per_col[j] for j in routed_j])
    mean_params      = float(params_routed.mean())
    cost_savings_pct = float((max_params - mean_params) / max_params * 100)

    oracle_params_per_q = np.array([
        params_per_col[np.nanargmin(np.where(bin_sub[i] > 0, params_per_col, np.full(K, np.inf)))]
        if np.any(bin_sub[i] > 0) else max_params
        for i in range(n)
    ])
    oracle_mean_params = float(np.nanmean(oracle_params_per_q))
    oracle_savings_pct = float((max_params - oracle_mean_params) / max_params * 100)
    random_mean_params = float(params_per_col.mean())

    model_counts = {m: 0 for m in cols}
    for q in eval_qids:
        m = preds[q]
        if m in model_counts:
            model_counts[m] += 1
    model_pct = {m: round(model_counts[m] / n, 6) for m in cols}

    print(f"  routing={routing_acc:.4f}  oracle={oracle_binary_mean:.4f}  "
          f"nECS={nECS_mean:.4f}  mean_params={mean_params:.2f}B")

    full_result = {
        "dataset":              dataset,
        "n":                    n,
        "routing_accuracy":     routing_acc,
        "oracle_binary":        oracle_binary_mean,
        "best_single_model":    best_model_acc,
        "random_accuracy":      random_acc,
        "nECS":                 nECS_mean,
        "oracle_nECS":          oracle_nECS,
        "mean_active_params_B": mean_params,
        "oracle_mean_params_B": oracle_mean_params,
        "random_mean_params_B": random_mean_params,
        "cost_savings_pct":     cost_savings_pct,
        "oracle_savings_pct":   oracle_savings_pct,
        "eff_score":            eff_score_mean,
        "oracle_eff":           oracle_eff_mean,
        "routing_distribution":     model_counts,
        "routing_distribution_pct": model_pct,
    }

    # ------------------------------------------------------------------
    # Subtask metrics (overall: same model choice, evaluated at subtask level)
    # ------------------------------------------------------------------
    try:
        subtasks_data = load_subtasks(dataset)
    except FileNotFoundError as e:
        print(f"  [subtask] skipped: {e}")
        return full_result, None

    st_router   = st_oracle   = 0
    q_router    = q_oracle    = 0
    total_subs  = n_with_subs = 0
    per_model_sub: dict[str, int] = {m: 0 for m in cols}

    for qid in eval_qids:
        if qid not in subtasks_data:
            continue
        entry    = subtasks_data[qid]
        sub_cols = entry["cols"]
        binary   = entry["binary"]
        n_subs   = len(binary)
        if not n_subs:
            continue
        n_with_subs    += 1
        total_subs     += n_subs
        sub_col2idx     = {m: j for j, m in enumerate(sub_cols)}

        chosen   = preds[qid]
        cj = sub_col2idx.get(chosen)
        if cj is not None:
            cc = sum(binary[s][cj] for s in range(n_subs))
            st_router += cc
            q_router  += int(cc == n_subs)

        best = 0
        for m in cols:
            j = sub_col2idx.get(m)
            if j is None:
                continue
            mc = sum(binary[s][j] for s in range(n_subs))
            best = max(best, mc)
            per_model_sub[m] += mc
        st_oracle += best
        q_oracle  += int(best == n_subs)

    if not total_subs:
        return full_result, None

    pma   = {m: per_model_sub[m] / total_subs for m in cols}
    best_m = max(pma, key=pma.get)
    print(f"  [subtask] router={st_router/total_subs:.4f}  oracle={st_oracle/total_subs:.4f}  "
          f"q_router={q_router/n_with_subs:.4f}  q_oracle={q_oracle/n_with_subs:.4f}")

    subtask_result = {
        "dataset":                  dataset,
        "n_questions":              n_with_subs,
        "n_subtasks":               total_subs,
        "oracle_subtask_accuracy":  round(st_oracle / total_subs,  6),
        "subtask_router_accuracy":  round(st_router / total_subs,  6),
        "subtask_router_correct":   st_router,
        "oracle_question_accuracy": round(q_oracle  / n_with_subs, 6),
        "question_router_accuracy": round(q_router  / n_with_subs, 6),
        "question_router_correct":  q_router,
        "best_single_model":        best_m,
        "best_single_subtask_acc":  round(pma[best_m], 6),
        "avg_single_subtask_acc":   round(sum(pma.values()) / len(pma), 6),
        "per_model_subtask_accuracy": {m: round(v, 6) for m, v in pma.items()},
    }
    return full_result, subtask_result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds",     required=True,
                   help="Path to predictions.json  ({dataset: {qid: model}})")
    p.add_argument("--out-dir",   required=True,
                   help="Directory to write eval_results.json and subtask_results.json")
    p.add_argument("--datasets",  nargs="+", default=DATASETS)
    p.add_argument("--qid-filter", default=None,
                   help="Path to {dataset: [qid, ...]} JSON to restrict questions")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    qid_filter = None
    if args.qid_filter:
        with open(args.qid_filter) as f:
            qid_filter = json.load(f)

    with open(args.preds) as f:
        all_preds = json.load(f)

    full_results    = []
    subtask_results = []
    for ds in args.datasets:
        if ds not in all_preds:
            print(f"[{ds}] not in predictions — skipping")
            continue
        allowed = set(qid_filter[ds]) if (qid_filter and ds in qid_filter) else None
        full_r, sub_r = compute_dataset_metrics(ds, all_preds[ds], allowed)
        if full_r:
            full_results.append(full_r)
        if sub_r:
            subtask_results.append(sub_r)

    out = os.path.join(args.out_dir, "eval_results.json")
    with open(out, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved → {out}")

    if subtask_results:
        sub_out = os.path.join(args.out_dir, "subtask_results.json")
        with open(sub_out, "w") as f:
            json.dump(subtask_results, f, indent=2)
        print(f"Saved → {sub_out}")
    else:
        print("No subtask results (subtask matrices not found for evaluated datasets).")


if __name__ == "__main__":
    main()
