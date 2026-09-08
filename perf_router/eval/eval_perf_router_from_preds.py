#!/usr/bin/env python3
"""
Derive perf-router metrics from an existing predictions.json — no GPU needed.

Useful for computing metrics on a question-ID subset (e.g. R+C questions)
without re-running full inference.  Chained subtask evaluation requires GPU
and is NOT handled here.

Usage:
  python eval_perf_router_from_preds.py \
      --preds ../../perf_router/results/no_qwen3/predictions.json \
      --out-dir ../../perf_router/results/rc_questions/no_qwen3 \
      --qid-filter ../../outputs/rc_qids.json
"""

import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR    = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(EVAL_DIR)

MATRICES_DIR = os.path.join(PROJECT_DIR, "outputs", "updated-matrices")
DATASETS     = ["morehopqa", "musique", "stepcot"]


def load_full_binary(dataset: str):
    path = os.path.join(MATRICES_DIR, f"{dataset}_full_binary.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        mat = json.load(f)
    return mat["rows"], mat["cols"], mat["values"]


def load_subtasks(dataset: str) -> dict:
    path = os.path.join(MATRICES_DIR, f"{dataset}_subtasks.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def compute_dataset_metrics(
    dataset: str,
    preds: dict[str, str],
    allowed_qids: set[str] | None,
) -> tuple[dict, dict | None]:
    qids, cols, values = load_full_binary(dataset)

    row_idx   = {q: i for i, q in enumerate(qids)}
    col_idx   = {m: j for j, m in enumerate(cols)}

    eval_qids = [q for q in qids if q in preds]
    if allowed_qids is not None:
        eval_qids = [q for q in eval_qids if q in allowed_qids]
    if not eval_qids:
        print(f"  [{dataset}] 0 questions after filtering — skipping")
        return {}, None
    n = len(eval_qids)
    print(f"\n[{dataset}] {n} questions")

    routing_correct = oracle_correct = 0
    per_model_correct = {m: 0 for m in cols}
    model_choice_counts = {m: 0 for m in cols}

    for qid in eval_qids:
        chosen    = preds[qid]
        row_vals  = values[row_idx[qid]]
        cj        = col_idx.get(chosen)

        if cj is not None:
            routing_correct += row_vals[cj]
            model_choice_counts[chosen] = model_choice_counts.get(chosen, 0) + 1

        oracle_correct += int(any(row_vals[col_idx[m]] for m in cols))

        for m in cols:
            per_model_correct[m] += row_vals[col_idx[m]]

    per_model_acc  = {m: per_model_correct[m] / n for m in cols}
    best_model     = max(per_model_acc, key=per_model_acc.get)
    avg_single_acc = sum(per_model_acc.values()) / len(per_model_acc)
    model_choice_pct = {m: round(model_choice_counts.get(m, 0) / n, 6) for m in cols}

    print(f"  router={routing_correct/n:.4f}  oracle={oracle_correct/n:.4f}  "
          f"best_single={per_model_acc[best_model]:.4f} ({best_model})")

    full_result = {
        "dataset":           dataset,
        "n_questions":       n,
        "oracle_accuracy":   round(oracle_correct  / n, 6),
        "router_accuracy":   round(routing_correct / n, 6),
        "router_correct":    routing_correct,
        "best_single_model": best_model,
        "best_single_acc":   round(per_model_acc[best_model], 6),
        "avg_single_acc":    round(avg_single_acc, 6),
        "per_model_accuracy":       {m: round(v, 6) for m, v in per_model_acc.items()},
        "routing_distribution":     {m: model_choice_counts.get(m, 0) for m in cols},
        "routing_distribution_pct": model_choice_pct,
    }

    # ------------------------------------------------------------------
    # Subtask metrics (overall: same model choice, evaluated at subtask level)
    # ------------------------------------------------------------------
    try:
        subtasks_data = load_subtasks(dataset)
    except FileNotFoundError as e:
        print(f"  [subtask] skipped: {e}")
        return full_result, None

    st_router = st_oracle = 0
    q_router  = q_oracle  = 0
    total_subs = n_with_subs = 0
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
        n_with_subs += 1
        total_subs  += n_subs
        s2j          = {m: j for j, m in enumerate(sub_cols)}

        chosen = preds[qid]
        cj = s2j.get(chosen)
        if cj is not None:
            cc = sum(binary[s][cj] for s in range(n_subs))
            st_router += cc
            q_router  += int(cc == n_subs)

        best = 0
        for m in cols:
            j = s2j.get(m)
            if j is None:
                continue
            mc = sum(binary[s][j] for s in range(n_subs))
            best = max(best, mc)
            per_model_sub[m] += mc
        st_oracle += best
        q_oracle  += int(best == n_subs)

    if not total_subs:
        return full_result, None

    pma    = {m: per_model_sub[m] / total_subs for m in cols}
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
                   help="Directory to write summary.json and subtask_summary.json")
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

    out = os.path.join(args.out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved → {out}")

    if subtask_results:
        sub_out = os.path.join(args.out_dir, "subtask_summary.json")
        with open(sub_out, "w") as f:
            json.dump(subtask_results, f, indent=2)
        print(f"Saved → {sub_out}")


if __name__ == "__main__":
    main()
