#!/usr/bin/env python3
"""
Derive EmbedLLM metrics from existing per-dataset predictions JSON files — no GPU needed.

Each dataset has a predictions file saved as <out-dir>/{dataset}_predictions.json
(a list of {question_id, chosen_model, correct, model_scores, ...} records).

Useful for computing metrics on a question-ID subset (e.g. R+C questions)
without re-running inference.  Chained subtask evaluation requires GPU
and is NOT handled here (run eval_subtasks.py with --qid-filter for that).

Usage:
  python eval_from_preds.py \
      --preds-dir eval_results/no_qwen3 \
      --out-dir   eval_results/rc_questions/no_qwen3 \
      --qid-filter ../outputs/rc_qids.json
"""

import argparse
import json
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MATRIX_DIR  = os.path.join(PROJECT_DIR, "outputs", "updated-matrices")
DATASETS    = ["morehopqa", "musique", "stepcot"]


def load_oracle(dataset: str):
    path = os.path.join(MATRIX_DIR, f"{dataset}_full_binary.json")
    with open(path) as f:
        d = json.load(f)
    return d["rows"], d["cols"], d["values"]


def load_subtask_oracle(dataset: str) -> dict | None:
    path = os.path.join(MATRIX_DIR, f"{dataset}_subtasks.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_dataset_metrics(
    dataset: str,
    predictions: list[dict],
    allowed_qids: set[str] | None,
) -> tuple[dict, dict | None]:
    oracle_rows, oracle_cols, oracle_values = load_oracle(dataset)
    row_idx = {q: i for i, q in enumerate(oracle_rows)}
    col_idx = {m: j for j, m in enumerate(oracle_cols)}

    if allowed_qids is not None:
        predictions = [p for p in predictions if p["question_id"] in allowed_qids]
    if not predictions:
        print(f"  [{dataset}] 0 questions after filtering — skipping")
        return {}, None
    n = len(predictions)
    print(f"\n[{dataset}] {n} questions")

    # Restrict to models present in the oracle
    shared_models = [m for m in oracle_cols if m in col_idx]

    routing_correct = oracle_correct = 0
    per_model_correct = {m: 0 for m in shared_models}
    routing_dist = {m: 0 for m in shared_models}

    for pred in predictions:
        qid    = pred["question_id"]
        chosen = pred["chosen_model"]
        ri     = row_idx.get(qid)
        if ri is None:
            continue

        row_vals = oracle_values[ri]

        cj = col_idx.get(chosen)
        if cj is not None:
            routing_correct += row_vals[cj]
            routing_dist[chosen] = routing_dist.get(chosen, 0) + 1

        oracle_correct += int(any(row_vals[col_idx[m]] for m in shared_models))

        for m in shared_models:
            per_model_correct[m] += row_vals[col_idx[m]]

    per_model_acc  = {m: per_model_correct[m] / n for m in shared_models}
    best_model     = max(per_model_acc, key=per_model_acc.get)
    avg_single_acc = sum(per_model_acc.values()) / len(per_model_acc)
    routing_dist_pct = {m: round(routing_dist.get(m, 0) / n, 6) for m in shared_models}

    print(f"  router={routing_correct/n:.4f}  oracle={oracle_correct/n:.4f}  "
          f"best_single={per_model_acc[best_model]:.4f} ({best_model})")

    full_result = {
        "dataset":                  dataset,
        "n_questions":              n,
        "oracle_accuracy":          round(oracle_correct  / n, 6),
        "router_accuracy":          round(routing_correct / n, 6),
        "router_correct":           routing_correct,
        "best_single_model":        best_model,
        "best_single_acc":          round(per_model_acc[best_model], 6),
        "avg_single_acc":           round(avg_single_acc, 6),
        "per_model_accuracy":       {m: round(v, 6) for m, v in per_model_acc.items()},
        "routing_distribution":     {m: routing_dist.get(m, 0) for m in shared_models},
        "routing_distribution_pct": routing_dist_pct,
    }

    # ------------------------------------------------------------------
    # Overall subtask metrics (same model choice evaluated at subtask level)
    # ------------------------------------------------------------------
    subtask_oracle = load_subtask_oracle(dataset)
    if subtask_oracle is None:
        print(f"  [subtask] skipped: {dataset}_subtasks.json not found")
        return full_result, None

    st_router = st_oracle = 0
    q_router  = q_oracle  = 0
    total_subs = n_with_subs = 0
    per_model_sub: dict[str, int] = {m: 0 for m in shared_models}

    for pred in predictions:
        qid = pred["question_id"]
        if qid not in subtask_oracle:
            continue
        entry    = subtask_oracle[qid]
        sub_cols = entry["cols"]
        binary   = entry["binary"]
        n_subs   = len(binary)
        if not n_subs:
            continue
        n_with_subs += 1
        total_subs  += n_subs
        s2j          = {m: j for j, m in enumerate(sub_cols)}

        chosen = pred["chosen_model"]
        cj = s2j.get(chosen)
        if cj is not None:
            cc = sum(binary[s][cj] for s in range(n_subs))
            st_router += cc
            q_router  += int(cc == n_subs)

        best = max(
            (sum(binary[s][s2j[m]] for s in range(n_subs))
             for m in shared_models if m in s2j),
            default=0,
        )
        st_oracle += best
        q_oracle  += int(best == n_subs)

        for m in shared_models:
            j = s2j.get(m)
            if j is not None:
                per_model_sub[m] += sum(binary[s][j] for s in range(n_subs))

    if not total_subs:
        return full_result, None

    pma    = {m: per_model_sub[m] / total_subs for m in shared_models}
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
    p.add_argument("--preds-dir", required=True,
                   help="Directory containing {dataset}_predictions.json files")
    p.add_argument("--out-dir",   required=True,
                   help="Directory to write summary.json and overall_subtask_summary.json")
    p.add_argument("--datasets",  nargs="+", default=DATASETS)
    p.add_argument("--qid-filter", default=None,
                   help="Path to {dataset: [qid, ...]} JSON to restrict questions")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    qid_filter = None
    if args.qid_filter:
        with open(args.qid_filter) as f:
            qid_filter = json.load(f)

    full_results    = []
    subtask_results = []
    for ds in args.datasets:
        pred_path = os.path.join(args.preds_dir, f"{ds}_predictions.json")
        if not os.path.isfile(pred_path):
            print(f"[{ds}] predictions not found at {pred_path} — skipping")
            continue
        with open(pred_path) as f:
            predictions = json.load(f)

        allowed = set(qid_filter[ds]) if (qid_filter and ds in qid_filter) else None
        full_r, sub_r = compute_dataset_metrics(ds, predictions, allowed)
        if full_r:
            full_results.append(full_r)
        if sub_r:
            subtask_results.append(sub_r)

    out = os.path.join(args.out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved → {out}")

    if subtask_results:
        sub_out = os.path.join(args.out_dir, "overall_subtask_summary.json")
        with open(sub_out, "w") as f:
            json.dump(subtask_results, f, indent=2)
        print(f"Saved → {sub_out}")


if __name__ == "__main__":
    main()
