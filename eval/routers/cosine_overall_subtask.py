#!/usr/bin/env python3
"""
Compute overall-subtask accuracy for the cosine-similarity router.

"Overall subtask": use the same model chosen for the full question, then
evaluate whether that model answered ALL subtasks of the question correctly
(strict all-or-nothing, matching the definition used by all other routers).

Reads  : outputs/routing/<model-set>/embedding_<dataset>_full.json
Reads  : outputs/updated-matrices/<dataset>_subtasks.json
Writes : outputs/routing/<model-set>/overall_subtask_summary.json

Usage:
    python -m eval.routers.cosine_overall_subtask --model-set no_qwen3
    python -m eval.routers.cosine_overall_subtask --model-set small4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[2]
MATRICES_DIR = REPO_ROOT / "outputs" / "updated-matrices"
DATASETS     = ["morehopqa", "musique"]  # stepcot was not evaluated


def compute(dataset: str, model_set: str) -> dict | None:
    routing_path = REPO_ROOT / "outputs" / "routing" / model_set / f"embedding_{dataset}_full.json"
    if not routing_path.exists():
        print(f"[{dataset}] full routing file not found: {routing_path}")
        return None

    with open(routing_path) as f:
        routing = json.load(f)

    model_ids  = routing["model_ids"]       # list of model name strings
    routed_idx = routing["routed_idx"]      # list[int], one per question
    row_ids    = routing["row_ids"]         # list of question IDs, same order
    assert len(routed_idx) == len(row_ids), "length mismatch"

    # Map question_id → chosen model name
    chosen_by_qid = {qid: model_ids[idx] for qid, idx in zip(row_ids, routed_idx)}

    # Load subtask matrix
    sub_path = MATRICES_DIR / f"{dataset}_subtasks.json"
    if not sub_path.exists():
        print(f"[{dataset}] subtask matrix not found: {sub_path}")
        return None
    with open(sub_path) as f:
        subtask_oracle = json.load(f)

    st_router = st_oracle = 0
    q_router  = q_oracle  = 0
    total_subs = n_with_subs = 0
    per_model_sub: dict[str, int]   = {m: 0 for m in model_ids}
    per_model_total: dict[str, int] = {m: 0 for m in model_ids}

    for qid, chosen in chosen_by_qid.items():
        if qid not in subtask_oracle:
            continue
        entry    = subtask_oracle[qid]
        sub_cols = entry["cols"]
        binary   = entry["binary"]   # list[list[int]]: [n_subs][n_models]
        n_subs   = len(binary)
        if not n_subs:
            continue
        n_with_subs += 1
        total_subs  += n_subs
        s2j = {m: j for j, m in enumerate(sub_cols)}

        # Router: chosen model
        cj = s2j.get(chosen)
        if cj is not None:
            cc = sum(binary[s][cj] for s in range(n_subs))
            st_router += cc
            q_router  += int(cc == n_subs)

        # Oracle: best single model for this question
        best = 0
        for m in model_ids:
            j = s2j.get(m)
            if j is None:
                continue
            sc = sum(binary[s][j] for s in range(n_subs))
            best = max(best, sc)
            per_model_sub[m]   += sc
            per_model_total[m] += n_subs
        st_oracle += best
        q_oracle  += int(best == n_subs)

    if not total_subs:
        print(f"[{dataset}] no subtask data found")
        return None

    pma     = {m: per_model_sub[m] / max(per_model_total[m], 1) for m in model_ids}
    best_m  = max(pma, key=pma.get)
    avg_acc = sum(pma.values()) / len(pma)

    result = {
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
        "avg_single_subtask_acc":   round(avg_acc, 6),
        "per_model_subtask_accuracy": {m: round(v, 6) for m, v in pma.items()},
    }

    print(f"[{dataset}] n_q={n_with_subs}  n_subs={total_subs}")
    print(f"  subtask_router={st_router/total_subs:.4f}  "
          f"oracle={st_oracle/total_subs:.4f}")
    print(f"  question_router={q_router/n_with_subs:.4f}  "
          f"q_oracle={q_oracle/n_with_subs:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-set", required=True, help="e.g. no_qwen3 or small4")
    ap.add_argument("--datasets",  nargs="+", default=DATASETS)
    args = ap.parse_args()

    results = []
    for ds in args.datasets:
        r = compute(ds, args.model_set)
        if r:
            results.append(r)

    if not results:
        print("No results — nothing saved.")
        return

    out_path = REPO_ROOT / "outputs" / "routing" / args.model_set / "overall_subtask_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
