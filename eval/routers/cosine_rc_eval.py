#!/usr/bin/env python3
"""
Compute cosine-routing metrics for R+C questions only.

Three metrics (matching definitions used for all other routers):
  full_task     : fraction of R+C questions routed correctly
  subtask_micro : fraction of subtask slots correct (same model as full-question routing)
  chained       : fraction of R+C questions where every subtask is routed correctly
                  (each subtask routed independently via the subtask embedding file)

Reads  : outputs/routing/<model-set>/embedding_<dataset>_full.json
         outputs/routing/<model-set>/embedding_<dataset>_subtask.json
         outputs/updated-matrices/<dataset>_full_binary.json
         outputs/updated-matrices/<dataset>_subtasks.json
         outputs/rc_qids.json
Writes : outputs/routing/<model-set>/rc_summary.json

Usage:
    python -m eval.routers.cosine_rc_eval --model-set no_qwen3
    python -m eval.routers.cosine_rc_eval --model-set small4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[2]
MATRICES_DIR = REPO_ROOT / "outputs" / "updated-matrices"
RC_QIDS_PATH = REPO_ROOT / "outputs" / "rc_qids.json"
DATASETS     = ["morehopqa", "musique"]


def compute(dataset: str, model_set: str, rc_qids: set[str]) -> dict | None:
    routing_dir = REPO_ROOT / "outputs" / "routing" / model_set

    full_path    = routing_dir / f"embedding_{dataset}_full.json"
    subtask_path = routing_dir / f"embedding_{dataset}_subtask.json"

    if not full_path.exists():
        print(f"[{dataset}] full routing file not found: {full_path}")
        return None

    with open(full_path) as f:
        full_data = json.load(f)

    model_ids  = full_data["model_ids"]
    row_ids    = full_data["row_ids"]
    routed_idx = full_data["routed_idx"]

    # ── Full-task R+C ──────────────────────────────────────────────────────────
    full_bin_path = MATRICES_DIR / f"{dataset}_full_binary.json"
    if not full_bin_path.exists():
        print(f"[{dataset}] full binary oracle not found: {full_bin_path}")
        return None

    with open(full_bin_path) as f:
        full_oracle = json.load(f)

    all_cols  = full_oracle["cols"]
    all_rows  = full_oracle["rows"]      # question IDs in oracle order
    values    = full_oracle["values"]
    col_idx   = {m: all_cols.index(m) for m in model_ids}
    row_idx   = {qid: i for i, qid in enumerate(all_rows)}

    full_router_correct = 0
    full_oracle_correct = 0
    n_full = 0

    for qid, ridx in zip(row_ids, routed_idx):
        if qid not in rc_qids:
            continue
        if qid not in row_idx:
            continue
        n_full += 1
        oracle_row = values[row_idx[qid]]
        chosen_model = model_ids[ridx]
        if oracle_row[col_idx[chosen_model]]:
            full_router_correct += 1
        if any(oracle_row[col_idx[m]] for m in model_ids):
            full_oracle_correct += 1

    if n_full == 0:
        print(f"[{dataset}] no R+C full-task questions found")
        return None

    print(f"[{dataset}] R+C full: n={n_full}  router={full_router_correct/n_full:.4f}"
          f"  oracle={full_oracle_correct/n_full:.4f}")

    # ── Subtask micro (same model as full-question routing) ────────────────────
    sub_oracle_path = MATRICES_DIR / f"{dataset}_subtasks.json"
    if not sub_oracle_path.exists():
        print(f"[{dataset}] subtask oracle not found: {sub_oracle_path}")
        return None

    with open(sub_oracle_path) as f:
        subtask_oracle = json.load(f)

    # Reuse full routing's chosen model per question
    chosen_by_qid = {
        qid: model_ids[ridx]
        for qid, ridx in zip(row_ids, routed_idx)
        if qid in rc_qids
    }

    st_router = st_oracle = 0
    n_with_subs = total_subs = 0

    for qid, chosen in chosen_by_qid.items():
        if qid not in subtask_oracle:
            continue
        entry   = subtask_oracle[qid]
        binary  = entry["binary"]
        n_subs  = len(binary)
        if not n_subs:
            continue
        n_with_subs += 1
        total_subs  += n_subs
        s2j = {m: j for j, m in enumerate(entry["cols"])}

        cj = s2j.get(chosen)
        if cj is not None:
            st_router += sum(binary[s][cj] for s in range(n_subs))

        best = max(
            (sum(binary[s][s2j[m]] for s in range(n_subs))
             for m in model_ids if m in s2j),
            default=0,
        )
        st_oracle += best

    print(f"[{dataset}] R+C subtask micro: n_q={n_with_subs}  n_subs={total_subs}"
          f"  router={st_router/max(total_subs,1):.4f}"
          f"  oracle={st_oracle/max(total_subs,1):.4f}")

    # ── Chained subtask (independent routing per subtask) ─────────────────────
    chained_router_correct = chained_oracle_correct = 0
    n_chained_qs = 0
    chained_result = None

    if subtask_path.exists():
        with open(subtask_path) as f:
            sub_data = json.load(f)

        sub_row_ids    = sub_data["row_ids"]    # 'qid::stepN'
        sub_routed_idx = sub_data["routed_idx"]
        sub_model_ids  = sub_data["model_ids"]  # may differ from full model_ids

        # Group subtask slots by base question ID
        from collections import defaultdict
        slots_by_qid: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for slot_id, ridx in zip(sub_row_ids, sub_routed_idx):
            base_qid = slot_id.split("::")[0]
            # subtask index from oracle
            slots_by_qid[base_qid].append((slot_id, ridx))

        for qid, slots in slots_by_qid.items():
            if qid not in rc_qids:
                continue
            if qid not in subtask_oracle:
                continue
            entry  = subtask_oracle[qid]
            binary = entry["binary"]
            n_subs = len(binary)
            if not n_subs or len(slots) != n_subs:
                continue
            s2j = {m: j for j, m in enumerate(entry["cols"])}
            n_chained_qs += 1

            # Check each subtask slot in order
            all_correct = True
            for sidx, (slot_id, ridx) in enumerate(slots):
                chosen = sub_model_ids[ridx]
                cj = s2j.get(chosen)
                if cj is None or not binary[sidx][cj]:
                    all_correct = False
                    break
            if all_correct:
                chained_router_correct += 1

            # Oracle: is it even possible for all to be correct?
            oracle_all = all(
                any(binary[s][s2j[m]] for m in sub_model_ids if m in s2j)
                for s in range(n_subs)
            )
            if oracle_all:
                chained_oracle_correct += 1

        chained_result = {
            "chained_n_questions":  n_chained_qs,
            "chained_accuracy":     round(chained_router_correct / max(n_chained_qs, 1), 6),
            "chained_correct":      chained_router_correct,
            "oracle_chained_accuracy": round(chained_oracle_correct / max(n_chained_qs, 1), 6),
        }
        print(f"[{dataset}] R+C chained: n_q={n_chained_qs}"
              f"  chained={chained_router_correct/max(n_chained_qs,1):.4f}"
              f"  oracle={chained_oracle_correct/max(n_chained_qs,1):.4f}")
    else:
        print(f"[{dataset}] subtask routing file not found — chained not computed")

    result: dict = {
        "dataset":                  dataset,
        "n_questions":              n_full,
        "full_router_accuracy":     round(full_router_correct / n_full,         6),
        "full_oracle_accuracy":     round(full_oracle_correct / n_full,         6),
        "n_subtask_questions":      n_with_subs,
        "n_subtasks":               total_subs,
        "subtask_router_accuracy":  round(st_router / max(total_subs, 1),       6),
        "oracle_subtask_accuracy":  round(st_oracle / max(total_subs, 1),       6),
    }
    if chained_result:
        result.update(chained_result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-set", required=True, help="e.g. no_qwen3 or small4")
    ap.add_argument("--datasets",  nargs="+", default=DATASETS)
    args = ap.parse_args()

    if not RC_QIDS_PATH.exists():
        raise FileNotFoundError(f"RC qids not found: {RC_QIDS_PATH}")
    with open(RC_QIDS_PATH) as f:
        rc_qids_by_dataset = json.load(f)

    results = []
    for ds in args.datasets:
        rc_qids = set(rc_qids_by_dataset.get(ds, []))
        if not rc_qids:
            print(f"[{ds}] no RC qids — skipping")
            continue
        r = compute(ds, args.model_set, rc_qids)
        if r:
            results.append(r)

    if not results:
        print("No results — nothing saved.")
        return

    out_path = REPO_ROOT / "outputs" / "routing" / args.model_set / "rc_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
