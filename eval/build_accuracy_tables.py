"""Generate accuracy table files from updated-matrices/.

full_accuracy_table.json
  Per (dataset, model_id), computed from <dataset>_full_binary.json:
    n_total    = number of questions in the matrix
    n_judged   = non-null entries for that model
    n_correct  = entries equal to 1 for that model
    accuracy   = n_correct / n_total

full_aggregated_accuracy.json
  Per dataset: the highest single-model accuracy from full_accuracy_table.json.
    n_total      = number of questions in the dataset
    best_model   = model_id achieving the highest accuracy
    n_correct    = that model's n_correct
    accuracy     = that model's accuracy (max across all models)

subtask_aggregated_accuracy.json
  Per dataset, computed from <dataset>_subtasks.json.
  A question is counted as correct if the pool of models collectively has at
  least one correct response (binary == 1) for every subtask step.
    n_total   = total questions in the dataset
    n_correct = questions where every subtask row has ≥1 model with binary == 1
    accuracy  = n_correct / n_total

full_task_accuracy_oracle.json
  Per dataset, computed from <dataset>_full_binary.json.
  Oracle upper bound: a question is correct if ANY model answers it correctly
  (i.e. what a perfect full-task router could achieve).
    n_total   = total questions in the dataset
    n_correct = questions where ≥1 model has binary == 1
    accuracy  = n_correct / n_total

subtask_accuracy_oracle.json
  Per dataset, computed from <dataset>_subtasks.json.
  Oracle upper bound: a question is correct if EVERY subtask hop has ≥1 model
  that answers it correctly (i.e. what a perfect per-hop subtask router could
  achieve). Identical definition to subtask_aggregated_accuracy.json but stored
  in a consistent oracle file alongside full_task_accuracy_oracle.json.
    n_total   = total questions in the dataset
    n_correct = questions where every subtask row has ≥1 model with binary == 1
    accuracy  = n_correct / n_total
"""
from __future__ import annotations

import json
from pathlib import Path

MATRICES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "updated-matrices"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

DATASETS = ["morehopqa", "musique", "stepcot"]


# ---------------------------------------------------------------------------
# full_accuracy_table.json
# ---------------------------------------------------------------------------

def build_full_accuracy_table() -> list[dict]:
    records = []
    for dataset in DATASETS:
        path = MATRICES_DIR / f"{dataset}_full_binary.json"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        mat = json.loads(path.read_text())
        rows = mat["rows"]
        cols = mat["cols"]
        values = mat["values"]
        n_questions = len(rows)
        for col_idx, model_id in enumerate(cols):
            n_judged = sum(1 for row in values if row[col_idx] is not None)
            n_correct = sum(1 for row in values if row[col_idx] == 1)
            records.append({
                "dataset": dataset,
                "model_id": model_id,
                "n_total": n_questions,
                "n_judged": n_judged,
                "n_correct": n_correct,
                "accuracy": n_correct / n_questions if n_questions else None,
            })
        print(f"  full: {dataset} — {n_questions} questions × {len(cols)} models")
    return records


# ---------------------------------------------------------------------------
# full_aggregated_accuracy.json
# ---------------------------------------------------------------------------

def build_full_aggregated_accuracy(full_records: list[dict]) -> list[dict]:
    """Per dataset: highest single-model accuracy from full_accuracy_table."""
    from collections import defaultdict
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in full_records:
        by_dataset[r["dataset"]].append(r)
    records = []
    for dataset in DATASETS:
        rows = by_dataset.get(dataset, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: r["accuracy"] or 0)
        records.append({
            "dataset": dataset,
            "n_total": best["n_total"],
            "best_model": best["model_id"],
            "n_correct": best["n_correct"],
            "accuracy": best["accuracy"],
        })
        print(f"  full aggregated: {dataset} — best={best['model_id']} ({best['accuracy']:.4f})")
    return records


# ---------------------------------------------------------------------------
# subtask_aggregated_accuracy.json
# ---------------------------------------------------------------------------

def build_subtask_accuracy_table() -> list[dict]:
    """Aggregated decomposition accuracy per dataset.

    A question is correct if every subtask row has at least one model with
    binary == 1 (i.e. the model pool can collectively solve every step).
    """
    records = []
    for dataset in DATASETS:
        path = MATRICES_DIR / f"{dataset}_subtasks.json"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        per_q: dict = json.loads(path.read_text())
        total = len(per_q)
        n_correct = 0
        for qid, entry in per_q.items():
            binary = entry["binary"]
            cols = entry["cols"]
            n_models = len(cols)
            # Question is correct if every subtask row has ≥1 model with binary == 1
            if all(any(row[c] == 1 for c in range(n_models)) for row in binary):
                n_correct += 1
        records.append({
            "dataset": dataset,
            "n_total": total,
            "n_correct": n_correct,
            "accuracy": n_correct / total if total else None,
        })
        print(f"  subtask: {dataset} — {n_correct}/{total} questions fully solvable by pool")
    return records


# ---------------------------------------------------------------------------
# full_task_accuracy_oracle.json
# ---------------------------------------------------------------------------

def build_full_task_oracle() -> list[dict]:
    """Per dataset: fraction of questions where ANY model answers correctly.

    This is the upper bound achievable by a perfect full-task router.
    """
    records = []
    for dataset in DATASETS:
        path = MATRICES_DIR / f"{dataset}_full_binary.json"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        mat = json.loads(path.read_text())
        values = mat["values"]
        n_total = len(values)
        n_correct = sum(1 for row in values if any(v == 1 for v in row))
        records.append({
            "dataset": dataset,
            "n_total": n_total,
            "n_correct": n_correct,
            "accuracy": n_correct / n_total if n_total else None,
        })
        print(f"  full oracle: {dataset} — {n_correct}/{n_total} ({n_correct/n_total:.4f})")
    return records


# ---------------------------------------------------------------------------
# subtask_accuracy_oracle.json
# ---------------------------------------------------------------------------

def build_subtask_oracle() -> list[dict]:
    """Per dataset: fraction of questions where every subtask hop has ≥1 correct model.

    This is the upper bound achievable by a perfect per-hop subtask router.
    Same definition as subtask_aggregated_accuracy, stored in oracle file format.
    """
    records = []
    for dataset in DATASETS:
        path = MATRICES_DIR / f"{dataset}_subtasks.json"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        per_q: dict = json.loads(path.read_text())
        total = len(per_q)
        n_correct = 0
        for qid, entry in per_q.items():
            binary = entry["binary"]
            n_models = len(entry["cols"])
            if all(any(row[c] == 1 for c in range(n_models)) for row in binary):
                n_correct += 1
        records.append({
            "dataset": dataset,
            "n_total": total,
            "n_correct": n_correct,
            "accuracy": n_correct / total if total else None,
        })
        print(f"  subtask oracle: {dataset} — {n_correct}/{total} ({n_correct/total:.4f})")
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building full_accuracy_table.json ...")
    full_records = build_full_accuracy_table()
    (OUT_DIR / "full_accuracy_table.json").write_text(json.dumps(full_records, indent=2))
    print(f"  wrote full_accuracy_table.json ({len(full_records)} records)")

    print("Building full_aggregated_accuracy.json ...")
    full_agg_records = build_full_aggregated_accuracy(full_records)
    (OUT_DIR / "full_aggregated_accuracy.json").write_text(json.dumps(full_agg_records, indent=2))
    print(f"  wrote full_aggregated_accuracy.json ({len(full_agg_records)} records)")

    print("Building subtask_aggregated_accuracy.json ...")
    subtask_records = build_subtask_accuracy_table()
    (OUT_DIR / "subtask_aggregated_accuracy.json").write_text(json.dumps(subtask_records, indent=2))
    print(f"  wrote subtask_aggregated_accuracy.json ({len(subtask_records)} records)")

    print("Building full_task_accuracy_oracle.json ...")
    full_oracle_records = build_full_task_oracle()
    (OUT_DIR / "full_task_accuracy_oracle.json").write_text(json.dumps(full_oracle_records, indent=2))
    print(f"  wrote full_task_accuracy_oracle.json ({len(full_oracle_records)} records)")

    print("Building subtask_accuracy_oracle.json ...")
    subtask_oracle_records = build_subtask_oracle()
    (OUT_DIR / "subtask_accuracy_oracle.json").write_text(json.dumps(subtask_oracle_records, indent=2))
    print(f"  wrote subtask_accuracy_oracle.json ({len(subtask_oracle_records)} records)")

    print("done.")


if __name__ == "__main__":
    main()
