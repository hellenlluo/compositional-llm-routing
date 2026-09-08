"""Export cached model generations to a portable JSONL tree.

Reads from outputs/cache.db and writes:

  outputs/runs/<dataset>/<model_id>/<phase>.jsonl

Each line is the minimum any judge (Prometheus, GPT-4, exact-match, ...) needs:
  {dataset, question_id, subtask_idx, model_id, question, gold, gold_aliases,
   options, response}

The giant inference-time prompt stays in SQLite — we don't round-trip it.

Usage:
  python -m eval.export_runs                         # export everything
  python -m eval.export_runs --model qwen3-8b        # one model only
  python -m eval.export_runs --dataset musique       # one dataset only
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.cache import Cache
from eval.loaders import iter_records
from eval.run_inference import CACHE_DEFAULT, REPO_ROOT, get_dataset_cfg, load_cfg


def _question_lookup(dataset: str, ds_cfg: dict) -> dict:
    """Build {question_id: Record} so we can pull per-subtask question text and gold."""
    path = REPO_ROOT / ds_cfg["path"]
    return {r.id: r for r in iter_records(ds_cfg["flavor"], path)}


def _line_from_row(row: dict, record) -> dict:
    sidx = row["subtask_idx"]
    if sidx == -1:
        question = record.question
        gold = record.gold
        gold_aliases = record.gold_aliases
        options = record.context_payload.get("final_options")
    else:
        s = record.subtasks[sidx]
        question = s.question
        gold = s.gold
        gold_aliases = []
        options = s.options
    return {
        "dataset": row["dataset"],
        "question_id": row["question_id"],
        "subtask_idx": sidx,
        "model_id": row["model_id"],
        "question": question,
        "gold": gold,
        "gold_aliases": gold_aliases,
        "options": options,
        "response": row["response"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-db", default=str(CACHE_DEFAULT))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "runs"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    _, datasets_cfg = load_cfg()

    lookups: dict[str, dict] = {}

    where, params = [], []
    if args.model:
        where.append("model_id = ?")
        params.append(args.model)
    if args.dataset:
        where.append("dataset = ?")
        params.append(args.dataset)
    where_sql = " AND ".join(where) if where else ""

    cache = Cache(args.cache_db)

    # Bucket by (dataset, model, phase) so we write each file once.
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in cache.iter_rows(where_sql, tuple(params)):
        ds = row["dataset"]
        if ds not in lookups:
            lookups[ds] = _question_lookup(ds, get_dataset_cfg(datasets_cfg, ds))
        record = lookups[ds].get(row["question_id"])
        if record is None:
            print(f"  skip: {row['question_id']} not in {ds} (was the dataset re-cleaned?)")
            continue
        phase = "full" if row["subtask_idx"] == -1 else "subtask"
        line = _line_from_row(row, record)
        buckets[(ds, row["model_id"], phase)].append(line)

    n_files = 0
    for (ds, model_id, phase), lines in buckets.items():
        # Stable ordering: (question_id, subtask_idx)
        lines.sort(key=lambda x: (x["question_id"], x["subtask_idx"]))
        out = out_root / ds / model_id / f"{phase}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        try:
            rel = out.relative_to(REPO_ROOT)
        except ValueError:
            rel = out
        print(f"  wrote {rel} ({len(lines)} rows)")
        n_files += 1
    print(f"\nexported {n_files} files from {sum(len(v) for v in buckets.values())} cache rows")


if __name__ == "__main__":
    main()
