"""Driver: fan every (dataset, question, subtask) out to every model.

Loads each model exactly once (expensive), then iterates all dataset/phase pairs
for that model before moving on. Cache makes this idempotent, so re-running
after a crash resumes where it left off.

Examples:
  # pilot across all 10 models, all 3 datasets, both phases:
  python -m eval.run_all --phases full subtask

  # restrict to a subset while iterating:
  python -m eval.run_all --models qwen1.5-0.5b-chat gemma-3-4b-it --datasets musique --phases full
  python -m eval.run_all --limit 10           # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from eval.cache import Cache
from eval.inference import make_backend
from eval.run_inference import (
    CACHE_DEFAULT,
    _records_for,
    get_dataset_cfg,
    get_model_cfg,
    load_cfg,
    run_full,
    run_subtasks_with_gold,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="Subset of model ids. Default: all.")
    ap.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets. Default: all.")
    ap.add_argument("--phases", nargs="+", choices=["full", "subtask"], default=["full", "subtask"])
    ap.add_argument("--limit", type=int, default=None, help="Override pilot.n_per_dataset. Use a small value for smoke tests.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--backend", choices=["auto", "hf", "vllm"], default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-db", default=str(CACHE_DEFAULT))
    args = ap.parse_args()

    models_cfg, datasets_cfg = load_cfg()
    all_models = [m["id"] for m in models_cfg["models"]]
    all_datasets = list(datasets_cfg["datasets"].keys())

    pick_models = args.models or all_models
    pick_datasets = args.datasets or all_datasets
    limit = args.limit if args.limit is not None else datasets_cfg.get("pilot", {}).get("n_per_dataset")

    # pre-cache records so we don't reload per-model
    records_by_ds = {}
    for ds in pick_datasets:
        records_by_ds[ds] = _records_for(ds, get_dataset_cfg(datasets_cfg, ds), limit, args.seed)
        print(f"[{ds}] loaded {len(records_by_ds[ds])} records")

    cache = Cache(args.cache_db)

    grand_total = 0
    for model_id in pick_models:
        model_cfg = get_model_cfg(models_cfg, model_id)
        t0 = time.monotonic()
        print(f"\n=== model: {model_id} ({model_cfg['hf_repo']}) ===")
        backend = make_backend(model_cfg, backend=args.backend, device=args.device)
        try:
            for ds in pick_datasets:
                records = records_by_ds[ds]
                for phase in args.phases:
                    if phase == "full":
                        n = run_full(cache, backend, model_id, ds, records, args.batch_size)
                    else:
                        n = run_subtasks_with_gold(cache, backend, model_id, ds, records, args.batch_size)
                    grand_total += n
                    print(f"  [{model_id}/{ds}/{phase}] +{n} rows")
        finally:
            backend.close()
        print(f"=== {model_id} done in {time.monotonic() - t0:.1f}s ===")

    print(f"\ntotal new rows: {grand_total}")
    print(f"cache size: {cache.count()} rows @ {args.cache_db}")


if __name__ == "__main__":
    main()
