"""Run one (model, dataset, phase) combination and populate the cache.

Examples:
  python -m eval.run_inference --model qwen1.5-0.5b-chat --dataset musique --phase full --limit 10
  python -m eval.run_inference --model qwen3-8b       --dataset morehopqa --phase subtask --limit 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eval.cache import Cache, prompt_hash
from eval.inference import make_backend, timed_generate
from eval.loaders import iter_records, pilot_sample
from eval.prompts import full_question, subtask as subtask_prompts


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DEFAULT = REPO_ROOT / "outputs" / "cache.db"


def load_cfg() -> tuple[dict, dict]:
    models = yaml.safe_load((REPO_ROOT / "eval/configs/models.yaml").read_text())
    datasets = yaml.safe_load((REPO_ROOT / "eval/configs/datasets.yaml").read_text())
    return models, datasets


def get_model_cfg(models: dict, model_id: str) -> dict:
    for m in models["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"unknown model id: {model_id}")


def get_dataset_cfg(datasets: dict, dataset: str) -> dict:
    if dataset not in datasets["datasets"]:
        raise KeyError(f"unknown dataset: {dataset}")
    return datasets["datasets"][dataset]


def _records_for(dataset: str, ds_cfg: dict, limit: int | None, seed: int,
                 shard_idx: int = 0, num_shards: int = 1) -> list:
    path = REPO_ROOT / ds_cfg["path"]
    recs = list(iter_records(ds_cfg["flavor"], path))
    if limit is not None:
        recs = pilot_sample(recs, n=limit, seed=seed)
    else:
        # Keep deterministic ordering so shards don't overlap across processes.
        recs = sorted(recs, key=lambda r: r.id)
    if num_shards > 1:
        recs = recs[shard_idx::num_shards]
    return recs


def run_full(cache: Cache, backend, model_id: str, dataset: str, records: list, batch_size: int) -> int:
    """Returns the number of new rows written."""
    pending = []
    for r in records:
        prompt = full_question.build(r)
        if cache.has(dataset, r.id, -1, model_id, prompt_hash(prompt)):
            continue
        pending.append((r, prompt))
    return _drain(cache, backend, model_id, dataset, pending, subtask_idx_fn=lambda _r: -1, batch_size=batch_size)


def run_subtasks(cache: Cache, backend, model_id: str, dataset: str, records: list, batch_size: int) -> int:
    pending = []
    for r in records:
        for s in r.subtasks:
            prompt = subtask_prompts.build(r, s.idx)
            if cache.has(dataset, r.id, s.idx, model_id, prompt_hash(prompt)):
                continue
            pending.append(((r, s.idx), prompt))
    return _drain(cache, backend, model_id, dataset, pending, subtask_idx_fn=lambda key: key[1],
                  record_fn=lambda key: key[0], batch_size=batch_size)


def _drain(
    cache: Cache, backend, model_id: str, dataset: str, pending: list, *,
    subtask_idx_fn, record_fn=lambda r: r, batch_size: int,
) -> int:
    n_written = 0
    gen_kwargs = backend.gen_kwargs.as_dict()
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        prompts = [p for _k, p in chunk]
        responses, walls = timed_generate(backend, prompts)
        for (key, prompt), resp, wall in zip(chunk, responses, walls):
            r = record_fn(key)
            cache.upsert(
                dataset=dataset,
                question_id=r.id,
                subtask_idx=subtask_idx_fn(key),
                model_id=model_id,
                prompt=prompt,
                response=resp,
                gold=r.gold,  # full-Q path; subtask path overrides below
                gold_aliases=r.gold_aliases,
                gen_kwargs=gen_kwargs,
                wall_ms=wall,
            )
            n_written += 1
        print(f"  [{model_id}/{dataset}] {min(i + batch_size, len(pending))}/{len(pending)}")
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--phase", choices=["full", "subtask"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="Sample N records deterministically. Omit for full dataset.")
    ap.add_argument("--no-limit", action="store_true", help="Override datasets.yaml pilot default and use the full dataset.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--backend", choices=["auto", "hf", "vllm"], default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-db", default=str(CACHE_DEFAULT))
    ap.add_argument("--max-new-tokens", type=int, default=None, help="Override GenKwargs.max_new_tokens (default 512).")
    ap.add_argument("--force-answer-regex", default=None, help="vLLM-only: constrain output to match this regex so ANSWER: must appear.")
    ap.add_argument("--shard-idx", type=int, default=0, help="This worker's shard index (0-based).")
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    args = ap.parse_args()

    models_cfg, datasets_cfg = load_cfg()
    model_cfg = get_model_cfg(models_cfg, args.model)
    ds_cfg = get_dataset_cfg(datasets_cfg, args.dataset)

    if args.limit is None and not args.no_limit:
        args.limit = datasets_cfg.get("pilot", {}).get("n_per_dataset")

    records = _records_for(args.dataset, ds_cfg, args.limit, args.seed,
                           shard_idx=args.shard_idx, num_shards=args.num_shards)
    cache = Cache(args.cache_db)
    backend = make_backend(model_cfg, backend=args.backend, device=args.device)
    if args.max_new_tokens:
        backend.gen_kwargs.max_new_tokens = args.max_new_tokens
    if args.force_answer_regex:
        backend.force_answer_regex = args.force_answer_regex
    try:
        if args.phase == "full":
            n = run_full(cache, backend, args.model, args.dataset, records, args.batch_size)
        else:
            n = run_subtasks_with_gold(cache, backend, args.model, args.dataset, records, args.batch_size)
    finally:
        backend.close()
    print(f"done: wrote {n} new rows ({args.model} / {args.dataset} / {args.phase})")


# --- subtask path that stamps the correct per-subtask gold ---
def run_subtasks_with_gold(cache: Cache, backend, model_id: str, dataset: str, records: list, batch_size: int) -> int:
    pending = []
    for r in records:
        for s in r.subtasks:
            prompt = subtask_prompts.build(r, s.idx)
            if cache.has(dataset, r.id, s.idx, model_id, prompt_hash(prompt)):
                continue
            pending.append((r, s, prompt))
    n_written = 0
    gen_kwargs = backend.gen_kwargs.as_dict()
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        prompts = [p for *_k, p in chunk]
        responses, walls = timed_generate(backend, prompts)
        for (r, s, prompt), resp, wall in zip(chunk, responses, walls):
            cache.upsert(
                dataset=dataset,
                question_id=r.id,
                subtask_idx=s.idx,
                model_id=model_id,
                prompt=prompt,
                response=resp,
                gold=s.gold,
                gold_aliases=[],
                gen_kwargs=gen_kwargs,
                wall_ms=wall,
            )
            n_written += 1
        print(f"  [{model_id}/{dataset}/subtask] {min(i + batch_size, len(pending))}/{len(pending)}")
    return n_written


if __name__ == "__main__":
    main()
