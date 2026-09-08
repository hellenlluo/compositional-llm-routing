"""Post-inference pass: for any cached row whose response lacks 'ANSWER:',
continue generating from '<response>\\n\\nANSWER:' and append the continuation
so every row has a parseable answer line.

Usage:
  python -m eval.force_answer --model <id> --cache-db /path/to/cache.db
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.cache import Cache
from eval.inference import make_backend, timed_generate
from eval.run_inference import CACHE_DEFAULT, get_model_cfg, load_cfg


def _needs_answer(response: str) -> bool:
    return "ANSWER:" not in response


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-db", default=str(CACHE_DEFAULT))
    ap.add_argument("--backend", choices=["auto", "hf", "vllm"], default="vllm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-continuation-tokens", type=int, default=128,
                    help="Tokens to generate after forcing ANSWER: prefix.")
    args = ap.parse_args()

    models_cfg, _ = load_cfg()
    model_cfg = get_model_cfg(models_cfg, args.model)

    cache = Cache(args.cache_db)

    # Find all rows needing a continuation
    pending = []
    for row in cache.iter_rows("model_id = ?", (args.model,)):
        if _needs_answer(row.get("response") or ""):
            pending.append(row)

    if not pending:
        print(f"force_answer: no rows need continuation for {args.model}")
        return
    print(f"force_answer: {len(pending)} rows need ANSWER: continuation")

    backend = make_backend(model_cfg, backend=args.backend, device=args.device)
    backend.gen_kwargs.max_new_tokens = args.max_continuation_tokens
    backend.gen_kwargs.temperature = 0.0

    # Build continuation prompts: <original_prompt><original_response>\n\nANSWER:
    def cont_prompt(row):
        return f"{row['prompt']}{row['response']}\n\nANSWER:"

    try:
        bs = args.batch_size
        rewritten = 0
        for i in range(0, len(pending), bs):
            chunk = pending[i : i + bs]
            prompts = [cont_prompt(r) for r in chunk]
            outs, _walls = timed_generate(backend, prompts)
            for row, extra in zip(chunk, outs):
                extra = extra.strip()
                # Truncate at first newline to keep the answer line concise
                first_line = extra.split("\n", 1)[0].strip()
                new_response = f"{row['response']}\n\nANSWER: {first_line}"
                cache.upsert(
                    dataset=row["dataset"],
                    question_id=row["question_id"],
                    subtask_idx=row["subtask_idx"],
                    model_id=row["model_id"],
                    prompt=row["prompt"],
                    response=new_response,
                    gold=row["gold"],
                    gold_aliases=json.loads(row["gold_aliases"]) if row.get("gold_aliases") else [],
                    gen_kwargs=json.loads(row["gen_kwargs"]) if row.get("gen_kwargs") else {},
                    wall_ms=row["wall_ms"],
                )
                rewritten += 1
            print(f"  {min(i + bs, len(pending))}/{len(pending)} ({rewritten} rewritten)")
    finally:
        backend.close()

    print(f"force_answer: done, {rewritten} responses now have ANSWER: lines")


if __name__ == "__main__":
    main()
