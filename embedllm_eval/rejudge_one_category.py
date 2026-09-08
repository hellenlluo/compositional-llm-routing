"""Re-judge one category of existing deepseek-r1-distill-llama-8b results.

Skips inference entirely — reads the existing result JSON, strips <think> blocks
from model_answer, then re-runs the same Qwen2.5-32B judge used in run_one_job.py
and overwrites the score field in-place.

Usage:
  python embedllm_eval/rejudge_one_category.py --category mmlu_marketing
  python embedllm_eval/rejudge_one_category.py --category law --backend auto --device cuda
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

MODEL_ID  = "deepseek-r1-distill-llama-8b"
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JUDGE_CFG_PATH = REPO_ROOT / "eval" / "configs" / "judge_qwen.yaml"
RESULTS_DIR    = REPO_ROOT / "embedllm_eval" / "results"


# ---------------------------------------------------------------------------
# Helpers copied verbatim from run_one_job.py so behaviour is identical
# ---------------------------------------------------------------------------

def load_judge_cfg() -> dict:
    return yaml.safe_load(JUDGE_CFG_PATH.read_text())["judge"]


def resolve_local_path(cfg: dict) -> dict:
    """Prefer local_path over hf_repo when the directory exists on disk."""
    import os
    local = cfg.get("local_path", "")
    if local and os.path.isdir(local):
        cfg = dict(cfg)
        cfg["hf_repo"] = local
    return cfg


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def run_judge(
    records: list[dict],
    responses: list[str],
    judge_cfg: dict,
    backend_name: str,
    device: str,
    batch_size: int,
) -> list[int]:
    """Return per-question scores (1 = correct, 0 = incorrect)."""
    from eval.inference import make_backend as _make_backend, timed_generate
    from eval.prompts.judge_qwen import build as build_judge_prompt, parse_correct

    judge_model_cfg = {
        "hf_repo":              judge_cfg["hf_repo"],
        "local_path":           judge_cfg.get("local_path", ""),
        "dtype":                judge_cfg.get("dtype", "bfloat16"),
        "max_model_len":        judge_cfg.get("max_model_len", 8192),
        "tensor_parallel_size": judge_cfg.get("tensor_parallel_size", 1),
    }

    backend = _make_backend(resolve_local_path(judge_model_cfg), backend=backend_name, device=device)
    backend.gen_kwargs.max_new_tokens = judge_cfg.get("max_new_tokens", 200)
    backend.gen_kwargs.temperature = 0.0

    judge_prompts = []
    for rec, resp in zip(records, responses):
        gold = rec["gold_answer"]
        if isinstance(gold, list):
            gold_str = (
                ", ".join(gold) if len(gold) == 1
                else "Any one of: " + ", ".join(gold)
            )
        else:
            gold_str = str(gold)
        judge_prompts.append(
            build_judge_prompt(
                dataset=rec["category"],
                question=rec["prompt"],
                response=resp,
                gold=gold_str,
            )
        )

    scores: list[int] = []
    for i in range(0, len(judge_prompts), batch_size):
        batch = judge_prompts[i : i + batch_size]
        outs, _ = timed_generate(backend, batch)
        for raw in outs:
            correct_bool, _ = parse_correct(raw)
            scores.append(1 if correct_bool else 0)
        done = min(i + batch_size, len(judge_prompts))
        print(f"  judge: {done}/{len(judge_prompts)}", flush=True)

    backend.close()
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=f"Re-judge existing {MODEL_ID} results with the Qwen2.5-32B judge."
    )
    ap.add_argument("--category",         required=True,
                    help="Category name, e.g. mmlu_marketing")
    ap.add_argument("--backend",          choices=["auto", "hf", "vllm"], default="auto")
    ap.add_argument("--device",           default="cuda")
    ap.add_argument("--judge-batch-size", type=int, default=1,
                    help="Batch size for the judge (keep small to avoid OOM)")
    args = ap.parse_args()

    result_path = RESULTS_DIR / MODEL_ID / f"{args.category}.json"
    if not result_path.exists():
        print(f"ERROR: result file not found: {result_path}", file=sys.stderr)
        sys.exit(1)

    t_start = time.monotonic()
    print(f"=== rejudge: model={MODEL_ID}  category={args.category} ===", flush=True)

    records: list[dict] = json.loads(result_path.read_text())
    print(f"  loaded {len(records)} records from {result_path.relative_to(REPO_ROOT)}")

    judge_cfg = load_judge_cfg()

    # Strip <think>...</think> from model_answer — identical to run_one_job.py
    responses_for_judge = [strip_think_blocks(r["model_answer"]) for r in records]
    n_stripped = sum(1 for r, s in zip(records, responses_for_judge)
                     if r["model_answer"] != s)
    if n_stripped:
        print(f"  stripped think blocks from {n_stripped}/{len(records)} responses")

    print(f"\n[Judging] with {judge_cfg['id']} ...", flush=True)
    scores = run_judge(
        records, responses_for_judge,
        judge_cfg=judge_cfg,
        backend_name=args.backend,
        device=args.device,
        batch_size=args.judge_batch_size,
    )
    print("  judge model unloaded", flush=True)

    # Update scores in-place and save
    for rec, score in zip(records, scores):
        rec["score"] = score

    result_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    n_correct = sum(r["score"] for r in records)
    elapsed   = time.monotonic() - t_start
    print(f"\n  accuracy: {n_correct}/{len(records)} = {n_correct / len(records):.3f}")
    print(f"  elapsed:  {elapsed:.1f}s")
    print(f"  saved  → {result_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
