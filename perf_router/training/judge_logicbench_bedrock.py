#!/usr/bin/env python3
"""
Judge LogicBench responses using Claude Haiku via AWS Bedrock.

For models that produce verbose reasoning (e.g. qwen3-4b-thinking-2507), a
simple startswith() judge gives wrong accuracy.  This script asks Claude to
determine correctness and writes a single output file:

    perf_router/lb_judged_accuracies.json

Format:
  {
    "qwen3-4b-thinking-2507": {
      "lb_BQA_first_order_logic_modus_ponens": 0.82,
      "lb_BQA_propositional_logic_modus_tollens": 0.65,
      ...
    },
    ...
  }

build_accuracy_vectors.py prefers these judged values over its regex judge
when lb_judged_accuracies.json is present.

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1

    python perf_router/judge_logicbench_bedrock.py
    python perf_router/judge_logicbench_bedrock.py --model qwen3-4b-thinking-2507
    python perf_router/judge_logicbench_bedrock.py --all-models
    python perf_router/judge_logicbench_bedrock.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR   = SCRIPT_DIR.parent                       # perf_router/
LB_RESULTS = EVAL_DIR / "results" / "logicbench"
OUT_PATH   = EVAL_DIR / "results" / "qwen3" / "lb_judged_accuracies.json"

BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a strict evaluator for logic question-answering tasks. "
    "Given a question, a model's response (which may contain verbose reasoning), "
    "and the expected correct answer, decide if the model's final answer is correct.\n\n"
    "Rules:\n"
    "- For yes/no questions: the model must ultimately commit to the correct choice.\n"
    "- For multiple-choice (A/B/C/D): the model must select the correct letter or "
    "its corresponding text.\n"
    "- Ignore reasoning steps and preamble — only the final committed answer matters.\n"
    "- If the response is truncated and never reaches a conclusion, it is INCORRECT.\n"
    "- If the model is ambiguous or commits to the wrong answer, it is INCORRECT.\n\n"
    "Respond ONLY with a single JSON object on one line:\n"
    '{"correct": true} or {"correct": false}\n'
    "No explanations, no code fences, no other text."
)


def _build_prompt(question: str, response: str, gold: str) -> str:
    return (
        f"{_SYSTEM}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED ANSWER:\n{gold}\n\n"
        f"MODEL RESPONSE:\n{response[-1500:]}\n\n"
        f"Output the JSON object now:"
    )


_JSON_RE = re.compile(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', re.I)


def _parse_correct(raw: str) -> bool | None:
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and "correct" in obj:
            return bool(obj["correct"])
    except Exception:
        pass
    m = _JSON_RE.search(raw)
    if m:
        return m.group(1).lower() == "true"
    return None


# ---------------------------------------------------------------------------
# Bedrock client (per-thread)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_client():
    if not hasattr(_thread_local, "client"):
        import boto3
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        _thread_local.client = boto3.client("bedrock-runtime", region_name=region)
    return _thread_local.client


def _judge_one(question: str, response: str, gold: str, retries: int = 4) -> bool | None:
    if not response:
        return False
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": _build_prompt(question, response, gold)}],
    })
    backoff = 2.0
    for attempt in range(retries):
        try:
            resp = _get_client().invoke_model(
                modelId=BEDROCK_MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            text = json.loads(resp["body"].read())["content"][0]["text"].strip()
            result = _parse_correct(text)
            if result is None:
                raise ValueError(f"Unparseable: {text!r}")
            return result
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                tqdm.write(f"  [WARN] judge failed: {exc}")
                return None
    return None


# ---------------------------------------------------------------------------
# Build question text from a sample + qa_pair
# ---------------------------------------------------------------------------

def _build_question_bqa(sample: dict, qa: dict) -> str:
    context  = sample.get("context", "")
    question = qa.get("question", "")
    return f"{context}\n\n{question}".strip()


def _build_question_mcqa(sample: dict) -> str:
    """Build question text for MCQA flat-structure samples."""
    context  = sample.get("context", "")
    question = sample.get("question", "")
    choices  = sample.get("choices", {})
    if isinstance(choices, str):
        # choices may be stored as a repr string; try to eval safely
        try:
            import ast
            choices = ast.literal_eval(choices)
        except Exception:
            choices = {}
    opts_block = "\n".join(
        f"{k}: {v}" for k, v in sorted(choices.items())
    ) if isinstance(choices, dict) else ""
    return f"{context}\n\n{question}\n{opts_block}".strip()


# ---------------------------------------------------------------------------
# Judge one model — returns {feature_name: accuracy}
# ---------------------------------------------------------------------------

def judge_model(model_id: str, workers: int, dry_run: bool) -> dict[str, float]:
    model_dir = LB_RESULTS / model_id
    if not model_dir.is_dir():
        print(f"  [skip] {model_id}: no LogicBench results directory")
        return {}

    # Collect all (results_path, feature_name, sample, qa_pair) tuples
    WorkItem = tuple  # (feature, question, response, gold)
    work_by_feature: dict[str, list[WorkItem]] = defaultdict(list)

    for results_path in sorted(model_dir.rglob("results.json")):
        try:
            data = json.loads(results_path.read_text())
        except Exception:
            continue
        task_type  = data.get("task_type", "BQA")
        logic_type = data.get("logic_type", "")
        # Coarse feature: aggregate at (task_type, logic_type) level
        feature    = f"lb_{task_type}_{logic_type}"

        for sample in data.get("samples", []):
            if task_type == "BQA":
                for qa in sample.get("qa_pairs", []):
                    question = _build_question_bqa(sample, qa)
                    response = qa.get("response", "")
                    gold     = qa.get("answer", "")
                    if question and gold:
                        work_by_feature[feature].append((question, response, gold))
            else:
                # MCQA: flat structure — one answer/response per sample
                question = _build_question_mcqa(sample)
                response = sample.get("response", "")
                gold     = sample.get("answer", "")
                if question and gold:
                    work_by_feature[feature].append((question, response, gold))

    total_items = sum(len(v) for v in work_by_feature.values())
    print(f"  {len(work_by_feature)} features, {total_items} QA pairs")

    if dry_run:
        for feat, items in list(work_by_feature.items())[:2]:
            q, r, g = items[0]
            print(f"\n  Feature: {feat}  gold={g!r}")
            print(f"  Prompt preview:\n{_build_prompt(q, r, g)[:300]}\n  ...")
        return {}

    # Flatten to a list of (feature, question, response, gold) for parallel dispatch
    flat_work = [
        (feat, q, r, g)
        for feat, items in work_by_feature.items()
        for q, r, g in items
    ]

    results: list[tuple[str, bool | None]] = []  # (feature, correct)

    def _task(item):
        feat, q, r, g = item
        return feat, _judge_one(q, r, g)

    bar = tqdm(total=len(flat_work), unit="qa", desc=model_id)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, item): item for item in flat_work}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                tqdm.write(f"  ERROR: {exc}")
            bar.update(1)
    bar.close()

    # Aggregate: {feature: (n_correct, n_total)}
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for feat, correct in results:
        if correct is not None:
            agg[feat][0] += int(correct)
            agg[feat][1] += 1

    accuracies = {
        feat: counts[0] / counts[1]
        for feat, counts in agg.items()
        if counts[1] > 0
    }

    n_correct = sum(c[0] for c in agg.values())
    n_total   = sum(c[1] for c in agg.values())
    print(f"  Overall: {n_correct}/{n_total} correct "
          f"({100*n_correct/n_total:.1f}%)" if n_total else "  No results")

    return accuracies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",       type=str, default="qwen3-4b-thinking-2507",
                   help="Model to judge (default: qwen3-4b-thinking-2507)")
    p.add_argument("--all-models",  action="store_true",
                   help="Judge all models with LogicBench results")
    p.add_argument("--workers",     type=int, default=20,
                   help="Concurrent Bedrock threads (default: 20)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print first few prompts without calling Bedrock")
    p.add_argument("--out",         type=str, default=str(OUT_PATH),
                   help=f"Output JSON path (default: {OUT_PATH})")
    return p.parse_args()


def main():
    args = parse_args()

    if args.all_models:
        if not LB_RESULTS.is_dir():
            print(f"ERROR: LogicBench results dir not found: {LB_RESULTS}")
            return
        models = sorted(d.name for d in LB_RESULTS.iterdir() if d.is_dir())
    else:
        models = [args.model]

    print(f"Judging {len(models)} model(s) via Claude Haiku on Bedrock...\n")

    # Load existing output so we can merge/update
    out_path = Path(args.out)
    existing: dict[str, dict[str, float]] = {}
    if out_path.is_file():
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded existing results from {out_path} "
              f"({len(existing)} model(s) already judged)\n")

    for model_id in models:
        print(f"--- {model_id} ---")
        accuracies = judge_model(model_id, args.workers, args.dry_run)
        if accuracies and not args.dry_run:
            existing[model_id] = accuracies

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\nSaved judged accuracies to {out_path}")
        print("Re-run build_accuracy_vectors.py to update accuracy_vectors.json")


if __name__ == "__main__":
    main()
