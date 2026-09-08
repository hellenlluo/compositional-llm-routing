#!/usr/bin/env python3
"""
Judge LogicBench responses using Claude Haiku via AWS Bedrock.

For models that produce verbose reasoning (e.g. qwen3-4b-thinking-2507), the
simple startswith() judge in build_perf_vectors.py gives wrong accuracy.
This script asks Claude to determine correctness and writes a sidecar file
  results_judged.json
next to each results.json.  build_perf_vectors.py prefers results_judged.json
when present.

Usage:
    # Set Bedrock credentials first (any boto3 auth method works)
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1

    # Judge a specific model (default: qwen3-4b-thinking-2507)
    python perf_router/judge_logicbench.py
    python perf_router/judge_logicbench.py --model qwen3-4b-thinking-2507
    python perf_router/judge_logicbench.py --model medgemma-4b-it
    python perf_router/judge_logicbench.py --all-models
    python perf_router/judge_logicbench.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
LB_RESULTS   = SCRIPT_DIR / "results" / "logicbench"

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
    "- Ignore reasoning steps, preamble, and hedging — only the final committed "
    "answer matters.\n"
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
    prompt = _build_prompt(question, response, gold)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": prompt}],
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
# Collect work items from a model's result files
# ---------------------------------------------------------------------------

def _build_question(sample: dict, qa: dict, task_type: str) -> str:
    """Reconstruct the question text seen by the model."""
    context  = sample.get("context", "")
    question = qa.get("question", "")
    if task_type == "MCQA":
        options = qa.get("options", [])
        opts_block = "\n".join(options) if options else ""
        return f"{context}\n\n{question}\n{opts_block}".strip()
    return f"{context}\n\n{question}".strip()


WorkItem = tuple[Path, str, str, str, str]  # (results_path, sample_id, qa_id, question, gold)


def collect_work(model_dir: Path) -> list[WorkItem]:
    """Walk all results.json files under model_dir and collect (path, ids, q, gold)."""
    work: list[WorkItem] = []
    for results_path in sorted(model_dir.rglob("results.json")):
        try:
            data = json.loads(results_path.read_text())
        except Exception:
            continue
        task_type = data.get("task_type", "BQA")
        for sample in data.get("samples", []):
            sid = str(sample.get("id", ""))
            for qa in sample.get("qa_pairs", []):
                question = _build_question(sample, qa, task_type)
                gold     = qa.get("answer", "")
                qa_id    = qa.get("unique_id", "")
                work.append((results_path, sid, qa_id, question, gold))
    return work


# ---------------------------------------------------------------------------
# Apply judgments back to results files and write results_judged.json
# ---------------------------------------------------------------------------

def write_judged_files(
    model_dir: Path,
    judgments: dict[str, bool | None],  # qa_id → correct
) -> int:
    """Write results_judged.json sidecar files; return number of files written."""
    # Gather all unique results.json paths
    results_paths = sorted(model_dir.rglob("results.json"))
    n_written = 0
    for rp in results_paths:
        try:
            data = json.loads(rp.read_text())
        except Exception:
            continue
        task_type = data.get("task_type", "BQA")
        modified = False
        for sample in data.get("samples", []):
            for qa in sample.get("qa_pairs", []):
                qa_id = qa.get("unique_id", "")
                if qa_id in judgments and judgments[qa_id] is not None:
                    qa["judged_correct"] = judgments[qa_id]
                    modified = True
        if modified:
            out_path = rp.parent / "results_judged.json"
            out_path.write_text(json.dumps(data, indent=2))
            n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="qwen3-4b-thinking-2507",
                   help="Model ID to judge (directory name under results/logicbench/)")
    p.add_argument("--all-models", action="store_true",
                   help="Judge all models with LogicBench results")
    p.add_argument("--workers", type=int, default=20,
                   help="Concurrent Bedrock threads (default: 20)")
    p.add_argument("--force", action="store_true",
                   help="Re-judge even if results_judged.json already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print first 3 prompts without calling Bedrock")
    return p.parse_args()


def judge_model(model_id: str, workers: int, force: bool, dry_run: bool) -> None:
    model_dir = LB_RESULTS / model_id
    if not model_dir.is_dir():
        print(f"  [skip] {model_id}: no LogicBench results directory")
        return

    # Check if already judged (unless --force)
    existing_judged = list(model_dir.rglob("results_judged.json"))
    if existing_judged and not force:
        print(f"  [skip] {model_id}: {len(existing_judged)} judged files already exist "
              f"(use --force to re-judge)")
        return

    print(f"\nCollecting work items for {model_id}...")
    work = collect_work(model_dir)
    print(f"  {len(work)} QA pairs to judge")

    if dry_run:
        print("  DRY RUN — first 3 judge prompts:")
        for rp, sid, qa_id, question, gold in work[:3]:
            print(f"\n    qa_id={qa_id}  gold={gold!r}")
            print("    Prompt:")
            print(_build_prompt(question, "[MODEL RESPONSE PLACEHOLDER]", gold)[:400])
        return

    # Judge in parallel
    judgments: dict[str, bool | None] = {}

    def _task(item):
        rp, sid, qa_id, question, gold = item
        # Load the actual response from the file (we only stored question above)
        data = json.loads(rp.read_text())
        resp = ""
        for sample in data.get("samples", []):
            for qa in sample.get("qa_pairs", []):
                if qa.get("unique_id") == qa_id:
                    resp = qa.get("response", "")
                    break
        correct = _judge_one(question, resp, gold)
        return qa_id, correct

    print(f"  Judging with {workers} workers...")
    bar = tqdm(total=len(work), unit="qa")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, item): item for item in work}
        for fut in as_completed(futures):
            try:
                qa_id, correct = fut.result()
                judgments[qa_id] = correct
            except Exception as exc:
                tqdm.write(f"  ERROR: {exc}")
            bar.update(1)
    bar.close()

    n_judged  = sum(1 for v in judgments.values() if v is not None)
    n_correct = sum(1 for v in judgments.values() if v is True)
    print(f"  Judged: {n_judged}/{len(judgments)}  Correct: {n_correct} "
          f"({100*n_correct/n_judged:.1f}%)" if n_judged else "  No valid judgments")

    n_written = write_judged_files(model_dir, judgments)
    print(f"  Wrote {n_written} results_judged.json files under {model_dir}")


def main():
    args = parse_args()

    if args.all_models:
        if not LB_RESULTS.is_dir():
            print(f"ERROR: LogicBench results dir not found: {LB_RESULTS}")
            return
        models = sorted(d.name for d in LB_RESULTS.iterdir() if d.is_dir())
    else:
        models = [args.model]

    print(f"Judging {len(models)} model(s) via Claude Haiku on Bedrock...")
    for model_id in models:
        judge_model(model_id, args.workers, args.force, args.dry_run)

    print("\nDone. Run build_accuracy_vectors.py to regenerate accuracy_vectors.json.")


if __name__ == "__main__":
    main()
