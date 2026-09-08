#!/usr/bin/env python3
"""
Run the circle-test judge phase using Claude Haiku via AWS Bedrock.
Replaces the local Qwen2.5-32B judge in circle_test.py --phase judge.

Reads all  data/responses/{model_id}.json  files, judges each response,
and writes  data/circle_test_{model_id}.json  (same format as the SLURM
judge job), then prints a summary.

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1   # default

    python modelsat_baseline/bedrock_judge_circle.py
    python modelsat_baseline/bedrock_judge_circle.py --workers 10 --dry-run

Credentials: any standard boto3 auth method works (env vars, ~/.aws/credentials,
instance profile, etc.).  Claude Haiku must be enabled for your Bedrock account.
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

MODELSAT_DIR  = Path(__file__).resolve().parent
RESPONSES_DIR = MODELSAT_DIR / "data" / "responses"
OUT_DIR       = MODELSAT_DIR / "data"

BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ---------------------------------------------------------------------------
# Judge prompt  (same semantics as eval/prompts/judge_qwen.py)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict, accurate evaluator for multiple-choice question-answering tasks. "
    "Given a question with choices, a model's answer, and the correct reference answer, "
    "decide if the model's answer is correct.\n\n"
    "Rules:\n"
    "- The model answer must identify the same choice as the reference answer.\n"
    "- A bare letter (e.g. 'B') matches a full choice label (e.g. 'B) Paris') or "
    "the choice text itself.\n"
    "- Ignore formatting differences (spaces, hyphens, case, bold markers).\n"
    "- If the model gives multiple candidates without committing, it is INCORRECT.\n"
    "- If the model refuses or says it cannot answer, it is INCORRECT.\n\n"
    "Respond ONLY with a single JSON object on one line:\n"
    "{\"correct\": true} or {\"correct\": false}\n"
    "Do not include explanations, code fences, or any other text."
)


def _build_judge_prompt(question: str, response: str, gold: str) -> str:
    return (
        f"{_JUDGE_SYSTEM}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"MODEL ANSWER:\n{response}\n\n"
        f"REFERENCE ANSWER:\n{gold}\n\n"
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
    """Call Claude via Bedrock; return True/False/None."""
    # Null responses are always wrong
    if not response:
        return False

    prompt = _build_judge_prompt(
        question=question[-4000:],
        response=response[-500:],
        gold=gold,
    )
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
                raise ValueError(f"Unparseable judge output: {text!r}")
            return result
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                tqdm.write(f"  [WARN] judge failed after {retries} attempts: {exc}")
                return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Judge circle-test responses via Claude on Bedrock.")
    ap.add_argument("--workers",  type=int, default=20,
                    help="Concurrent Bedrock threads (default: 20)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print first 3 prompts without calling Bedrock")
    args = ap.parse_args()

    response_files = sorted(RESPONSES_DIR.glob("*.json"))
    if not response_files:
        raise FileNotFoundError(f"No response files in {RESPONSES_DIR}. Run inference first.")

    # Build a flat work list: (model_id, prompt_id, question_prompt, response, gold)
    work: list[tuple[str, int, str, str | None, str]] = []
    uninformative_by_model: dict[str, list] = {}

    for rf in response_files:
        data = json.loads(rf.read_text())
        model_id = data["model"]
        uninformative_by_model[model_id] = data.get("uninformative", [])
        for q, r in zip(data["questions"], data["responses"]):
            work.append((model_id, q["prompt_id"], q["prompt"], r, q["gold"]))

    print(f"Loaded {len(work)} responses from {len(response_files)} models.")

    if args.dry_run:
        print("DRY RUN — first 3 judge prompts:")
        for model_id, pid, prompt, resp, gold in work[:3]:
            print(f"\n  model={model_id}  pid={pid}  gold={gold!r}  response={resp!r}")
            print(_build_judge_prompt(prompt[-400:], (resp or "")[-200:], gold))
        return

    # Judge in parallel
    results: dict[tuple[str, int], bool | None] = {}

    def _task(item):
        model_id, pid, prompt, resp, gold = item
        correct = _judge_one(prompt, resp, gold)
        return model_id, pid, correct

    print(f"\nJudging {len(work)} responses with {args.workers} workers...", flush=True)
    bar = tqdm(total=len(work), unit="resp")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, item): item for item in work}
        for fut in as_completed(futures):
            try:
                model_id, pid, correct = fut.result()
                results[(model_id, pid)] = correct
            except Exception as exc:
                item = futures[fut]
                tqdm.write(f"  ERROR [{item[0]} pid={item[1]}]: {exc}")
            bar.update(1)
    bar.close()

    # Group failures by model
    failures_by_model: dict[str, list[int]] = defaultdict(list)
    parse_failures = 0
    for (model_id, pid), correct in results.items():
        if correct is None:
            parse_failures += 1
        if not correct:
            failures_by_model[model_id].append(pid)

    # Write per-model output files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_id, uninformative in uninformative_by_model.items():
        out = OUT_DIR / f"circle_test_{model_id}.json"
        out.write_text(json.dumps({
            "model":         model_id,
            "failures":      sorted(failures_by_model.get(model_id, [])),
            "uninformative": sorted(uninformative),
        }, indent=2))
        n_fail = len(failures_by_model.get(model_id, []))
        n_total = sum(1 for (m, _) in results if m == model_id)
        print(f"  {model_id}: {n_fail}/{n_total} failed  → {out.name}")

    total_fail = sum(len(v) for v in failures_by_model.values())
    print(f"\nDone. {total_fail}/{len(work)} failed circle test. "
          f"({parse_failures} unparseable judge outputs counted as failures)")
    print("Run merge_circle_test.py to combine into circle_test_failures.json")


if __name__ == "__main__":
    main()
