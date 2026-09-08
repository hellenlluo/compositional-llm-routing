#!/usr/bin/env python3
"""
Run circle-test inference for one model via HuggingFace Inference API.
Produces the same output format as circle_test.py --phase inference.

Usage:
    export HF_TOKEN=hf_...
    python modelsat_baseline/api_infer_circle.py --model deepseek-r1-distill-llama-8b

The API model string can be overridden with --api-model if the default
'deepseek-ai/DeepSeek-R1-Distill-Llama-8B:novita' doesn't work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODELSAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELSAT_DIR))

# ---------------------------------------------------------------------------
# Default API model IDs per circle-test model
# ---------------------------------------------------------------------------
API_MODEL_DEFAULTS = {
    "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B:nscale",
    "llama-3.1-8b-instruct":         "meta-llama/Llama-3.1-8B-Instruct:nscale",
    "llama-3.1-nemotron-nano-8b":    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1:nscale",
    "mathstral-7b":                   "mistralai/Mathstral-7B-v0.1:nscale",
    "medgemma-4b-it":                 "google/medgemma-4b-it:nscale",
    "mistral-7b-instruct-v0.3":      "mistralai/Mistral-7B-Instruct-v0.3:nscale",
    "phi-4-mini-instruct":            "microsoft/Phi-4-mini-instruct:nscale",
    "qwen1.5-0.5b-chat":             "Qwen/Qwen1.5-0.5B-Chat:nscale",
    "qwen3-30b-a3b":                  "Qwen/Qwen3-30B-A3B:nscale",
    "qwen3-4b-thinking-2507":         "Qwen/Qwen3-4B:nscale",
}

SYSTEM_PROMPT = (
    "You are a concise, accurate assistant. "
    "For multiple-choice questions, respond with only the letter of the correct "
    "choice (e.g. A, B, C, or D) and nothing else. "
    "Do not include any explanation or additional text."
)

RESPONSES_DIR = Path(__file__).resolve().parent / "data" / "responses"


def run_api_inference(questions: list[dict], api_model: str, max_tokens: int) -> list[str]:
    from openai import OpenAI

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN environment variable first.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )
    responses = []
    for i, q in enumerate(questions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": q["prompt"]},
        ]
        completion = client.chat.completions.create(
            model=api_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        msg = completion.choices[0].message
        # Thinking models (DeepSeek R1) put reasoning in reasoning_content and
        # the final answer in content. When content is present, use it directly.
        # If content is None (model ran out of tokens or thinking-only output),
        # fall back to extracting the last paragraph of reasoning_content.
        response = msg.content or None
        if not response:
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                lines = [l.strip() for l in rc.splitlines() if l.strip()]
                response = lines[-1] if lines else rc
            else:
                response = str(msg)
        # Strip <think>...</think> blocks (some models include them in content)
        if response and "</think>" in response:
            response = response.split("</think>", 1)[1].strip() or response
        responses.append(response)
        print(f"  [{i+1}/{len(questions)}] → {response!r}", flush=True)

    return responses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Circle-test model ID (e.g. deepseek-r1-distill-llama-8b)")
    p.add_argument("--api-model", default=None,
                   help="Override HF API model string (org/model:provider)")
    p.add_argument("--max-tokens", type=int, default=32768)
    args = p.parse_args()

    api_model = args.api_model or API_MODEL_DEFAULTS.get(args.model)
    if not api_model:
        raise ValueError(f"No default API model for '{args.model}'. Pass --api-model explicitly.")

    print(f"Model:     {args.model}")
    print(f"API model: {api_model}")

    # Load questions from the circle-test pipeline
    from circle_test import find_one_correct_questions  # type: ignore
    one_correct, uninformative = find_one_correct_questions()

    from collections import defaultdict
    by_model: dict[str, list[dict]] = defaultdict(list)
    for pid, info in one_correct.items():
        by_model[info["model"]].append({"prompt_id": pid, **info})

    questions = by_model.get(args.model, [])
    if not questions:
        print(f"No one-correct questions for '{args.model}'. Writing empty file.")
        RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        out = RESPONSES_DIR / f"{args.model}.json"
        out.write_text(json.dumps({
            "model": args.model, "uninformative": sorted(uninformative),
            "questions": [], "responses": [],
        }, indent=2))
        return

    print(f"\nRunning API inference on {len(questions)} questions...", flush=True)
    responses = run_api_inference(questions, api_model, args.max_tokens)

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    out = RESPONSES_DIR / f"{args.model}.json"
    out.write_text(json.dumps({
        "model":         args.model,
        "uninformative": sorted(uninformative),
        "questions":     questions,
        "responses":     responses,
    }, indent=2))
    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
