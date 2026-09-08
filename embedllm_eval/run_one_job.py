"""Evaluate ONE model on ONE category, then judge every answer with Qwen2.5-32B.

Each SLURM job calls this script once:
  python embedllm_eval/run_one_job.py --model-id qwen3-4b-thinking-2507 --category gpqa_main_n_shot

Output:
  embedllm_eval/results/<model_id>/<category>.json

Each entry in the output JSON:
  {
    "model_name":   str,   # model id from models.yaml
    "category":     str,
    "family":       str,
    "prompt_id":    int,
    "prompt":       str,
    "gold_answer":  str | list,
    "model_answer": str,   # raw response from the candidate model
    "score":        int    # 1 = correct, 0 = incorrect (from Qwen2.5-32B judge)
  }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROMPTS_FILE  = REPO_ROOT / "data_preprocessing" / "embedllm-sampled" / "prompts_subset.jsonl"
MODELS_CFG    = REPO_ROOT / "eval" / "configs" / "models.yaml"
JUDGE_CFG     = REPO_ROOT / "eval" / "configs" / "judge_qwen.yaml"
RESULTS_DIR   = REPO_ROOT / "embedllm_eval" / "results"

# Injected as a system message for every candidate model.
# Thinking models (Qwen3, DeepSeek-R1) will still reason internally inside
# <think>...</think> — the instruction only shapes the final answer text.
SYSTEM_PROMPT = (
    "You are a concise, accurate assistant. "
    "For multiple-choice questions, respond with only the letter of the correct "
    "choice (e.g. A, B, C, or D) and nothing else. "
    "For all other questions, give only your final answer — a single word, number, "
    "or short phrase. Do not explain your reasoning."
)

MODEL_IDS = [
    "qwen3-4b-thinking-2507",
    "deepseek-r1-distill-llama-8b",
    "medgemma-4b-it",
    "llama-3.1-8b-instruct",
    "qwen3-30b-a3b",
    "phi-4-mini-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen1.5-0.5b-chat",
    "mathstral-7b-v0.1",
    "llama-3.1-nemotron-nano-8b-v1",
]


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_models_cfg() -> dict:
    return yaml.safe_load(MODELS_CFG.read_text())


def get_model_cfg(models_cfg: dict, model_id: str) -> dict:
    for m in models_cfg["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"unknown model id '{model_id}'. Available: {[m['id'] for m in models_cfg['models']]}")


def load_judge_cfg() -> dict:
    return yaml.safe_load(JUDGE_CFG.read_text())["judge"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_category_questions(category: str) -> list[dict]:
    questions = []
    with PROMPTS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["category"] == category:
                questions.append(entry)
    if not questions:
        raise ValueError(
            f"No questions found for category '{category}'. "
            "Run: python embedllm_eval/list_categories.py"
        )
    questions.sort(key=lambda q: q["prompt_id"])
    return questions


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def resolve_model_path(model_cfg: dict) -> dict:
    """Return a copy of model_cfg with hf_repo replaced by local_path if it exists on disk."""
    import os
    local = model_cfg.get("local_path", "")
    if local and os.path.isdir(local):
        cfg = dict(model_cfg)
        cfg["hf_repo"] = local
        return cfg
    return model_cfg


def make_backend(model_cfg: dict, backend: str, device: str):
    import types
    from eval.inference import make_backend as _make_backend

    b = _make_backend(resolve_model_path(model_cfg), backend=backend, device=device)

    # Patch _format to inject SYSTEM_PROMPT as a system message via the chat
    # template. Falls back to a plain-text prefix if no chat template exists.
    def _format_with_system(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        return f"{SYSTEM_PROMPT}\n\n{prompt}"

    b._format = types.MethodType(_format_with_system, b)
    return b


def question_for_judge(prompt: str, max_chars: int = 4000) -> str:
    """Return the prompt unchanged unless it exceeds max_chars.

    In practice, all prompts in prompts_subset.jsonl are well under 2000 chars,
    so this is a safety guard only — it does not truncate meaningful content.
    """
    return prompt[-max_chars:] if len(prompt) > max_chars else prompt


def truncate_response_for_judge(response: str, max_chars: int = 500) -> str:
    """Keep only the tail of a (possibly very long) model response.

    DeepSeek-R1-Distill sometimes ignores the system prompt's 'answer only'
    instruction and generates thousands of tokens of plain-text reasoning with
    no <think> tags for strip_think_blocks to remove.  The judge only needs
    to see the final answer, which always appears at the end of such chains.
    500 chars (~125 tokens) is more than enough to capture any final-answer
    sentence while keeping the judge prompt safely under 8 192 tokens.
    """
    return response[-max_chars:] if len(response) > max_chars else response


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    Thinking models (Qwen3, DeepSeek-R1) wrap their chain-of-thought in these
    tags. We strip them so the judge only sees the final answer.

    Three cases handled:
    1. Standard: <think>...</think> paired tags — remove the block.
    2. Unclosed: <think> opened but model ran out of tokens before </think> —
       discard everything from the opening tag to end of string.
    3. Qwen3-Thinking format: the chat template injects '<think>' as the
       generation prefix (part of the prompt, not the output), so the raw
       output begins with reasoning text and ends with '</think>\\n\\n<answer>'.
       Strip everything up to and including the last </think>.
    """
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    # Case 3: orphan </think> with no matching opening tag (Qwen3 format).
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    return cleaned.strip()


def run_inference(backend, questions: list[dict], batch_size: int, max_new_tokens: int) -> list[str]:
    """Return raw model responses in the same order as questions."""
    from eval.inference import timed_generate

    backend.gen_kwargs.max_new_tokens = max_new_tokens
    backend.gen_kwargs.temperature = 0.0

    responses: list[str] = []
    prompts = [q["prompt"] for q in questions]

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outs, _ = timed_generate(backend, batch)
        responses.extend(outs)
        done = min(i + batch_size, len(prompts))
        print(f"  inference: {done}/{len(prompts)}", flush=True)

    return responses


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def run_judge(
    questions: list[dict],
    responses: list[str],
    judge_cfg: dict,
    backend_name: str,
    device: str,
    batch_size: int,
) -> list[int]:
    """Return per-question scores (1/0). None responses from the judge are treated as 0."""
    from eval.inference import make_backend as _make_backend, timed_generate
    from eval.prompts.judge_qwen import build as build_judge_prompt, parse_correct

    judge_model_cfg = {
        "hf_repo":              judge_cfg["hf_repo"],
        "local_path":           judge_cfg.get("local_path", ""),
        "dtype":                judge_cfg.get("dtype", "bfloat16"),
        "max_model_len":        judge_cfg.get("max_model_len", 8192),
        "tensor_parallel_size": judge_cfg.get("tensor_parallel_size", 1),
    }

    backend = _make_backend(resolve_model_path(judge_model_cfg), backend=backend_name, device=device)
    backend.gen_kwargs.max_new_tokens = judge_cfg.get("max_new_tokens", 200)
    backend.gen_kwargs.temperature = 0.0

    # Strip <think> blocks first, then truncate whatever remains so that
    # long plain-text reasoning chains (DeepSeek-R1 without think tags)
    # don't overflow the judge model's 8192-token context window.
    judge_prompts = []
    for q, resp in zip(questions, responses):
        gold = q["gold_answer"]
        if isinstance(gold, list):
            gold_str = (
                ", ".join(gold) if len(gold) == 1
                else "Any one of: " + ", ".join(gold)
            )
        else:
            gold_str = str(gold)
        judge_prompts.append(
            build_judge_prompt(
                dataset=q["category"],
                question=question_for_judge(q["prompt"]),
                response=truncate_response_for_judge(resp),
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
# Output
# ---------------------------------------------------------------------------

def save_results(model_id: str, category: str, records: list[dict]) -> Path:
    out_dir = RESULTS_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{category}.json"
    sorted_records = sorted(records, key=lambda r: r["prompt_id"])
    out_path.write_text(json.dumps(sorted_records, indent=2, ensure_ascii=False))
    print(f"  saved → {out_path.relative_to(REPO_ROOT)}  ({len(records)} entries)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate one model on one category, judge with Qwen2.5-32B."
    )
    ap.add_argument("--model-id",       required=True, choices=MODEL_IDS)
    ap.add_argument("--category",       required=True)
    ap.add_argument("--backend",        choices=["auto", "hf", "vllm"], default="auto")
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--batch-size",     type=int, default=8,
                    help="Batch size for candidate model inference")
    ap.add_argument("--judge-batch-size", type=int, default=1,
                    help="Batch size for Qwen2.5-32B judge (keep low to avoid OOM)")
    args = ap.parse_args()

    t_start = time.monotonic()
    print(f"=== run_one_job: model={args.model_id}  category={args.category} ===", flush=True)

    questions = load_category_questions(args.category)
    print(f"  questions: {len(questions)}  family: {questions[0]['family']}")

    models_cfg = load_models_cfg()
    model_cfg  = get_model_cfg(models_cfg, args.model_id)
    judge_cfg  = load_judge_cfg()

    # Use the model's full context window for generation — reserve 2048 tokens
    # for the input prompt, give the rest to the output (important for thinking
    # models whose <think> blocks can be thousands of tokens).
    max_new_tokens = model_cfg.get("max_model_len", 32768) - 2048
    print(f"  max_new_tokens: {max_new_tokens} (derived from model max_model_len)")

    # ---- Phase 1: candidate model inference --------------------------------
    print(f"\n[Phase 1] Running inference with {args.model_id} ...", flush=True)
    backend = make_backend(model_cfg, backend=args.backend, device=args.device)
    try:
        responses = run_inference(backend, questions, args.batch_size, max_new_tokens)
    finally:
        backend.close()
        print("  candidate model unloaded", flush=True)

    # Strip <think>...</think> blocks before judging — thinking models
    # (Qwen3, DeepSeek-R1) wrap reasoning in these tags; the judge should
    # only see the final answer that follows.
    responses_for_judge = [strip_think_blocks(r) for r in responses]
    n_stripped = sum(1 for r, s in zip(responses, responses_for_judge) if r != s)
    if n_stripped:
        print(f"  stripped think blocks from {n_stripped}/{len(responses)} responses")

    # ---- Phase 2: Qwen2.5-32B judge ----------------------------------------
    print(f"\n[Phase 2] Judging with {judge_cfg['id']} ...", flush=True)
    scores = run_judge(
        questions, responses_for_judge,
        judge_cfg=judge_cfg,
        backend_name=args.backend,
        device=args.device,
        batch_size=args.judge_batch_size,
    )
    print("  judge model unloaded", flush=True)

    # ---- Assemble and save --------------------------------------------------
    records = [
        {
            "model_name":   args.model_id,
            "category":     q["category"],
            "family":       q["family"],
            "prompt_id":    q["prompt_id"],
            "prompt":       q["prompt"],
            "gold_answer":  q["gold_answer"],
            "model_answer": stripped,   # post-think-strip answer sent to judge
            "raw_response": resp,       # full original output (includes <think> if any)
            "score":        score,
        }
        for q, resp, stripped, score in zip(questions, responses, responses_for_judge, scores)
    ]

    out_path = save_results(args.model_id, args.category, records)

    n_correct = sum(r["score"] for r in records)
    elapsed   = time.monotonic() - t_start
    print(f"\n  accuracy: {n_correct}/{len(records)} = {n_correct / len(records):.3f}")
    print(f"  elapsed:  {elapsed:.1f}s")
    print(f"  output:   {out_path}")


if __name__ == "__main__":
    main()
