#!/usr/bin/env python3
"""
Circle test for questions where exactly one model got it right.

For each such question (with exactly A/B/C/D choices), rotates the answer
choices: A B C D → D A B C (i.e. new_A=old_D, new_B=old_A, new_C=old_B,
new_D=old_C), updates the gold answer accordingly, re-runs inference with
the model that originally got it right, and checks whether it still answers
correctly.

Output: modelsat_baseline/data/circle_test_failures.json
  List of prompt_ids where the model answered incorrectly on the rotated prompt,
  indicating a possible lucky guess on the original.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT    = Path(__file__).resolve().parent.parent
RESULTS_DIR  = REPO_ROOT / "embedllm-eval" / "results"
MODELS_CFG   = REPO_ROOT / "eval" / "configs" / "models.yaml"
JUDGE_CFG    = REPO_ROOT / "eval" / "configs" / "judge_qwen.yaml"
PAIRS_FILE    = Path(__file__).resolve().parent / "modelsat_data" / "training_pairs.jsonl"
OUT_FILE      = Path(__file__).resolve().parent / "data" / "circle_test_failures.json"
RESPONSES_DIR = Path(__file__).resolve().parent / "data" / "responses"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYSTEM_PROMPT = (
    "You are a concise, accurate assistant. "
    "For multiple-choice questions, respond with only the letter of the correct "
    "choice (e.g. A, B, C, or D) and nothing else. "
    "For all other questions, give only your final answer — a single word, number, "
    "or short phrase. Do not explain your reasoning."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_models_cfg() -> dict:
    return yaml.safe_load(MODELS_CFG.read_text())


def get_model_cfg(models_cfg: dict, model_id: str) -> dict:
    for m in models_cfg["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"Unknown model id '{model_id}'")


def load_judge_cfg() -> dict:
    return yaml.safe_load(JUDGE_CFG.read_text())["judge"]


def resolve_model_path(model_cfg: dict) -> dict:
    import os
    local = model_cfg.get("local_path", "")
    if local and os.path.isdir(local):
        cfg = dict(model_cfg)
        cfg["hf_repo"] = local
        return cfg
    return model_cfg


def strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    return cleaned.strip()


def make_backend(model_cfg: dict, backend: str, device: str):
    from eval.inference import make_backend as _make_backend

    b = _make_backend(resolve_model_path(model_cfg), backend=backend, device=device)

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


# ---------------------------------------------------------------------------
# Choice rotation  (works for any number of choices: 2, 3, 4, 5, …)
# ---------------------------------------------------------------------------

# Matches "Choices:\n(A) …\n(B) …\n…\nAnswer:" with any number of options.
_CHOICES_BLOCK = re.compile(
    r"(Choices:\n)((?:\([A-Z]\) .+\n)+)(Answer:)",
    re.MULTILINE,
)
_CHOICE_LINE = re.compile(r"^\(([A-Z])\) (.+)$")


def _parse_choices(choices_text: str) -> list[tuple[str, str]] | None:
    """Return [(letter, content), …] or None if any line is malformed."""
    result = []
    for line in choices_text.rstrip("\n").split("\n"):
        m = _CHOICE_LINE.match(line)
        if not m:
            return None
        result.append((m.group(1), m.group(2)))
    return result if len(result) >= 2 else None


def rotate_prompt(prompt: str) -> str | None:
    """
    Rotate choices one slot forward: last option's content moves to the first
    slot, every other option shifts one slot down.  Works for any N ≥ 2.
    Returns the modified prompt, or None if no valid choices block is found.
    """
    m = _CHOICES_BLOCK.search(prompt)
    if not m:
        return None
    parsed = _parse_choices(m.group(2))
    if parsed is None:
        return None
    letters  = [p[0] for p in parsed]
    contents = [p[1] for p in parsed]
    # Slot i gets old content i-1 (last wraps to first)
    rotated_contents = [contents[-1]] + contents[:-1]
    new_choices   = "".join(f"({l}) {c}\n" for l, c in zip(letters, rotated_contents))
    rotated_block = m.group(1) + new_choices + m.group(3)
    return prompt[: m.start()] + rotated_block + prompt[m.end():]


def rotate_gold(gold: str, prompt: str) -> str | None:
    """
    Map old gold label → new gold label after rotation.
    Old content at index i moves to slot i+1 (mod N), so the new gold is the
    letter that follows the old gold letter in the choices list.
    """
    m = _CHOICES_BLOCK.search(prompt)
    if not m:
        return None
    parsed = _parse_choices(m.group(2))
    if parsed is None:
        return None
    letters = [p[0] for p in parsed]
    gold    = gold.strip().upper()
    if gold not in letters:
        return None
    return letters[(letters.index(gold) + 1) % len(letters)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_one_correct_questions() -> tuple[dict[int, dict], list[int]]:
    """
    Returns:
      - one_correct: {prompt_id: {"model", "prompt", "gold", "original_gold"}}
        for questions where exactly one model got score=1 and the prompt has
        a valid multiple-choice choices block (any number of options).
      - uninformative: list of prompt_ids where no model or every model was
        correct (these questions carry no signal for distinguishing models).
    """
    by_prompt: dict[int, list[dict]] = defaultdict(list)
    with open(PAIRS_FILE) as f:
        for line in f:
            r = json.loads(line)
            by_prompt[r["prompt_id"]].append(r)

    n_models = len({r["model_name"] for records in by_prompt.values() for r in records})

    # Load gold answers from any model's all_results.json (shared across models)
    gold_map: dict[int, str] = {}
    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        path = model_dir / "all_results.json"
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        for rec in records:
            if rec["prompt_id"] not in gold_map and rec.get("gold_answer"):
                gold = rec["gold_answer"]
                if isinstance(gold, list):
                    gold = gold[0]
                gold_map[rec["prompt_id"]] = gold
        break  # all models share the same prompts/golds

    one_correct: dict[int, dict] = {}
    uninformative: list[int] = []

    for pid, records in by_prompt.items():
        n_correct = sum(1 for r in records if r["score"] == 1)

        if n_correct == 0 or n_correct == len(records):
            uninformative.append(pid)
            continue

        if n_correct != 1:
            continue

        correct = [r for r in records if r["score"] == 1]
        prompt = correct[0]["prompt"]
        gold = gold_map.get(pid)
        if gold is None:
            continue

        rotated_prompt = rotate_prompt(prompt)
        if rotated_prompt is None:
            continue

        rotated_gold = rotate_gold(gold, prompt)
        if rotated_gold is None:
            continue

        one_correct[pid] = {
            "model":          correct[0]["model_name"],
            "prompt":         rotated_prompt,
            "gold":           rotated_gold,
            "original_gold":  gold,
        }

    return one_correct, uninformative


# ---------------------------------------------------------------------------
# Inference + judge
# ---------------------------------------------------------------------------

def run_inference_for_model(
    model_id: str,
    questions: list[dict],
    backend_name: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    from eval.inference import timed_generate

    models_cfg = load_models_cfg()
    model_cfg  = get_model_cfg(models_cfg, model_id)
    backend    = make_backend(model_cfg, backend=backend_name, device=device)
    backend.gen_kwargs.max_new_tokens = max_new_tokens
    backend.gen_kwargs.temperature    = 0.0

    responses: list[str] = []
    prompts = [q["prompt"] for q in questions]
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        outs, _ = timed_generate(backend, batch)
        responses.extend(outs)
        print(f"  [{model_id}] inference {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)

    backend.close()
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    return responses


def run_judge(
    questions: list[dict],
    responses: list[str],
    judge_cfg: dict,
    backend_name: str,
    device: str,
    batch_size: int,
) -> list[int]:
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
    backend.gen_kwargs.temperature    = 0.0

    judge_prompts = []
    for q, resp in zip(questions, responses):
        stripped = strip_think_blocks(resp)
        truncated = stripped[-500:] if len(stripped) > 500 else stripped
        judge_prompts.append(
            build_judge_prompt(
                dataset="circle_test",
                question=q["prompt"][-4000:],
                response=truncated,
                gold=q["gold"],
            )
        )

    scores: list[int] = []
    for i in range(0, len(judge_prompts), batch_size):
        batch = judge_prompts[i: i + batch_size]
        outs, _ = timed_generate(backend, batch)
        for raw in outs:
            correct_bool, _ = parse_correct(raw)
            scores.append(1 if correct_bool else 0)
        print(f"  judge {min(i + batch_size, len(judge_prompts))}/{len(judge_prompts)}", flush=True)

    backend.close()
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",             default=None,
                   help="Run circle test for this model only (omit to run all).")
    p.add_argument("--phase",             choices=["inference", "judge", "both"], default="both",
                   help="'inference': run model, save responses; "
                        "'judge': load saved responses, run judge, write results; "
                        "'both': original end-to-end behaviour.")
    p.add_argument("--backend",           choices=["auto", "hf", "vllm"], default="auto")
    p.add_argument("--device",            default="cuda")
    p.add_argument("--batch-size",        type=int, default=8)
    p.add_argument("--judge-batch-size",  type=int, default=1)
    p.add_argument("--max-new-tokens",    type=int, default=4096)
    return p.parse_args()


def _build_by_model(args):
    """Return (by_model, uninformative) filtered to args.model if set."""
    one_correct, uninformative = find_one_correct_questions()
    print(f"  {len(one_correct)} questions with exactly 1 correct model (will run circle test)")
    print(f"  {len(uninformative)} uninformative questions (0 or all models correct — excluded)")

    by_model: dict[str, list[dict]] = defaultdict(list)
    for pid, info in one_correct.items():
        by_model[info["model"]].append({"prompt_id": pid, **info})

    if args.model is not None:
        by_model = {args.model: by_model.get(args.model, [])}
        print(f"  Filtering to model: {args.model} ({len(by_model[args.model])} questions)")
    else:
        for model_id, qs in by_model.items():
            print(f"  {model_id}: {len(qs)} questions", flush=True)

    return by_model, uninformative


def run_inference_phase(args) -> None:
    """Run model inference and save raw responses to RESPONSES_DIR."""
    if args.model is None:
        raise ValueError("--model is required for --phase inference")

    print("Finding questions where exactly one model was correct...", flush=True)
    by_model, uninformative = _build_by_model(args)
    questions = by_model.get(args.model, [])

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESPONSES_DIR / f"{args.model}.json"

    if not questions:
        print(f"No one-correct questions for '{args.model}'. Writing empty response file.")
        out_path.write_text(json.dumps({
            "model": args.model, "uninformative": sorted(uninformative),
            "questions": [], "responses": [],
        }, indent=2))
        return

    print(f"\n[{args.model}] Running inference on {len(questions)} questions...", flush=True)
    models_cfg  = load_models_cfg()
    model_cfg   = get_model_cfg(models_cfg, args.model)
    max_new_tokens = int(model_cfg.get("max_model_len", 32768)) - 2048

    responses = run_inference_for_model(
        args.model, questions,
        backend_name=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
    )

    out_path.write_text(json.dumps({
        "model":        args.model,
        "uninformative": sorted(uninformative),
        "questions":    questions,
        "responses":    responses,
    }, indent=2))
    print(f"\nDone. Responses saved → {out_path}")


def run_judge_phase(args) -> None:
    """Load saved responses from RESPONSES_DIR, run judge, write partial results."""
    response_files = sorted(RESPONSES_DIR.glob("*.json"))
    if not response_files:
        raise FileNotFoundError(f"No response files found in {RESPONSES_DIR}. Run --phase inference first.")

    all_questions: list[dict] = []
    all_responses: list[str]  = []
    # Track which model each question belongs to and the uninformative set per model
    question_models: list[str] = []
    uninformative_by_model: dict[str, list] = {}

    for rf in response_files:
        data = json.loads(rf.read_text())
        model_id = data["model"]
        uninformative_by_model[model_id] = data.get("uninformative", [])
        for q, r in zip(data["questions"], data["responses"]):
            all_questions.append(q)
            all_responses.append(r)
            question_models.append(model_id)

    print(f"Loaded {len(all_questions)} responses from {len(response_files)} models.", flush=True)

    judge_cfg = load_judge_cfg()
    print(f"\nRunning judge over {len(all_questions)} responses...", flush=True)
    scores = run_judge(
        all_questions, all_responses,
        judge_cfg=judge_cfg,
        backend_name=args.backend,
        device=args.device,
        batch_size=args.judge_batch_size,
    )

    # Group failures back by model
    failures_by_model: dict[str, list] = defaultdict(list)
    for q, s, model_id in zip(all_questions, scores, question_models):
        if s == 0:
            failures_by_model[model_id].append(q["prompt_id"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    for model_id, uninformative in uninformative_by_model.items():
        partial = OUT_FILE.parent / f"circle_test_{model_id}.json"
        partial.write_text(json.dumps({
            "model":         model_id,
            "failures":      sorted(failures_by_model.get(model_id, [])),
            "uninformative": sorted(uninformative),
        }, indent=2))
        print(f"  Wrote {partial.name}")

    total_failures = sum(len(v) for v in failures_by_model.values())
    print(f"\nDone. {total_failures}/{len(all_questions)} failed circle test.")
    print(f"Run merge_circle_test.py to combine into {OUT_FILE.name}")


def main():
    args = parse_args()

    if args.phase == "inference":
        run_inference_phase(args)
        return

    if args.phase == "judge":
        run_judge_phase(args)
        return

    # --phase both: original end-to-end behaviour
    print("Finding questions where exactly one model was correct...", flush=True)
    by_model, uninformative = _build_by_model(args)

    judge_cfg = load_judge_cfg()

    all_questions: list[dict] = []
    all_responses: list[str]  = []

    for model_id, questions in by_model.items():
        if not questions:
            continue
        print(f"\n[{model_id}] Running inference on {len(questions)} questions...", flush=True)
        models_cfg = load_models_cfg()
        model_cfg  = get_model_cfg(models_cfg, model_id)
        max_new_tokens = int(model_cfg.get("max_model_len", 32768)) - 2048

        responses = run_inference_for_model(
            model_id, questions,
            backend_name=args.backend,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=max_new_tokens,
        )
        all_questions.extend(questions)
        all_responses.extend(responses)

    print(f"\nRunning judge over {len(all_questions)} responses...", flush=True)
    scores = run_judge(
        all_questions, all_responses,
        judge_cfg=judge_cfg,
        backend_name=args.backend,
        device=args.device,
        batch_size=args.judge_batch_size,
    )

    failures = [q["prompt_id"] for q, s in zip(all_questions, scores) if s == 0]
    print(f"  {len(failures)}/{len(all_questions)} failed circle test", flush=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if args.model is not None:
        partial = OUT_FILE.parent / f"circle_test_{args.model}.json"
        partial.write_text(json.dumps({
            "model":         args.model,
            "failures":      sorted(failures),
            "uninformative": sorted(uninformative),
        }, indent=2))
        print(f"\nDone. Partial results → {partial}")
    else:
        output = {
            "circle_test_failures": sorted(failures),
            "uninformative":        sorted(uninformative),
            "exclude":              sorted(set(failures) | set(uninformative)),
        }
        OUT_FILE.write_text(json.dumps(output, indent=2))
        print(f"\nDone.")
        print(f"  {len(failures)} prompt_ids failed the circle test")
        print(f"  {len(uninformative)} prompt_ids uninformative (0 or all correct)")
        print(f"  {len(output['exclude'])} total prompt_ids to exclude → {OUT_FILE}")


if __name__ == "__main__":
    main()
