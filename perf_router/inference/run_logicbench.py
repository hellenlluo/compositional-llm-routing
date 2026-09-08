#!/usr/bin/env python3
"""
Run LogicBench (Eval) inference across all 8 configured models.

Crawls every BQA and MCQA data file under LogicBench(Eval) and runs
inference for each supported model.  Output files mirror the source tree:

    results/logicbench/<model_id>/<task_type>/<logic_type>/<axiom>/results.json

Each result file has the same shape as the input but with `response`,
`elapsed_ms`, and `unique_id` added to every sample / qa_pair.

Resume behaviour: if a result file already exists it is skipped, so the
script can be interrupted and restarted safely.

Usage:
    python perf_router/run_logicbench.py [--hf-token TOKEN]
    python perf_router/run_logicbench.py --model deepseek-r1-distill-llama-8b
    python perf_router/run_logicbench.py --task BQA --logic first_order_logic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from eval.inference import make_backend  # noqa: E402
from perf_router.run_inference import MODELS  # noqa: E402

LOGICBENCH_EVAL_DIR = (
    REPO_ROOT
    / "data_preprocessing/data/logicbench/data/LogicBench(Eval)"
)
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "logicbench"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_bqa_prompt(context: str, question: str) -> str:
    return (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer with only 'yes' or 'no'."
    )


def build_mcqa_prompt(context: str, question: str, choices: dict) -> str:
    choice_lines = "\n".join(
        f"  ({k.replace('choice_', '')}): {v}" for k, v in choices.items()
    )
    return (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choice_lines}\n\n"
        "Answer with only the choice number (e.g. '1', '2', '3', or '4')."
    )


# ---------------------------------------------------------------------------
# Model filter — skip models that cannot use the api/bedrock backends
# ---------------------------------------------------------------------------

def is_supported(model_cfg: dict, backend_name: str) -> bool:
    """Return True if this model has a functional backend configured."""
    effective = model_cfg.get("backend", backend_name)
    local_mode = backend_name in ("hf", "vllm", "auto", "vlm_hf")

    if local_mode:
        # SageMaker-only models are fine locally if they have a local_backend
        if effective == "sagemaker" and not model_cfg.get("local_backend"):
            return False
        # Skip models that are purely remote-only
        if model_cfg.get("backend") in ("api", "bedrock"):
            return False
    else:
        # API mode: skip models with no provider
        if effective == "api" and not model_cfg.get("api_model"):
            return False
        # Skip sagemaker-backend models in api mode
        if effective == "sagemaker":
            return False
    return True


# ---------------------------------------------------------------------------
# Core: run one data file for one model
# ---------------------------------------------------------------------------

def build_backend_cfg(model_cfg: dict, backend_name: str) -> dict:
    """Resolve the config dict to pass to make_backend (handles local_backend override)."""
    effective_backend = model_cfg.get("backend", backend_name)
    local_mode = backend_name in ("hf", "vllm", "auto", "vlm_hf")
    if local_mode and effective_backend == "sagemaker":
        effective_backend = model_cfg["local_backend"]
    if local_mode:
        return {
            **model_cfg,
            "backend": effective_backend,
            "hf_repo": model_cfg.get("local_path", model_cfg["hf_repo"]),
        }
    return model_cfg


def run_file(
    data_path: Path,
    task_type: str,      # "BQA" or "MCQA"
    logic_type: str,     # e.g. "first_order_logic"
    axiom: str,          # e.g. "bidirectional_dilemma"
    model_cfg: dict,
    backend,             # already-initialised Backend instance
    out_path: Path,
) -> None:
    with data_path.open() as f:
        data = json.load(f)

    model_id = model_cfg["id"]

    # Build (prompt, metadata) pairs so we can batch all at once
    records: list[dict] = []  # each entry describes one generation slot
    prompts: list[str] = []

    if task_type == "BQA":
        for sample in data["samples"]:
            sid = sample["id"]
            ctx = sample["context"]
            for qi, qap in enumerate(sample["qa_pairs"]):
                uid = f"BQA/{logic_type}/{axiom}/{sid}/{qi}"
                prompt = build_bqa_prompt(ctx, qap["question"])
                prompts.append(prompt)
                records.append({
                    "sample_id": sid,
                    "qa_idx": qi,
                    "unique_id": uid,
                    "answer": qap["answer"],
                })
    else:  # MCQA
        for sample in data["samples"]:
            sid = sample["id"]
            uid = f"MCQA/{logic_type}/{axiom}/{sid}"
            prompt = build_mcqa_prompt(
                sample["context"], sample["question"], sample["choices"]
            )
            prompts.append(prompt)
            records.append({
                "sample_id": sid,
                "unique_id": uid,
                "answer": sample["answer"],
            })

    # Run inference — batch the whole file in one backend call
    # Short answers only (yes/no or choice number) — cap output tokens
    t0 = time.monotonic()
    responses = backend.generate(prompts)
    total_ms = int((time.monotonic() - t0) * 1000)

    ms_each = total_ms // max(len(prompts), 1)

    # Re-assemble results into the same shape as the input file
    resp_iter = iter(zip(records, responses))

    if task_type == "BQA":
        out_samples = []
        for sample in data["samples"]:
            sid = sample["id"]
            out_qa = []
            for qap in sample["qa_pairs"]:
                rec, resp = next(resp_iter)
                out_qa.append({
                    "unique_id": rec["unique_id"],
                    "question": qap["question"],
                    "answer": qap["answer"],
                    "response": resp,
                    "elapsed_ms": ms_each,
                })
            out_samples.append({
                "id": sid,
                "unique_id": f"BQA/{logic_type}/{axiom}/{sid}",
                "context": sample["context"],
                "qa_pairs": out_qa,
            })
    else:
        out_samples = []
        for sample in data["samples"]:
            rec, resp = next(resp_iter)
            out_samples.append({
                "id": sample["id"],
                "unique_id": rec["unique_id"],
                "context": sample["context"],
                "question": sample["question"],
                "choices": sample["choices"],
                "answer": sample["answer"],
                "response": resp,
                "elapsed_ms": ms_each,
            })

    result = {
        "model_id": model_id,
        "task_type": task_type,
        "logic_type": logic_type,
        "axiom": axiom,
        "samples": out_samples,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"    Saved {len(prompts)} results → {out_path.relative_to(REPO_ROOT)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Discovery: collect all (task_type, logic_type, axiom, data_path) tuples
# ---------------------------------------------------------------------------

def discover_files(
    task_filter: str | None = None,
    logic_filter: str | None = None,
) -> list[tuple[str, str, str, Path]]:
    entries = []
    for task_dir in sorted(LOGICBENCH_EVAL_DIR.iterdir()):
        task_type = task_dir.name  # "BQA" or "MCQA"
        if task_filter and task_type != task_filter:
            continue
        filename = "data_instances.json" if task_type == "BQA" else "MCQ_data_instances.json"
        for logic_dir in sorted(task_dir.iterdir()):
            logic_type = logic_dir.name
            if logic_filter and logic_type != logic_filter:
                continue
            for axiom_dir in sorted(logic_dir.iterdir()):
                data_path = axiom_dir / filename
                if data_path.exists():
                    entries.append((task_type, logic_type, axiom_dir.name, data_path))
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument(
        "--backend",
        choices=["api", "auto", "hf", "vllm"],
        default=None,
        help="Global inference backend (default: api when HF_TOKEN set, else auto)",
    )
    p.add_argument(
        "--model",
        dest="model_filter",
        default=None,
        help="Run only this model ID (e.g. deepseek-r1-distill-llama-8b)",
    )
    p.add_argument("--task", default=None, choices=["BQA", "MCQA"],
                   help="Restrict to one task type")
    p.add_argument("--logic", default=None,
                   help="Restrict to one logic type (e.g. first_order_logic)")
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-run and overwrite existing result files",
    )
    args = p.parse_args()

    hf_token = args.hf_token
    backend_name = args.backend or ("api" if hf_token else "auto")

    supported_models = [
        m for m in MODELS
        if is_supported(m, backend_name)
        and (args.model_filter is None or m["id"] == args.model_filter)
    ]

    if not supported_models:
        print("No supported models found. Check --model filter or backend config.")
        sys.exit(1)

    files = discover_files(task_filter=args.task, logic_filter=args.logic)
    if not files:
        print("No data files found under", LOGICBENCH_EVAL_DIR)
        sys.exit(1)

    print(f"Models  : {[m['id'] for m in supported_models]}")
    print(f"Files   : {len(files)} data files")
    print(f"Backend : {backend_name}")
    print()

    total_files = len(supported_models) * len(files)
    done = skipped = errors = 0

    for model_cfg in supported_models:
        model_id = model_cfg["id"]
        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"{'='*60}")

        # Check how many files actually need running (skip already-done ones)
        pending = [
            (tt, lt, ax, dp)
            for tt, lt, ax, dp in files
            if not (RESULTS_ROOT / model_id / tt / lt / ax / "results.json").exists()
            or args.overwrite
        ]
        if not pending:
            print(f"  All {len(files)} files already done, skipping model.", flush=True)
            skipped += len(files)
            continue

        # Load the backend ONCE for all files of this model
        print(f"  Loading backend ({backend_name}) ...", flush=True)
        cfg = build_backend_cfg(model_cfg, backend_name)
        try:
            backend = make_backend(cfg, backend=backend_name, token=hf_token)
        except Exception as exc:
            print(f"  ERROR loading backend: {exc}", flush=True)
            errors += len(pending)
            skipped += len(files) - len(pending)
            continue

        try:
            for task_type, logic_type, axiom, data_path in files:
                out_path = (
                    RESULTS_ROOT
                    / model_id
                    / task_type
                    / logic_type
                    / axiom
                    / "results.json"
                )

                if out_path.exists() and not args.overwrite:
                    print(
                        f"  [skip] {task_type}/{logic_type}/{axiom} (already exists)",
                        flush=True,
                    )
                    skipped += 1
                    continue

                label = f"{task_type}/{logic_type}/{axiom}"
                print(f"  [{done + skipped + errors + 1}/{total_files}] {label}", flush=True)

                try:
                    run_file(
                        data_path=data_path,
                        task_type=task_type,
                        logic_type=logic_type,
                        axiom=axiom,
                        model_cfg=model_cfg,
                        backend=backend,
                        out_path=out_path,
                    )
                    done += 1
                except Exception as exc:
                    print(f"    ERROR: {exc}", flush=True)
                    errors += 1
        finally:
            backend.close()

    print(f"\n{'='*60}")
    print(f"Done. completed={done}  skipped={skipped}  errors={errors}")
    print(f"Results under: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
