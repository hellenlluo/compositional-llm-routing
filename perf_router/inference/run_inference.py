#!/usr/bin/env python3
"""
Run inference on "What is the capital of France?" for the 10 project models.

Models excluded: all-mpnet-base-v2, e5-large-v2, phi-3-mini-128k-instruct,
                 prometheus-7b-v2.0, qwen2.5-32b-instruct

Results are written to results/capital_of_france.json.

Usage:
    python perf_router/run_inference.py [--backend auto|hf|vllm] [--hf-token TOKEN]

When --hf-token is provided (or HF_TOKEN is set in the environment), models are
loaded directly from HuggingFace Hub using that token instead of local disk paths.
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

# Load .env from repo root so HF_TOKEN (and other secrets) are available
# without needing to export them manually in the shell.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from eval.inference import make_backend  # noqa: E402

PROMPT = "What is the capital of France?"

# api_model: "{hf_repo}:{provider}" used with --backend api.
# Check each model's HF page (Deploy button) to find its available providers.
# Leave unset (or set to None) to skip a model in api mode.
MODELS = [
    {
        "id": "deepseek-r1-distill-llama-8b",
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "local_path": "/n/fs/scratch/dl3533/models/DeepSeek-R1-Distill-Llama-8B",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "api_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B:nscale",
    },
    {
        "id": "llama-3.1-8b-instruct",
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "local_path": "/n/fs/scratch/dl3533/models/Llama-3.1-8B-Instruct",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "api_model": "meta-llama/Llama-3.1-8B-Instruct:novita",
    },
    {
        "id": "llama-3.1-nemotron-nano-8b",
        "hf_repo": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "local_path": "/n/fs/scratch/dl3533/models/Llama-3.1-Nemotron-Nano-8B-v1",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "api_model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1:featherless-ai",
    },
    {
        "id": "mathstral-7b",
        "hf_repo": "mistralai/Mathstral-7B-v0.1",
        "local_path": "/n/fs/scratch/dl3533/models/Mathstral-7B-v0.1",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "api_model": None,  # TODO: check HF page for provider
    },
    {
        "id": "medgemma-4b-it",
        "hf_repo": "google/medgemma-4b-it",
        "local_path": "/n/fs/scratch/dl3533/models/medgemma-1.5-4b-it",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "api_model": None,
        "max_new_tokens": 1024,
        "vlm_batch_size": 4,         # batched generation in VLMTextBackend
        "vlm_max_input_length": 2048, # truncate long LogicBench contexts
        "backend": "sagemaker",      # remote: deploy via SageMaker
        "local_backend": "vlm_hf",   # local: AutoModelForImageTextToText, text-only
        "sm_instance_type": "ml.g5.2xlarge",
        "sm_num_gpus": 1,
    },
    {
        "id": "mistral-7b-instruct-v0.3",
        "hf_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "local_path": "/n/fs/scratch/dl3533/models/Mistral-7B-Instruct-v0.3",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "api_model": None,
        "backend": "bedrock",
        "bedrock_model_id": "mistral.mistral-7b-instruct-v0:2",
    },
    {
        "id": "phi-4-mini-instruct",
        "hf_repo": "microsoft/Phi-4-mini-instruct",
        "local_path": "/n/fs/scratch/dl3533/models/Phi-4-mini-instruct",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "trust_remote_code": False,
        "api_model": "microsoft/Phi-4-mini-instruct:featherless-ai",  # TODO: check HF page for provider
    },
    {
        "id": "qwen1.5-0.5b-chat",
        "hf_repo": "Qwen/Qwen1.5-0.5B-Chat",
        "local_path": "/n/fs/scratch/dl3533/models/Qwen1.5-0.5B-Chat",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "api_model": "Qwen/Qwen1.5-0.5B-Chat:featherless-ai",  # TODO: check HF page for provider
    },
    {
        "id": "qwen3-30b-a3b",
        "hf_repo": "Qwen/Qwen3-30B-A3B",
        "local_path": "/n/fs/scratch/dl3533/models/Qwen3-30B-A3B",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "tensor_parallel_size": 2,
        "api_model": "Qwen/Qwen3-30B-A3B:novita",  # TODO: check HF page for provider
    },
    {
        "id": "qwen3-4b-thinking-2507",
        "hf_repo": "Qwen/Qwen3-4B-Thinking-2507",
        "local_path": "/n/fs/scratch/dl3533/models/Qwen3-4B-Thinking-2507",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "api_model": "Qwen/Qwen3-4B-Thinking-2507:nscale",  # TODO: check HF page for provider
    },
]


def run(backend_name: str, hf_token: str | None = None) -> None:
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "capital_of_france.json"

    all_results: list[dict] = []

    for i, model_cfg in enumerate(MODELS, 1):
        model_id = model_cfg["id"]
        effective_backend = model_cfg.get("backend", backend_name)
        if effective_backend == "api":
            if not model_cfg.get("api_model"):
                print(f"\n[{i}/{len(MODELS)}] Skipping {model_id} (no api_model set — check HF page for provider)", flush=True)
                continue
            model_source = model_cfg["hf_repo"]
            print(f"\n[{i}/{len(MODELS)}] {model_id} via {model_cfg['api_model']} ...", flush=True)
        elif effective_backend == "bedrock":
            model_source = model_cfg["hf_repo"]
            print(f"\n[{i}/{len(MODELS)}] {model_id} via Bedrock ({model_cfg['bedrock_model_id']}) ...", flush=True)
        elif effective_backend == "sagemaker":
            model_source = model_cfg["hf_repo"]
            print(f"\n[{i}/{len(MODELS)}] {model_id} via SageMaker ({model_cfg.get('sm_instance_type', 'ml.g5.2xlarge')}) ...", flush=True)
        elif hf_token:
            model_source = model_cfg["hf_repo"]
            print(f"\n[{i}/{len(MODELS)}] Loading {model_id} from HuggingFace Hub ({model_source}) ...", flush=True)
        else:
            model_source = model_cfg["local_path"]
            print(f"\n[{i}/{len(MODELS)}] Loading {model_id} from {model_source} ...", flush=True)

        cfg_for_backend = {**model_cfg, "hf_repo": model_source}

        try:
            backend = make_backend(cfg_for_backend, backend=backend_name, token=hf_token)
            t0 = time.monotonic()
            responses = backend.generate([PROMPT])
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            response = responses[0]
            backend.close()

            print(f"  Response ({elapsed_ms} ms): {response!r}", flush=True)
            all_results.append({
                "model_id": model_id,
                "prompt": PROMPT,
                "response": response,
                "elapsed_ms": elapsed_ms,
                "error": None,
            })
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            all_results.append({
                "model_id": model_id,
                "prompt": PROMPT,
                "response": None,
                "elapsed_ms": None,
                "error": str(exc),
            })

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")

    print("\n=== Summary ===")
    for r in all_results:
        status = r["response"] if r["error"] is None else f"ERROR: {r['error']}"
        print(f"  {r['model_id']}: {status!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["auto", "hf", "vllm", "api"],
        default=None,
        help="Inference backend: api (HF Inference Providers, default when HF_TOKEN is set), "
             "auto (vllm if available, else hf), hf, or vllm",
    )
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace Hub token. When set, models are downloaded from the Hub "
             "instead of local disk paths. Defaults to the HF_TOKEN environment variable.",
    )
    args = p.parse_args()
    backend = args.backend
    if backend is None:
        backend = "api" if args.hf_token else "auto"
    run(backend, hf_token=args.hf_token)


if __name__ == "__main__":
    main()
