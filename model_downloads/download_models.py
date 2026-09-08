#!/usr/bin/env python3
"""Download missing models to /n/fs/scratch/dl3533/models/
Skips models whose folder already contains .safetensors or .bin files.
Set HF_TOKEN env var before running to access gated models (Llama, MedGemma).
"""

import os
import sys
import traceback

sys.path.insert(0, "/n/fs/scratch/dl3533/pip_tmp")

os.environ.setdefault("HF_HOME", "/n/fs/scratch/dl3533/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/n/fs/scratch/dl3533/.cache/huggingface/hub")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import huggingface_hub
print(f"huggingface_hub version: {huggingface_hub.__version__}", flush=True)

from huggingface_hub import snapshot_download, login, whoami
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

MODELS_DIR = "/n/fs/scratch/dl3533/models"

MODELS = [
    ("Qwen/Qwen3-4B-Thinking-2507",             "Qwen3-4B-Thinking-2507"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Llama-8B"),
    ("google/medgemma-1.5-4b-it",                "medgemma-1.5-4b-it"),
    ("meta-llama/Llama-3.1-8B-Instruct",         "Llama-3.1-8B-Instruct"),
    ("Qwen/Qwen3-30B-A3B",                       "Qwen3-30B-A3B"),
    ("microsoft/Phi-4-mini-instruct",             "Phi-4-mini-instruct"),
    ("mistralai/Mistral-7B-Instruct-v0.3",       "Mistral-7B-Instruct-v0.3"),
    ("Qwen/Qwen1.5-0.5B-Chat",                   "Qwen1.5-0.5B-Chat"),
    ("mistralai/Mathstral-7B-v0.1",              "Mathstral-7B-v0.1"),
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1",     "Llama-3.1-Nemotron-Nano-8B-v1"),
    # Evaluator LLM
    ("Qwen/Qwen2.5-32B-Instruct",                "Qwen2.5-32B-Instruct"),
]

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if token:
    login(token=token, add_to_git_credential=False)
    try:
        user = whoami(token=token)
        print(f"Logged in as: {user['name']}", flush=True)
    except Exception as e:
        print(f"[WARN] Token login check failed: {e}", flush=True)
else:
    print("[WARN] No HF_TOKEN set — gated models will fail.", flush=True)

results = {"ok": [], "skipped": [], "gated": [], "failed": []}

for repo_id, folder_name in MODELS:
    dest = os.path.join(MODELS_DIR, folder_name)
    if os.path.isdir(dest) and any(
        f.endswith((".safetensors", ".bin", ".gguf"))
        for f in os.listdir(dest)
    ):
        print(f"[SKIP]  {repo_id} — already present")
        results["skipped"].append(repo_id)
        continue

    print(f"[DL]    {repo_id} → {dest} ...", flush=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            token=token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"[OK]    {repo_id}")
        results["ok"].append(repo_id)
    except GatedRepoError:
        print(f"[GATED] {repo_id} — set HF_TOKEN with accepted license and re-run.")
        results["gated"].append(repo_id)
    except RepositoryNotFoundError:
        print(f"[404]   {repo_id} — repo not found or private.")
        results["failed"].append(repo_id)
    except Exception as e:
        print(f"[ERR]   {repo_id} — {type(e).__name__}: {e}")
        traceback.print_exc()
        results["failed"].append(repo_id)

print("\n=== Summary ===")
print(f"  Downloaded : {len(results['ok'])}")
print(f"  Skipped    : {len(results['skipped'])}")
print(f"  Gated      : {len(results['gated'])}")
print(f"  Failed     : {len(results['failed'])}")
if results["gated"]:
    print("\nTo download gated models, run:")
    print("  HF_TOKEN=hf_xxx python3 download_models.py")
