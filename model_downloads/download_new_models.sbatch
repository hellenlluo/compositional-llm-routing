#!/bin/bash
#SBATCH --job-name=dl-new-models
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/n/fs/scratch/dl3533/models/download_new_models_%j.out
#SBATCH --error=/n/fs/scratch/dl3533/models/download_new_models_%j.err

set -euo pipefail

MODELS_DIR="/n/fs/scratch/dl3533/models"
PIP_TMP="/n/fs/scratch/dl3533/pip_tmp"

export HF_HOME="/n/fs/scratch/dl3533/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET="1"
export PYTHONPATH="$PIP_TMP:$PYTHONPATH"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/scratch/dl3533/conda_envs/MARIO

python3 - <<'PYEOF'
import os, sys, traceback
sys.path.insert(0, "/n/fs/scratch/dl3533/pip_tmp")

import huggingface_hub
print(f"huggingface_hub version: {huggingface_hub.__version__}", flush=True)
from huggingface_hub import snapshot_download, login, whoami
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

MODELS_DIR = "/n/fs/scratch/dl3533/models"

MODELS = [
    ("mistralai/Mathstral-7B-v0.1",          "Mathstral-7B-v0.1"),
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "Llama-3.1-Nemotron-Nano-8B-v1"),
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
    print("[WARN] No HF_TOKEN set — public models only.", flush=True)

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
        print(f"[OK]    {repo_id}", flush=True)
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
PYEOF
