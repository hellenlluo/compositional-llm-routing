#!/usr/bin/env bash
# Train the EfficiencyRouter (frozen encoder mode by default).
#
# First run: encodes ~3000 MMLU prompts with Qwen3-Embedding-0.6B (~2 min),
#            caches the embeddings, then trains the routing head (fast).
# Subsequent runs: loads cached embeddings, skips encoding entirely.
#
# Usage:
#   bash efficiency_router/submit_train.sh                    # defaults
#   bash efficiency_router/submit_train.sh --offset 0.5       # pure efficiency
#   bash efficiency_router/submit_train.sh --finetune-encoder # joint training
#   bash efficiency_router/submit_train.sh --rank-weight 0.1  # + pairwise loss
#
# All extra args are forwarded directly to train.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

# Forward any extra CLI args to train.py (e.g. --offset 0 --rank-weight 0.1)
EXTRA_ARGS="${*}"

sbatch \
    --job-name=effrouter \
    --account=allcs \
    --gres=gpu:1 \
    --time=1:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output="$LOGS_DIR/train_%j.out" \
    --error="$LOGS_DIR/train_%j.out" \
    --wrap="cd $SCRIPT_DIR && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON train.py $EXTRA_ARGS"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: $LOGS_DIR/train_<jobid>.out"
