#!/usr/bin/env bash
# Submit the embedding training job to SLURM.
# Usage: bash embedllm_eval/submit_train_embeddings.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/pvl-chgin-v2/david/conda_envs/embedllm/bin/python"

sbatch \
    --job-name=train-embeddings \
    --partition=all \
    --account=allcs \
    --gres=gpu:1 \
    --time=01:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output="$SLURM_LOGS/train_embeddings_%j.out" \
    --error="$SLURM_LOGS/train_embeddings_%j.out" \
    --wrap="cd $REPO_ROOT && \
        XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        $CONDA_PYTHON embedllm_eval/train_embeddings.py \
        --epochs 50 \
        --batch-size 2048 \
        --lr 1e-3"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: $SLURM_LOGS/train_embeddings_<jobid>.out"
