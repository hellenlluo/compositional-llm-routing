#!/usr/bin/env bash
# Submit the specialized evaluation job to SLURM.
# Usage: bash embedllm_eval/submit_eval_specialized.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

sbatch \
    --job-name=eval-specialized \
    --partition=all \
    --account=allcs \
    --gres=gpu:1 \
    --time=01:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output="$SLURM_LOGS/eval_specialized_%j.out" \
    --error="$SLURM_LOGS/eval_specialized_%j.out" \
    --wrap="cd $REPO_ROOT && \
        XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        $CONDA_PYTHON embedllm_eval/eval_specialized.py"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: $SLURM_LOGS/eval_specialized_<jobid>.out"
echo "Results will be written to: $REPO_ROOT/embedllm_eval/eval_results/qwen3/"
