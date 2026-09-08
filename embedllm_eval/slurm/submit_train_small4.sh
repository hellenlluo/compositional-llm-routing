#!/usr/bin/env bash
# Train EmbedLLM restricted to the 4 small/weak models:
#   mistral-7b-instruct-v0.3, qwen1.5-0.5b-chat,
#   phi-4-mini-instruct, llama-3.1-nemotron-nano-8b
#
# Usage:
#   bash embedllm_eval/submit_train_small4.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
SAVE_PATH="/n/fs/scratch/dl3533/models/embedclassifier_small4.pt"
EXCLUDED="deepseek-r1-distill-llama-8b llama-3.1-8b-instruct medgemma-4b-it mathstral-7b qwen3-30b-a3b qwen3-4b-thinking-2507"

sbatch \
    --job-name=embedllm-small4 \
    --partition=pvl \
    --account=allcs \
    --gres=gpu:a6000:1 \
    --time=01:00:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/train_small4_%j.out" \
    --error="$SLURM_LOGS/train_small4_%j.out" \
    --wrap="cd $REPO_ROOT && \
        XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        $CONDA_PYTHON embedllm_eval/train_embeddings.py \
            --epochs 50 \
            --batch-size 2048 \
            --lr 1e-3 \
            --exclude-models $EXCLUDED \
            --save-path $SAVE_PATH"

echo "Submitted embedllm-small4. Logs: $SLURM_LOGS/train_small4_<jobid>.out"
echo "Checkpoint: $SAVE_PATH"
