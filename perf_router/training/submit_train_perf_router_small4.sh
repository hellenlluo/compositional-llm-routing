#!/usr/bin/env bash
# Train the accuracy-augmented perf router restricted to the 4 small/weak models:
#   mistral-7b-instruct-v0.3, qwen1.5-0.5b-chat,
#   phi-4-mini-instruct, llama-3.1-nemotron-nano-8b
#
# Requires:
#   perf_router/results/qwen3/accuracy_vectors.json
#
# Usage:
#   bash perf_router/training/submit_train_perf_router_small4.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_LOGS="$REPO_ROOT/perf_router/logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
EXTRA_ARGS="${*}"

PERF_VECTORS="$REPO_ROOT/perf_router/results/qwen3/accuracy_vectors.json"
if [[ ! -f "$PERF_VECTORS" ]]; then
    echo "ERROR: accuracy_vectors.json not found at $PERF_VECTORS"
    echo "Run submit_build_accuracy_vectors.sh first."
    exit 1
fi

EXCLUDED="deepseek-r1-distill-llama-8b llama-3.1-8b-instruct medgemma-4b-it mathstral-7b qwen3-30b-a3b qwen3-4b-thinking-2507"
SAVE_PATH="/n/fs/scratch/dl3533/models/perf_router/perf_router_small4.pt"

sbatch \
    --job-name="perf_router_small4" \
    --account=allcs \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=00:45:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/train_perf_router_small4_%j.out" \
    --error="$SLURM_LOGS/train_perf_router_small4_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        $CONDA_PYTHON perf_router/training/train_perf_router.py \
            --epochs 75 \
            --batch-size 512 \
            --lr 1e-3 \
            --weight-decay 1e-3 \
            --dropout 0.3 \
            --embed-dim 1024 \
            --eval-every 5 \
            --exclude-models $EXCLUDED \
            --save-path $SAVE_PATH \
            $EXTRA_ARGS"

echo "Submitted perf_router_small4. Logs: $SLURM_LOGS/train_perf_router_small4_<jobid>.out"
echo "Model will be saved to: $SAVE_PATH"
