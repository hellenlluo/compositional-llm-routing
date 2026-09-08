#!/usr/bin/env bash
# Train the accuracy-augmented perf router with qwen3-30b-a3b and
# qwen3-4b-thinking-2507 excluded from training.
# This forces the router to discriminate among the remaining 8 models.
#
# Requires:
#   perf_router/results/accuracy_vectors.json  (run submit_build_accuracy_vectors.sh first)
#
# Usage:
#   bash perf_router/training/submit_train_perf_router_no_qwen3.sh

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

EXCLUDED="qwen3-30b-a3b qwen3-4b-thinking-2507"
SAVE_PATH="/n/fs/scratch/dl3533/models/perf_router/perf_router_no_qwen3.pt"

sbatch \
    --job-name="perf_router_no_qwen3" \
    --account=allcs \
    --gres=gpu:a6000:1 \
    --time=00:45:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/train_perf_router_no_qwen3_%j.out" \
    --error="$SLURM_LOGS/train_perf_router_no_qwen3_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        $CONDA_PYTHON perf_router/training/train_perf_router.py \
            --epochs 100 \
            --batch-size 512 \
            --lr 1e-3 \
            --weight-decay 1e-3 \
            --dropout 0.2 \
            --embed-dim 1024 \
            --eval-every 5 \
            --exclude-models $EXCLUDED \
            --save-path $SAVE_PATH \
            $EXTRA_ARGS"

echo "Submitted perf_router_no_qwen3. Logs: $SLURM_LOGS/train_perf_router_no_qwen3_<jobid>.out"
echo "Model will be saved to: $SAVE_PATH"
