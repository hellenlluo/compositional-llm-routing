#!/usr/bin/env bash
# Train the EfficiencyRouter with qwen3-30b-a3b and qwen3-4b-thinking-2507 excluded.
# This forces the router to discriminate among the remaining 8 models.
#
# Checkpoint saved to: efficiency_router/checkpoints/effrouter_no_qwen3.pt
#
# Usage:
#   bash efficiency_router/submit_train_no_qwen3.sh
#   bash efficiency_router/submit_train_no_qwen3.sh --offset 0.5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
EXTRA_ARGS="${*}"

EXCLUDED="qwen3-30b-a3b qwen3-4b-thinking-2507"

sbatch \
    --job-name=effrouter_no_qwen3 \
    --account=allcs \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=1:00:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/train_no_qwen3_%j.out" \
    --error="$LOGS_DIR/train_no_qwen3_%j.out" \
    --wrap="cd $SCRIPT_DIR && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON train.py \
            --exclude-models $EXCLUDED \
            --ckpt-name effrouter_no_qwen3 \
            --epochs 50 \
            $EXTRA_ARGS"

echo "Submitted effrouter_no_qwen3. Logs: $LOGS_DIR/train_no_qwen3_<jobid>.out"
echo "Checkpoint: $SCRIPT_DIR/checkpoints/effrouter_no_qwen3.pt"
