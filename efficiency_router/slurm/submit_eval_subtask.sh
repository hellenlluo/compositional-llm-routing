#!/usr/bin/env bash
# Evaluate the trained EfficiencyRouter with both full-task and subtask metrics.
#
# Usage:
#   bash efficiency_router/submit_eval_subtask.sh
#   bash efficiency_router/submit_eval_subtask.sh --ckpt path/to/effrouter.pt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results/qwen3"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
EXTRA_ARGS="${*}"

sbatch \
    --job-name=effrouter-eval-subtask \
    --account=allcs \
    --gres=gpu:1 \
    --time=0:30:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output="$LOGS_DIR/eval_subtask_%j.out" \
    --error="$LOGS_DIR/eval_subtask_%j.out" \
    --wrap="cd $SCRIPT_DIR && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON eval.py \
            --out $RESULTS_DIR/eval_results.json \
            --subtask-out $RESULTS_DIR/subtask_results.json \
            --predictions-out $RESULTS_DIR/predictions.json \
            $EXTRA_ARGS"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs:            $LOGS_DIR/eval_subtask_<jobid>.out"
echo "Full-task out:   $RESULTS_DIR/eval_results.json"
echo "Subtask out:     $RESULTS_DIR/subtask_results.json  (overall + chained)"
