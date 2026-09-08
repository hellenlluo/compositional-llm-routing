#!/usr/bin/env bash
# Evaluate the small4 EfficiencyRouter checkpoint (full-task + subtask + chained).
# Results saved to efficiency_router/results/small4/
#
# Usage:
#   bash efficiency_router/submit_eval_small4.sh
#   bash efficiency_router/submit_eval_small4.sh --ckpt path/to/effrouter_small4_offset_N.pt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results/small4"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
# Best offset from sweep (override: bash …/submit_eval_small4.sh --ckpt path/to/other.pt)
CKPT="$SCRIPT_DIR/checkpoints/effrouter_small4_offset_10.pt"
EXTRA_ARGS="${*}"

sbatch \
    --job-name=effrouter-eval-small4 \
    --account=allcs \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=0:45:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/eval_small4_%j.out" \
    --error="$LOGS_DIR/eval_small4_%j.out" \
    --wrap="cd $SCRIPT_DIR && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON eval.py \
            --ckpt $CKPT \
            --out $RESULTS_DIR/eval_results.json \
            --subtask-out $RESULTS_DIR/subtask_results.json \
            --predictions-out $RESULTS_DIR/predictions.json \
            $EXTRA_ARGS"

echo "Submitted effrouter-eval-small4. Logs: $LOGS_DIR/eval_small4_<jobid>.out"
echo "Results: $RESULTS_DIR/"
echo "  Full-task:       $RESULTS_DIR/eval_results.json"
echo "  Subtask:         $RESULTS_DIR/subtask_results.json  (overall + chained)"
