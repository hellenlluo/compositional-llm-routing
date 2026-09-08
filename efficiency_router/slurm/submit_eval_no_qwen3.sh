#!/usr/bin/env bash
# Evaluate the no-qwen3 EfficiencyRouter checkpoint.
# Results saved to efficiency_router/results/no_qwen3/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results/no_qwen3"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
# Best offset from sweep (override: bash …/submit_eval_no_qwen3.sh --ckpt path/to/other.pt)
CKPT="$SCRIPT_DIR/checkpoints/effrouter_no_qwen3_offset_10.pt"
EXTRA_ARGS="${*}"

sbatch \
    --job-name=effrouter-eval-no-qwen3 \
    --account=allcs \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=0:30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/eval_no_qwen3_%j.out" \
    --error="$LOGS_DIR/eval_no_qwen3_%j.out" \
    --wrap="cd $SCRIPT_DIR && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON eval.py \
            --ckpt $CKPT \
            --out $RESULTS_DIR/eval_results.json \
            --subtask-out $RESULTS_DIR/subtask_results.json \
            --predictions-out $RESULTS_DIR/predictions.json \
            $EXTRA_ARGS"

echo "Submitted. Logs: $LOGS_DIR/eval_no_qwen3_<jobid>.out"
echo "Results: $RESULTS_DIR/"
echo "  Full-task:  $RESULTS_DIR/eval_results.json"
echo "  Subtask:    $RESULTS_DIR/subtask_results.json  (overall + chained)"
