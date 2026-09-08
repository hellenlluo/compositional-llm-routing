#!/usr/bin/env bash
# Evaluate the small4 perf router checkpoint.
# Results saved to perf_router/results/small4/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_LOGS="$REPO_ROOT/perf_router/logs"
RESULTS_DIR="$REPO_ROOT/perf_router/results/small4"
mkdir -p "$SLURM_LOGS" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
CKPT="/n/fs/scratch/dl3533/models/perf_router/perf_router_small4_best.pt"

sbatch \
    --job-name="eval_perf_small4" \
    --account=allcs \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=0:30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/eval_perf_small4_%j.out" \
    --error="$SLURM_LOGS/eval_perf_small4_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON perf_router/eval/eval_perf_router.py \
            --ckpt $CKPT \
            --summary-out $RESULTS_DIR/summary.json \
            --subtask-out $RESULTS_DIR/subtask_summary.json \
            --predictions-out $RESULTS_DIR/predictions.json"

echo "Submitted eval_perf_small4. Logs: $SLURM_LOGS/eval_perf_small4_<jobid>.out"
echo "Results: $RESULTS_DIR/"
echo "  Full-task: $RESULTS_DIR/summary.json"
echo "  Subtask:   $RESULTS_DIR/subtask_summary.json  (overall + chained)"
