#!/usr/bin/env bash
# Evaluate a trained EfficiencyRouter checkpoint on the test oracle matrices.
#
# Usage:
#   bash efficiency_router/submit_eval.sh                                  # default checkpoint
#   bash efficiency_router/submit_eval.sh --ckpt path/to/effrouter.pt
#   bash efficiency_router/submit_eval.sh --out efficiency_router/results.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

EXTRA_ARGS="${*}"

# Only inject the default --out if the caller didn't supply their own.
DEFAULT_OUT=""
if [[ "$EXTRA_ARGS" != *"--out"* ]]; then
    DEFAULT_OUT="--out $RESULTS_DIR/eval_results.json"
fi

sbatch \
    --job-name=effrouter-eval \
    --account=allcs \
    --gres=gpu:1 \
    --time=0:30:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output="$LOGS_DIR/eval_%j.out" \
    --error="$LOGS_DIR/eval_%j.out" \
    --wrap="cd $SCRIPT_DIR && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON eval.py \
            $DEFAULT_OUT \
            $EXTRA_ARGS"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: $LOGS_DIR/eval_<jobid>.out"
echo "Results: $RESULTS_DIR/eval_results.json"
