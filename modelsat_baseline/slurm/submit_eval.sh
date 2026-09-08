#!/usr/bin/env bash
# Evaluate the current ModelSAT checkpoint on morehopqa, musique, stepcot.
#
# Usage:
#   bash modelsat_baseline/submit_eval.sh
#   bash modelsat_baseline/submit_eval.sh --ckpt /path/to/other.pt
#   bash modelsat_baseline/submit_eval.sh --out-dir modelsat_baseline/results/stage2_step3000

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/modelsat_baseline"
LOGS_DIR="$SCRIPT_DIR/slurm-logs"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
CKPT="$SCRIPT_DIR/router_checkpoint_v2/model_sat_stage2.pt"
EXTRA_ARGS="${*}"

if [[ ! -f "$CKPT" ]] && [[ "$*" != *"--ckpt"* ]]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

sbatch \
    --job-name=modelsat-eval \
    --account=pvl \
    --gres=gpu:a6000:1 \
    --time=3:00:00 \
    --mem=48G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/eval_%j.out" \
    --error="$LOGS_DIR/eval_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON modelsat_baseline/eval.py \
            --ckpt $CKPT \
            --out-dir $RESULTS_DIR \
            --batch-size 4 \
            $EXTRA_ARGS"

echo "Submitted modelsat-eval."
echo "Logs:    $LOGS_DIR/eval_<jobid>.out"
echo "Results: $RESULTS_DIR/summary.json"
