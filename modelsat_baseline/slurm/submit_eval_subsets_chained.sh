#!/usr/bin/env bash
# Run ModelSAT chained subtask evaluation for no_qwen3 and small4 subsets.
# Re-runs ModelSAT inference on individual subtask texts (slow — requires GPU).
# Merges chained fields into subtask_summary.json alongside the overall subtask metrics.
#
# Usage:
#   bash modelsat_baseline/submit_eval_subsets_chained.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/modelsat_baseline"
LOGS_DIR="$SCRIPT_DIR/slurm-logs"
CKPT="$SCRIPT_DIR/router_checkpoint_v2/model_sat_stage2.pt"
mkdir -p "$LOGS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

NO_QWEN3="deepseek-r1-distill-llama-8b llama-3.1-8b-instruct llama-3.1-nemotron-nano-8b mathstral-7b medgemma-4b-it mistral-7b-instruct-v0.3 phi-4-mini-instruct qwen1.5-0.5b-chat"
SMALL4="mistral-7b-instruct-v0.3 qwen1.5-0.5b-chat phi-4-mini-instruct llama-3.1-nemotron-nano-8b"

sbatch \
    --job-name=modelsat-chained \
    --account=pvl \
    --gres=gpu:a6000:1 \
    --time=6:00:00 \
    --mem=48G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/eval_subsets_chained_%j.out" \
    --error="$LOGS_DIR/eval_subsets_chained_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        echo '=== no_qwen3 (chained) ===' && \
        $CONDA_PYTHON modelsat_baseline/eval_subset.py \
            --models $NO_QWEN3 \
            --out-dir modelsat_baseline/results/no_qwen3 \
            --ckpt $CKPT \
            --chained && \
        echo '=== small4 (chained) ===' && \
        $CONDA_PYTHON modelsat_baseline/eval_subset.py \
            --models $SMALL4 \
            --out-dir modelsat_baseline/results/small4 \
            --ckpt $CKPT \
            --chained"

echo "Submitted modelsat-chained."
echo "Logs:    $LOGS_DIR/eval_subsets_chained_<jobid>.out"
echo "Results: modelsat_baseline/results/no_qwen3/subtask_summary.json  (with chained fields)"
echo "         modelsat_baseline/results/small4/subtask_summary.json    (with chained fields)"
