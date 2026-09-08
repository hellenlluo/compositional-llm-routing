#!/usr/bin/env bash
# Re-derive ModelSAT summaries for no_qwen3 and small4 subsets
# from the existing predictions.json — no re-inference needed.
#
# Usage:
#   bash modelsat_baseline/submit_eval_subsets.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/modelsat_baseline"
LOGS_DIR="$SCRIPT_DIR/slurm-logs"
mkdir -p "$LOGS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

NO_QWEN3="deepseek-r1-distill-llama-8b llama-3.1-8b-instruct llama-3.1-nemotron-nano-8b mathstral-7b medgemma-4b-it mistral-7b-instruct-v0.3 phi-4-mini-instruct qwen1.5-0.5b-chat"
SMALL4="mistral-7b-instruct-v0.3 qwen1.5-0.5b-chat phi-4-mini-instruct llama-3.1-nemotron-nano-8b"

sbatch \
    --job-name=modelsat-subsets \
    --account=allcs \
    --partition=pvl \
    --time=1:00:00 \
    --mem=32G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/eval_subsets_%j.out" \
    --error="$LOGS_DIR/eval_subsets_%j.out" \
    --wrap="cd $REPO_ROOT && \
        echo '=== no_qwen3 ===' && \
        $CONDA_PYTHON modelsat_baseline/eval_subset.py \
            --models $NO_QWEN3 \
            --out-dir modelsat_baseline/results/no_qwen3 && \
        echo '=== small4 ===' && \
        $CONDA_PYTHON modelsat_baseline/eval_subset.py \
            --models $SMALL4 \
            --out-dir modelsat_baseline/results/small4"

echo "Submitted modelsat-subsets."
echo "Logs:    $LOGS_DIR/eval_subsets_<jobid>.out"
echo "Results: modelsat_baseline/results/no_qwen3/summary.json"
echo "         modelsat_baseline/results/no_qwen3/subtask_summary.json"
echo "         modelsat_baseline/results/small4/summary.json"
echo "         modelsat_baseline/results/small4/subtask_summary.json"
