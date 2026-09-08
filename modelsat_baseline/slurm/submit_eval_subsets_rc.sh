#!/usr/bin/env bash
# Re-derive ModelSAT summaries for no_qwen3 and small4 subsets on R+C questions only.
# Uses existing predictions.json — no re-inference needed.
#
# Prerequisites:
#   python router_analysis/generate_rc_qids.py   (run once to produce outputs/rc_qids.json)
#
# Usage:
#   bash modelsat_baseline/submit_eval_subsets_rc.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/modelsat_baseline"
LOGS_DIR="$SCRIPT_DIR/slurm-logs"
QID_FILTER="$REPO_ROOT/outputs/rc_qids.json"
mkdir -p "$LOGS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

NO_QWEN3="deepseek-r1-distill-llama-8b llama-3.1-8b-instruct llama-3.1-nemotron-nano-8b mathstral-7b medgemma-4b-it mistral-7b-instruct-v0.3 phi-4-mini-instruct qwen1.5-0.5b-chat"
SMALL4="mistral-7b-instruct-v0.3 qwen1.5-0.5b-chat phi-4-mini-instruct llama-3.1-nemotron-nano-8b"

sbatch \
    --job-name=modelsat-rc \
    --account=allcs \
    --partition=pvl \
    --time=0:30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/eval_subsets_rc_%j.out" \
    --error="$LOGS_DIR/eval_subsets_rc_%j.out" \
    --wrap="cd $REPO_ROOT && \
        echo '=== no_qwen3 (R+C) ===' && \
        $CONDA_PYTHON modelsat_baseline/eval_subset.py \
            --models $NO_QWEN3 \
            --qid-filter $QID_FILTER \
            --out-dir modelsat_baseline/results/rc_questions/no_qwen3 && \
        echo '=== small4 (R+C) ===' && \
        $CONDA_PYTHON modelsat_baseline/eval_subset.py \
            --models $SMALL4 \
            --qid-filter $QID_FILTER \
            --out-dir modelsat_baseline/results/rc_questions/small4"

echo "Submitted modelsat-rc."
echo "Logs:    $LOGS_DIR/eval_subsets_rc_<jobid>.out"
echo "Results: modelsat_baseline/results/rc_questions/no_qwen3/summary.json"
echo "         modelsat_baseline/results/rc_questions/no_qwen3/subtask_summary.json"
echo "         modelsat_baseline/results/rc_questions/small4/summary.json"
echo "         modelsat_baseline/results/rc_questions/small4/subtask_summary.json"
