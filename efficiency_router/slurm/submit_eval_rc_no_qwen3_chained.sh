#!/usr/bin/env bash
# Chained subtask evaluation for no-qwen3 EfficiencyRouter on R+C questions (GPU).
# Runs eval.py with --qid-filter, producing full-task + overall + chained subtask.
# Overwrites rc_questions/no_qwen3/subtask_results.json with chained fields included.
#
# Prerequisites:
#   1. python router_analysis/generate_rc_qids.py   (outputs/rc_qids.json)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results/rc_questions/no_qwen3"
QID_FILTER="$REPO_ROOT/outputs/rc_qids.json"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
CKPT="$SCRIPT_DIR/checkpoints/effrouter_no_qwen3_offset_10.pt"

sbatch \
    --job-name=effrouter-rc-chain-nq \
    --account=allcs \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=0:20:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/eval_rc_no_qwen3_chained_%j.out" \
    --error="$LOGS_DIR/eval_rc_no_qwen3_chained_%j.out" \
    --wrap="cd $SCRIPT_DIR && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON eval.py \
            --ckpt $CKPT \
            --qid-filter $QID_FILTER \
            --out $RESULTS_DIR/eval_results.json \
            --subtask-out $RESULTS_DIR/subtask_results.json \
            --predictions-out $RESULTS_DIR/predictions.json"

echo "Submitted effrouter-rc-chain-nq. Logs: $LOGS_DIR/eval_rc_no_qwen3_chained_<jobid>.out"
echo "Results: $RESULTS_DIR/"
echo "  Full-task: $RESULTS_DIR/eval_results.json"
echo "  Subtask:   $RESULTS_DIR/subtask_results.json  (overall + chained)"
