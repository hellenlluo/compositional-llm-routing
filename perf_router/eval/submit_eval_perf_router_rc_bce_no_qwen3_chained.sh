#!/usr/bin/env bash
# Chained subtask evaluation for BCE no-qwen3 perf router on R+C questions (GPU).
# Runs eval_perf_router.py with --qid-filter, producing full-task + overall + chained.
# Overwrites rc_questions/bce_no_qwen3/subtask_summary.json with chained fields included.
#
# Prerequisites:
#   1. python router_analysis/generate_rc_qids.py   (outputs/rc_qids.json)
#   2. bash perf_router/eval/submit_eval_perf_router_bce_no_qwen3.sh  (checkpoint must exist)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_LOGS="$REPO_ROOT/perf_router/logs"
RESULTS_DIR="$REPO_ROOT/perf_router/results/rc_questions/bce_no_qwen3"
QID_FILTER="$REPO_ROOT/outputs/rc_qids.json"
mkdir -p "$SLURM_LOGS" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
CKPT="/n/fs/scratch/dl3533/models/perf_router/perf_router_bce_no_qwen3_best.pt"

sbatch \
    --job-name="perf_rc_chain_bce_nq" \
    --account=pvl \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=0:20:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/eval_perf_rc_bce_no_qwen3_chained_%j.out" \
    --error="$SLURM_LOGS/eval_perf_rc_bce_no_qwen3_chained_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON perf_router/eval/eval_perf_router.py \
            --ckpt $CKPT \
            --qid-filter $QID_FILTER \
            --summary-out $RESULTS_DIR/summary.json \
            --subtask-out $RESULTS_DIR/subtask_summary.json \
            --predictions-out $RESULTS_DIR/predictions.json"

echo "Submitted perf_rc_chain_bce_nq. Logs: $SLURM_LOGS/eval_perf_rc_bce_no_qwen3_chained_<jobid>.out"
echo "Results: $RESULTS_DIR/"
echo "  Full-task: $RESULTS_DIR/summary.json"
echo "  Subtask:   $RESULTS_DIR/subtask_summary.json  (overall + chained)"
