#!/usr/bin/env bash
# Derive perf-router R+C metrics from existing predictions — no GPU needed.
# Full-task + overall subtask saved to perf_router/results/rc_questions/small4/
#
# Prerequisites:
#   1. python router_analysis/generate_rc_qids.py   (outputs/rc_qids.json)
#   2. bash perf_router/eval/submit_eval_perf_router_small4.sh  (produces predictions.json)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_LOGS="$REPO_ROOT/perf_router/logs"
PREDS="$REPO_ROOT/perf_router/results/small4/predictions.json"
RESULTS_DIR="$REPO_ROOT/perf_router/results/rc_questions/small4"
QID_FILTER="$REPO_ROOT/outputs/rc_qids.json"
mkdir -p "$SLURM_LOGS" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

sbatch \
    --job-name="eval_perf_rc_small4" \
    --account=allcs \
    --partition=pvl \
    --time=0:10:00 \
    --mem=8G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/eval_perf_rc_small4_%j.out" \
    --error="$SLURM_LOGS/eval_perf_rc_small4_%j.out" \
    --wrap="cd $REPO_ROOT && \
        $CONDA_PYTHON perf_router/eval/eval_perf_router_from_preds.py \
            --preds $PREDS \
            --out-dir $RESULTS_DIR \
            --qid-filter $QID_FILTER"

echo "Submitted eval_perf_rc_small4 (CPU). Logs: $SLURM_LOGS/eval_perf_rc_small4_<jobid>.out"
echo "Results: $RESULTS_DIR/"
echo "  Full-task: $RESULTS_DIR/summary.json"
echo "  Subtask:   $RESULTS_DIR/subtask_summary.json"
