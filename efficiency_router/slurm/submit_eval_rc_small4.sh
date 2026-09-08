#!/usr/bin/env bash
# Derive EfficiencyRouter R+C metrics from existing predictions — no GPU needed.
# Full-task + overall subtask saved to efficiency_router/results/rc_questions/small4/
#
# Prerequisites:
#   1. python router_analysis/generate_rc_qids.py   (outputs/rc_qids.json)
#   2. bash efficiency_router/submit_eval_small4.sh  (produces predictions.json)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
PREDS="$SCRIPT_DIR/results/small4/predictions.json"
RESULTS_DIR="$SCRIPT_DIR/results/rc_questions/small4"
QID_FILTER="$REPO_ROOT/outputs/rc_qids.json"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

sbatch \
    --job-name=effrouter-rc-small4 \
    --account=allcs \
    --partition=pvl \
    --time=0:10:00 \
    --mem=8G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/eval_rc_small4_%j.out" \
    --error="$LOGS_DIR/eval_rc_small4_%j.out" \
    --wrap="cd $SCRIPT_DIR && \
        $CONDA_PYTHON eval_from_preds.py \
            --preds $PREDS \
            --out-dir $RESULTS_DIR \
            --qid-filter $QID_FILTER"
echo "Submitted effrouter-rc-small4 (CPU). Logs: $LOGS_DIR/eval_rc_small4_<jobid>.out"
echo "Results: $RESULTS_DIR/"
echo "  Full-task: $RESULTS_DIR/eval_results.json"
echo "  Subtask:   $RESULTS_DIR/subtask_results.json"
echo "  (Chained subtask: run submit_eval_rc_small4_chained.sh — requires GPU)"

