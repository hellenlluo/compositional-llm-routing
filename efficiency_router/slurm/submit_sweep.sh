#!/usr/bin/env bash
# Sweep --offset values: trains one EfficiencyRouter per value, evaluates each,
# and collects results under efficiency_router/results/sweep/.
# Runs as 2 parallel jobs: offsets 1-5 and 6-10.
#
# Usage:
#   bash efficiency_router/submit_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
SWEEP_DIR="$SCRIPT_DIR/results/sweep"
mkdir -p "$LOGS_DIR" "$SWEEP_DIR"

sbatch \
    --job-name=effrouter-sweep-a \
    --account=allcs \
    --gres=gpu:1 \
    --time=30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/sweep_a_%j.out" \
    --error="$LOGS_DIR/sweep_a_%j.out" \
    "$SCRIPT_DIR/run_sweep.sh" "$SCRIPT_DIR" "$SWEEP_DIR" 1 2 3 4 5

sbatch \
    --job-name=effrouter-sweep-b \
    --account=allcs \
    --gres=gpu:1 \
    --time=30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/sweep_b_%j.out" \
    --error="$LOGS_DIR/sweep_b_%j.out" \
    "$SCRIPT_DIR/run_sweep.sh" "$SCRIPT_DIR" "$SWEEP_DIR" 6 7 8 9 10

echo "Submitted 2 sweep jobs (offsets 1-5 and 6-10)."
echo "Logs:    $LOGS_DIR/sweep_a_<jobid>.out  /  sweep_b_<jobid>.out"
echo "Results: $SWEEP_DIR/"
