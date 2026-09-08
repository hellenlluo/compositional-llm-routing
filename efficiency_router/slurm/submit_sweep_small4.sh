#!/usr/bin/env bash
# Sweep --offset values for the small4 EfficiencyRouter (offsets 1-10).
# Runs as 2 parallel jobs: offsets 1-5 and 6-10.
# Results saved to efficiency_router/results/sweep_small4/
#
# Usage:
#   bash efficiency_router/submit_sweep_small4.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/efficiency_router"
LOGS_DIR="$SCRIPT_DIR/logs"
SWEEP_DIR="$SCRIPT_DIR/results/sweep_small4"
mkdir -p "$LOGS_DIR" "$SWEEP_DIR"

sbatch \
    --job-name=effrouter-sweep-s4-a \
    --account=allcs \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=1:00:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/sweep_small4_a_%j.out" \
    --error="$LOGS_DIR/sweep_small4_a_%j.out" \
    "$SCRIPT_DIR/run_sweep_small4.sh" "$SCRIPT_DIR" "$SWEEP_DIR" 1 2 3 4 5

sbatch \
    --job-name=effrouter-sweep-s4-b \
    --account=allcs \
    --partition=pvl \
    --gres=gpu:a6000:1 \
    --time=1:00:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$LOGS_DIR/sweep_small4_b_%j.out" \
    --error="$LOGS_DIR/sweep_small4_b_%j.out" \
    "$SCRIPT_DIR/run_sweep_small4.sh" "$SCRIPT_DIR" "$SWEEP_DIR" 6 7 8 9 10

echo "Submitted 2 sweep jobs (offsets 1-5 and 6-10, small4)."
echo "Logs:    $LOGS_DIR/sweep_small4_a_<jobid>.out  /  sweep_small4_b_<jobid>.out"
echo "Results: $SWEEP_DIR/"
