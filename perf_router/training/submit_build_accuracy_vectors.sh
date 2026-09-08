#!/usr/bin/env bash
# Build per-model accuracy vectors from all benchmark result sources.
# This is a lightweight CPU job; no GPU required.
#
# Usage:
#   bash perf_router/submit_build_accuracy_vectors.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_LOGS="$REPO_ROOT/perf_router/logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

sbatch \
    --job-name="build_acc_vecs" \
    --account=pvl \
    --time=0:30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/build_acc_vecs_%j.out" \
    --error="$SLURM_LOGS/build_acc_vecs_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON perf_router/training/build_accuracy_vectors.py"

echo "Submitted build_accuracy_vectors. Logs: $SLURM_LOGS/build_acc_vecs_<jobid>.out"
