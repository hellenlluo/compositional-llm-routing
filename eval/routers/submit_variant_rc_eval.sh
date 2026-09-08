#!/usr/bin/env bash
# Compute R+C metrics for SIMROUTE Variant B (example) and Variant C (strength_prior).
# CPU-only — reads existing output files, no GPU needed.
#
# Prerequisites:
#   bash eval/routers/submit_variants_bc.sh  (jobs must have finished)
#
# Usage:
#   bash eval/routers/submit_variant_rc_eval.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="$REPO_ROOT/eval/routers/logs"
mkdir -p "$LOGS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
ENV_VARS="XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface"

for PREFIX in example strength_prior; do
    for MODEL_SET in no_qwen3 small4; do
        JOB="variantRC-${PREFIX}-${MODEL_SET}"
        sbatch \
            --job-name="$JOB" \
            --account=pvl \
            --partition=pvl \
            --time=0:10:00 \
            --mem=8G \
            --cpus-per-task=2 \
            --output="$LOGS_DIR/${JOB}_%j.out" \
            --error="$LOGS_DIR/${JOB}_%j.out" \
            --wrap="cd $REPO_ROOT && $ENV_VARS \
                $CONDA_PYTHON -m eval.routers.variant_rc_eval \
                    --prefix $PREFIX \
                    --model-set $MODEL_SET"
        echo "  Submitted $JOB"
    done
done

echo ""
echo "Results will be written to:"
echo "  outputs/routing/{no_qwen3,small4}/example_rc_summary.json"
echo "  outputs/routing/{no_qwen3,small4}/strength_prior_rc_summary.json"
