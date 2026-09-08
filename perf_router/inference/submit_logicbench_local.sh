#!/usr/bin/env bash
# Run LogicBench inference locally (vLLM) for models that have no API provider.
# Currently: mathstral-7b
#
# Usage:
#   bash perf_router/submit_logicbench_local.sh [model_id]
#
# Examples:
#   bash perf_router/submit_logicbench_local.sh mathstral-7b
#   bash perf_router/submit_logicbench_local.sh   # defaults to mathstral-7b

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/perf_router/logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
MODEL="${1:-mathstral-7b}"
shift 2>/dev/null || true   # consume $1 so remaining args can be forwarded
EXTRA_ARGS="${*}"

sbatch \
    --job-name="lb_${MODEL}" \
    --account=pvl \
    --gres=gpu:rtx_3090:1 \
    --time=4:00:00 \
    --mem=32G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/lb_${MODEL}_%j.out" \
    --error="$SLURM_LOGS/lb_${MODEL}_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        $CONDA_PYTHON perf_router/run_logicbench.py \
            --backend vllm \
            --model ${MODEL} \
            $EXTRA_ARGS"

echo "Submitted lb_${MODEL}. Monitor with: squeue -u \$USER"
echo "Logs: $SLURM_LOGS/lb_${MODEL}_<jobid>.out"
