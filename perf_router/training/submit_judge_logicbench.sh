#!/usr/bin/env bash
# Judge LogicBench responses using Claude Haiku on AWS Bedrock.
# Requires AWS credentials to be set (see judge_logicbench.py).
#
# Usage:
#   bash perf_router/submit_judge_logicbench.sh [model_id]
#   bash perf_router/submit_judge_logicbench.sh --all-models
#
# Examples:
#   bash perf_router/submit_judge_logicbench.sh qwen3-4b-thinking-2507
#   bash perf_router/submit_judge_logicbench.sh --all-models

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/perf_router/logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
MODEL_ARG="${1:-qwen3-4b-thinking-2507}"

# Build the python argument: --all-models flag passes straight through,
# otherwise wrap as --model <name>
if [[ "$MODEL_ARG" == "--all-models" ]]; then
    PY_ARGS="--all-models"
    JOB_NAME="judge_lb_all"
else
    PY_ARGS="--model ${MODEL_ARG}"
    JOB_NAME="judge_lb_${MODEL_ARG}"
fi

sbatch \
    --job-name="$JOB_NAME" \
    --account=pvl \
    --time=1:00:00 \
    --mem=8G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/${JOB_NAME}_%j.out" \
    --error="$SLURM_LOGS/${JOB_NAME}_%j.out" \
    --wrap="cd $REPO_ROOT && \
        AWS_ACCESS_KEY_ID=\$AWS_ACCESS_KEY_ID \
        AWS_SECRET_ACCESS_KEY=\$AWS_SECRET_ACCESS_KEY \
        AWS_DEFAULT_REGION=\${AWS_DEFAULT_REGION:-us-east-1} \
        $CONDA_PYTHON perf_router/judge_logicbench.py \
            $PY_ARGS \
            --workers 30"

echo "Submitted $JOB_NAME."
echo "After it completes, re-run build_perf_vectors.py to update perf_vectors.json."
echo "Logs: $SLURM_LOGS/${JOB_NAME}_<jobid>.out"
