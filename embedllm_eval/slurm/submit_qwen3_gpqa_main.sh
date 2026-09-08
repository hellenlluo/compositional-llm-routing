#!/usr/bin/env bash
# Resubmit the 4 gpqa_main categories that failed for qwen3-4b-thinking-2507
# due to the judge's context window being too small (now fixed to 16384).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
mkdir -p "$SLURM_LOGS"

MODEL_ID="qwen3-4b-thinking-2507"
CATEGORIES=(
    "gpqa_main_cot_zeroshot"
    "gpqa_main_generative_n_shot"
    "gpqa_main_n_shot"
    "gpqa_main_zeroshot"
)

for CATEGORY in "${CATEGORIES[@]}"; do
    JOB_NAME="emb-qwen3-4b-${CATEGORY:0:20}"
    LOG_FILE="$SLURM_LOGS/${MODEL_ID}__${CATEGORY}_%j.out"

    CMD="cd $REPO_ROOT \
        && HF_HUB_OFFLINE=1 \
        XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
        /n/fs/scratch/dl3533/conda_envs/MARIO/bin/python embedllm_eval/run_one_job.py \
        --model-id $MODEL_ID \
        --category $CATEGORY \
        --backend auto \
        --batch-size 3"

    sbatch \
        --job-name="$JOB_NAME" \
        --partition=all \
        --account=allcs \
        --gres=gpu:a6000:2 \
        --time=08:00:00 \
        --mem=128G \
        --cpus-per-task=8 \
        --output="$LOG_FILE" \
        --error="$LOG_FILE" \
        --wrap="$CMD"

    echo "Submitted: $MODEL_ID / $CATEGORY"
done

echo ""
echo "Monitor with: squeue -u \$USER"
