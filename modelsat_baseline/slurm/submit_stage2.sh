#!/usr/bin/env bash
# Submit the Model-SAT Stage 2 fine-tuning job to SLURM.
# Resumes from a checkpoint and trains all parameters on the full dataset.
# Each job saves optimizer + scheduler state so LR continues smoothly.
#
# 5-epoch chain (1 epoch per job):
#   Job 1: bash submit_stage2.sh router_checkpoint_v2/model_sat_stage1.pt
#   Job 2: bash submit_stage2.sh router_checkpoint_stage2_fixed/model_sat_stage2.pt
#   Job 3: bash submit_stage2.sh router_checkpoint_stage2_fixed/model_sat_stage2.pt
#   Job 4: bash submit_stage2.sh router_checkpoint_stage2_fixed/model_sat_stage2.pt
#   Job 5: bash submit_stage2.sh router_checkpoint_stage2_fixed/model_sat_stage2.pt
#   (each job overwrites router_checkpoint_stage2_fixed/model_sat_stage2.pt)
#
# Usage: bash modelsat_baseline/submit_stage2.sh [path/to/checkpoint.pt] [epochs_this_job]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/modelsat_baseline/slurm-logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

# Optional checkpoint to resume from; omit for a fresh run
RESUME_FROM="${1:-}"
# Epochs to run in this job (default 1)
EPOCHS_THIS_JOB="${2:-1}"

RESUME_ARG=""
if [[ -n "$RESUME_FROM" ]]; then
    if [[ ! -f "$RESUME_FROM" ]]; then
        echo "[ERROR] Checkpoint not found: $RESUME_FROM"
        exit 1
    fi
    echo "Resuming from: $RESUME_FROM"
    RESUME_ARG="--resume-from $RESUME_FROM"
else
    echo "Starting fresh (no checkpoint)"
fi

sbatch \
    --job-name=modelsat-s2 \
    --account=pvl \
    --gres=gpu:a6000:1 \
    --time=8:00:00 \
    --mem=32G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/stage2_%j.out" \
    --error="$SLURM_LOGS/stage2_%j.out" \
    --wrap="cd $REPO_ROOT/modelsat_baseline && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        $CONDA_PYTHON finetune_router.py \
            --stage1-epochs 0 \
            --stage2-epochs $EPOCHS_THIS_JOB \
            --batch-k 4 \
            --batch-groups 6 \
            --lr-connector 1e-3 \
            --lr-encoder 1e-4 \
            --lr-llm 1e-5 \
            --ckpt-name model_sat_stage2 \
            --ckpt-dir $REPO_ROOT/modelsat_baseline/router_checkpoint_stage2_fixed \
            $RESUME_ARG"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: $SLURM_LOGS/stage2_<jobid>.out"
