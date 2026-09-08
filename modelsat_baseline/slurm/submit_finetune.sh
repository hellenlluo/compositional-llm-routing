#!/usr/bin/env bash
# Submit the Model-SAT fine-tuning job to SLURM.
# Usage: bash modelsat_baseline/submit_finetune.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/modelsat_baseline/slurm-logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

sbatch \
    --job-name=modelsat-finetune \
    --account=pvl \
    --gres=gpu:a6000:1 \
    --time=8:00:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --output="$SLURM_LOGS/finetune_%j.out" \
    --error="$SLURM_LOGS/finetune_%j.out" \
    --wrap="cd $REPO_ROOT/modelsat_baseline && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        $CONDA_PYTHON finetune_router.py \
            --stage1-epochs 2 \
            --stage2-epochs 0 \
            --batch-k 8 \
            --batch-groups 4 \
            --lr-connector 1e-3 \
            --lr-encoder 1e-4 \
            --lr-llm 1e-5 \
            --ckpt-name model_sat_stage1 \
            --ckpt-dir $REPO_ROOT/modelsat_baseline/router_checkpoint_v2"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: $SLURM_LOGS/finetune_<jobid>.out"
