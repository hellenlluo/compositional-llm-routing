#!/usr/bin/env bash
# Submit the Model-SAT model download job to SLURM.
# Usage: bash modelsat_baseline/submit_download_modelsat_models.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/modelsat_baseline/slurm-logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

sbatch \
    --job-name=dl-modelsat-models \
    --account=pvl \
    --time=02:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --output="$SLURM_LOGS/download_modelsat_models_%j.out" \
    --error="$SLURM_LOGS/download_modelsat_models_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_CACHE=/n/fs/scratch/dl3533/.cache/huggingface/hub \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON modelsat_baseline/download_modelsat_models.py"

echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: $SLURM_LOGS/download_modelsat_models_<jobid>.out"
