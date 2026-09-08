#!/usr/bin/env bash
# Chain N additional Stage 2 epochs as SLURM dependencies.
# Each epoch job starts immediately when the previous one finishes,
# resuming from the latest checkpoint.  The cosine LR schedule is
# stretched across the full planned training run so LR decays smoothly
# instead of resetting to zero at the start of every chained job.
#
# Usage:
#   bash modelsat_baseline/submit_stage2_chain.sh <current_job_id> <n_additional_epochs>
#
# Example — epoch 1 is running as job 28543245; chain 4 more epochs:
#   bash modelsat_baseline/submit_stage2_chain.sh 28543245 4
#
# The script prints each submitted job ID and the starting LR for that epoch.
#
# LR maths
# --------
# Epoch 1 ran with --batch-groups 2  →  7070 steps,  T_max=7070 (LR 1e-3 → 0).
# Epochs 2-N run with --batch-groups 4  →  ceil(14139/4)=3535 steps each.
# Total planned steps = 7070 + N_additional × 1768.
# We pass --scheduler-t-max=<total> to every chained job so the cosine
# curve spans the full training run:
#
#   Epoch 2 start LR  ≈ lr_max × ½(1 + cos(π × 7070/T_max))
#   Epoch 5 end   LR  → 0   (if 4 additional epochs)
#
# Adjust EPOCH1_STEPS if you re-run epoch 1 with different batch-groups.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/modelsat_baseline/slurm-logs"
CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

PREV_JOB="${1:?Usage: $0 <current_job_id> <n_additional_epochs>}"
N_ADDITIONAL="${2:?Usage: $0 <current_job_id> <n_additional_epochs>}"

# Checkpoint written by epoch 1 and overwritten by every subsequent epoch.
CKPT_DIR="$REPO_ROOT/modelsat_baseline/router_checkpoint_v2"
CKPT_FILE="$CKPT_DIR/model_sat_stage2.pt"

# Scheduler steps already completed (saved in checkpoint last_epoch).
# Job 28543245 ran batch-groups=2 and was checkpointed mid-epoch at step 5500.
EPOCH1_STEPS=5500
# Steps per epoch for chained jobs (batch-groups=4 → ceil(14139/4) = 3535).
# Reduced from 6 to 4 after OOM at step 3 with batch-groups=6 on the A6000:
# the GPU was at 46.08 GiB / 47.40 GiB total with only 1.31 GiB free but
# needed 1.47 GiB. batch-groups=4 cuts per-step activation memory ~33%.
STEPS_PER_EPOCH=3535
# Total planned optimizer steps across all epochs.
TOTAL_STEPS=$(( EPOCH1_STEPS + N_ADDITIONAL * STEPS_PER_EPOCH ))

echo "Chaining ${N_ADDITIONAL} epoch(s) after job ${PREV_JOB}"
echo "Checkpoint : ${CKPT_FILE}"
echo "T_max      : ${EPOCH1_STEPS} + ${N_ADDITIONAL}×${STEPS_PER_EPOCH} = ${TOTAL_STEPS} steps"
echo ""

for i in $(seq 1 "$N_ADDITIONAL"); do
    EPOCH_NUM=$(( i + 1 ))

    # Approximate starting LR for this epoch (connector param group).
    # cos(π × cumulative_steps / T_max) at the start of this epoch.
    STEPS_SO_FAR=$(( EPOCH1_STEPS + (i - 1) * STEPS_PER_EPOCH ))
    # Python one-liner: print LR as a float for the log message.
    START_LR=$(python3 -c "
import math
t = ${STEPS_SO_FAR}; T = ${TOTAL_STEPS}; lr = 1e-3
print(f'{lr * 0.5 * (1 + math.cos(math.pi * t / T)):.2e}')
" 2>/dev/null || echo "~?")

    JOB_ID=$(sbatch \
        --job-name=test \
        --account=pvl \
        --gres=gpu:a6000:1 \
        --time=8:00:00 \
        --mem=32G \
        --cpus-per-task=2 \
        --dependency=afterany:${PREV_JOB} \
        --kill-on-invalid-dep=yes \
        --output="${SLURM_LOGS}/stage2_%j.out" \
        --error="${SLURM_LOGS}/stage2_%j.out" \
        --parsable \
        --wrap="cd $REPO_ROOT/modelsat_baseline && \
            HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
            HF_HUB_DISABLE_XET=1 \
            PYTORCH_ALLOC_CONF=expandable_segments:True \
            $CONDA_PYTHON finetune_router.py \
                --stage1-epochs 0 \
                --stage2-epochs 1 \
                --batch-k 4 \
                --batch-groups 4 \
                --lr-connector 1e-3 \
                --lr-encoder 1e-4 \
                --lr-llm 1e-5 \
                --scheduler-t-max ${TOTAL_STEPS} \
                --ckpt-name model_sat_stage2 \
                --ckpt-dir ${CKPT_DIR} \
                --resume-from ${CKPT_FILE}")

    echo "  Epoch ${EPOCH_NUM}: job ${JOB_ID}  (depends on ${PREV_JOB}, connector LR start ≈ ${START_LR})"
    PREV_JOB=$JOB_ID
done

echo ""
echo "Chain submitted. Final job: ${PREV_JOB}"
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${SLURM_LOGS}/stage2_<jobid>.out"
