#!/usr/bin/env bash
# Two-phase circle test submission.
#
# Phase 1 (--phase inference): 10 small-GPU jobs, one per model.
#   Each job runs model inference and saves raw responses to data/responses/.
#   Any GPU with ~11+ GB works (RTX 2080 Ti, 3090, etc.).
#
# Phase 2 (--phase judge): 1 job loads all saved responses and runs the
#   32B judge. Needs 2x A6000 (48 GB each) via -A pvl.
#
# Usage:
#   bash submit_circle_test.sh inference   # submit phase 1
#   bash submit_circle_test.sh judge       # submit phase 2 (after phase 1 finishes)
#   bash submit_circle_test.sh             # submit phase 1, then auto-chain phase 2

set -euo pipefail

PHASE="${1:-both}"   # inference | judge | both

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/modelsat_baseline/slurm-logs"
mkdir -p "$SLURM_LOGS"

PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

MODELS=(
    # Comment out models already done; leave all for a fresh run
    # qwen1.5-0.5b-chat
    # deepseek-r1-distill-llama-8b
    # llama-3.1-8b-instruct
    # llama-3.1-nemotron-nano-8b
    # mathstral-7b
    # medgemma-4b-it
    # mistral-7b-instruct-v0.3
    # phi-4-mini-instruct
    # qwen3-30b-a3b
    # qwen3-4b-thinking-2507
)

# ---------------------------------------------------------------------------
# Phase 1: inference (small GPUs, one job per model)
# ---------------------------------------------------------------------------
submit_inference() {
    local INFERENCE_JOB_IDS=()

    for MODEL in "${MODELS[@]}"; do
        CMD="cd $REPO_ROOT \
            && HF_HUB_OFFLINE=1 \
            XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
            PYTORCH_ALLOC_CONF=expandable_segments:True \
            $PYTHON modelsat_baseline/circle_test.py \
            --model $MODEL \
            --phase inference \
            --backend hf \
            --batch-size 4"

        # qwen3-30b-a3b needs 2 A6000s (~60 GB); everything else fits on 1
        if [[ "$MODEL" == "qwen3-30b-a3b" ]]; then
            N_GPUS=2
            MEM=128G
        else
            N_GPUS=1
            MEM=64G
        fi

        JOB_ID=$(sbatch \
            --job-name="ci-inf-${MODEL}" \
            --account=pvl \
            --gres=gpu:a6000:$N_GPUS \
            --time=01:00:00 \
            --mem=$MEM \
            --cpus-per-task=8 \
            --output="$SLURM_LOGS/circle_inf_${MODEL}_%j.out" \
            --error="$SLURM_LOGS/circle_inf_${MODEL}_%j.out" \
            --parsable \
            --wrap="$CMD")

        INFERENCE_JOB_IDS+=("$JOB_ID")
        echo "  [inference] $MODEL → job $JOB_ID" >&2
    done

    # Only job IDs go to stdout so callers can capture them cleanly
    echo "${INFERENCE_JOB_IDS[@]}"
}

# ---------------------------------------------------------------------------
# Phase 2: judge (2x A6000 via pvl, single job over all saved responses)
# ---------------------------------------------------------------------------
submit_judge() {
    local DEPENDENCY="${1:-}"   # e.g. "afterok:123:456:789"

    CMD="cd $REPO_ROOT \
        && HF_HUB_OFFLINE=1 \
        XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        $PYTHON modelsat_baseline/circle_test.py \
        --phase judge \
        --backend hf \
        --judge-batch-size 1"

    SBATCH_ARGS=(
        --job-name="ci-judge"
        --account=pvl
        --gres=gpu:a6000:2
        --time=02:00:00
        --mem=128G
        --cpus-per-task=8
        --output="$SLURM_LOGS/circle_judge_%j.out"
        --error="$SLURM_LOGS/circle_judge_%j.out"
        --parsable
    )
    [[ -n "$DEPENDENCY" ]] && SBATCH_ARGS+=(--dependency="$DEPENDENCY")

    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" --wrap="$CMD")
    echo "  [judge] job $JOB_ID${DEPENDENCY:+ (depends on: $DEPENDENCY)}"
    echo "$JOB_ID"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
if [[ "$PHASE" == "inference" ]]; then
    echo "Submitting phase 1 (inference) jobs..."
    submit_inference > /dev/null
    echo "Done. Monitor with: squeue -u \$USER"
    echo "When all finish, run: bash submit_circle_test.sh judge"

elif [[ "$PHASE" == "judge" ]]; then
    echo "Submitting phase 2 (judge) job..."
    submit_judge
    echo "Done. Monitor with: squeue -u \$USER"
    echo "When finished, merge results with: python modelsat_baseline/merge_circle_test.py"

else
    echo "Submitting phase 1 (inference) jobs..."
    IDS=($(submit_inference))
    echo ""

    # Build dependency string: afterok:id1:id2:...
    DEP="afterok:$(IFS=:; echo "${IDS[*]}")"
    echo "Submitting phase 2 (judge) job with dependency on all inference jobs..."
    submit_judge "$DEP"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "When judge finishes, merge results with: python modelsat_baseline/merge_circle_test.py"
fi
