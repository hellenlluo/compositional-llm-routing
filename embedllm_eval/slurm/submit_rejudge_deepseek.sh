#!/usr/bin/env bash
# Submit deepseek-r1-distill-llama-8b rejudge jobs in batches.
# Submits MAX_JOBS at a time, waits for all to finish, then submits the next batch.
#
# Usage:
#   bash embedllm_eval/submit_rejudge_deepseek.sh               # run all batches of 5
#   bash embedllm_eval/submit_rejudge_deepseek.sh --max-jobs 10 # batches of 10
#   bash embedllm_eval/submit_rejudge_deepseek.sh --dry-run     # preview first batch only
#   bash embedllm_eval/submit_rejudge_deepseek.sh --category mmlu_marketing  # one category
#
# After all batches are done, run:
#   python embedllm_eval/stitch_results.py

set -euo pipefail

MODEL_ID="deepseek-r1-distill-llama-8b"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/embedllm_eval/results/$MODEL_ID"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
mkdir -p "$SLURM_LOGS"

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
ACCOUNT="${ACCOUNT:-pvl}"
GRES="${GRES:-gpu:a6000:2}"  # must specify GPU type — gtx/rtx nodes fail (too old/small for bfloat16)
                              # other valid options: gpu:l40:2  gpu:a40:2  gpu:a100:2
TIME="${TIME:-02:00:00}"
MEM="${MEM:-128G}"
CPUS="${CPUS:-8}"
BACKEND="${BACKEND:-auto}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-1}"
MAX_JOBS="${MAX_JOBS:-5}"
POLL_INTERVAL=60   # seconds between squeue checks
FILTER_CATEGORY=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)          DRY_RUN=true; shift ;;
        --account)          ACCOUNT="$2"; shift 2 ;;
        --gres)             GRES="$2"; shift 2 ;;
        --time)             TIME="$2"; shift 2 ;;
        --mem)              MEM="$2"; shift 2 ;;
        --backend)          BACKEND="$2"; shift 2 ;;
        --judge-batch-size) JUDGE_BATCH_SIZE="$2"; shift 2 ;;
        --max-jobs)         MAX_JOBS="$2"; shift 2 ;;
        --poll-interval)    POLL_INTERVAL="$2"; shift 2 ;;
        --category)         FILTER_CATEGORY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

submit_job() {
    local CATEGORY="$1"
    local JOB_NAME="rejudge-${CATEGORY:0:28}"
    local LOG_FILE="$SLURM_LOGS/rejudge_${MODEL_ID}__${CATEGORY}_%j.out"

    local CMD="cd $REPO_ROOT \
        && HF_HUB_OFFLINE=1 \
        XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
        /n/fs/scratch/dl3533/conda_envs/MARIO/bin/python embedllm_eval/rejudge_one_category.py \
        --category $CATEGORY \
        --backend $BACKEND \
        --judge-batch-size $JUDGE_BATCH_SIZE"

    local SBATCH_CMD="sbatch \
        --job-name=$JOB_NAME \
        --account=$ACCOUNT \
        --gres=$GRES \
        --time=$TIME \
        --mem=$MEM \
        --cpus-per-task=$CPUS \
        --output=$LOG_FILE \
        --error=$LOG_FILE \
        --parsable \
        --wrap=\"$CMD\""

    eval "$SBATCH_CMD"  # prints job ID only (--parsable)
}

wait_for_jobs() {
    local -a JOB_IDS=("$@")
    echo "  Waiting for jobs: ${JOB_IDS[*]}"
    while true; do
        local STILL_RUNNING=0
        for JID in "${JOB_IDS[@]}"; do
            if squeue -j "$JID" -h &>/dev/null; then
                STILL_RUNNING=$(( STILL_RUNNING + 1 ))
            fi
        done
        if [[ $STILL_RUNNING -eq 0 ]]; then
            break
        fi
        echo "  $(date '+%H:%M:%S')  $STILL_RUNNING / ${#JOB_IDS[@]} jobs still running..."
        sleep "$POLL_INTERVAL"
    done
    echo "  Batch complete."
}

# --------------------------------------------------------------------------
# Collect categories
# --------------------------------------------------------------------------
if [[ -n "$FILTER_CATEGORY" ]]; then
    ALL_CATEGORIES="$FILTER_CATEGORY"
else
    ALL_CATEGORIES=$(
        find "$RESULTS_DIR" -maxdepth 1 -name "*.json" -printf "%f\n" \
        | sed 's/\.json$//' \
        | sort
    )
fi

N_TOTAL=$(echo "$ALL_CATEGORIES" | grep -c . || true)
N_BATCHES=$(( (N_TOTAL + MAX_JOBS - 1) / MAX_JOBS ))

echo "=== Re-judging $MODEL_ID ==="
echo "  $N_TOTAL categories  |  batch size $MAX_JOBS  |  $N_BATCHES batches"
echo "  account=$ACCOUNT  gres=$GRES  time=$TIME  mem=$MEM"
echo ""

if $DRY_RUN; then
    echo "[DRY-RUN] Would submit in $N_BATCHES batches of up to $MAX_JOBS:"
    BATCH=1
    COUNT=0
    for CAT in $ALL_CATEGORIES; do
        if [[ $(( COUNT % MAX_JOBS )) -eq 0 ]]; then
            echo "  --- Batch $BATCH ---"
            BATCH=$(( BATCH + 1 ))
        fi
        echo "    $CAT"
        COUNT=$(( COUNT + 1 ))
    done
    exit 0
fi

# --------------------------------------------------------------------------
# Submit batches and wait
# --------------------------------------------------------------------------
BATCH=1
CATS=()
for CAT in $ALL_CATEGORIES; do
    CATS+=("$CAT")
done

IDX=0
while [[ $IDX -lt ${#CATS[@]} ]]; do
    BATCH_CATS=("${CATS[@]:$IDX:$MAX_JOBS}")
    echo "--- Batch $BATCH / $N_BATCHES (categories $((IDX+1))–$((IDX+${#BATCH_CATS[@]})) of $N_TOTAL) ---"

    JOB_IDS=()
    for CAT in "${BATCH_CATS[@]}"; do
        JID=$(submit_job "$CAT")
        echo "  submitted $CAT → job $JID"
        JOB_IDS+=("$JID")
    done

    wait_for_jobs "${JOB_IDS[@]}"
    echo ""

    IDX=$(( IDX + MAX_JOBS ))
    BATCH=$(( BATCH + 1 ))
done

echo "=== All $N_BATCHES batches complete. ==="
echo "Run: python embedllm_eval/stitch_results.py"
