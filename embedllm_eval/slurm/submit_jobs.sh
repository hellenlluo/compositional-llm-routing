#!/usr/bin/env bash
# Submit one SLURM job per category for the EmbedLLM evaluation.
#
# Usage:
#   bash embedllm_eval/submit_jobs.sh                    # submit all categories
#   bash embedllm_eval/submit_jobs.sh --dry-run          # print commands without submitting
#   bash embedllm_eval/submit_jobs.sh --partition gpu     # custom partition
#
# Each job evaluates ALL 8 models on the questions in one category.
# After all jobs complete, run:
#   python embedllm_eval/combine_results.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
mkdir -p "$SLURM_LOGS"

# Defaults — override via env or arguments
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
TIME="${TIME:-04:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-4}"
BACKEND="${BACKEND:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --gres) GRES="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Get all category names from the dataset
CATEGORIES=$(python3 -c "
import json
cats = set()
with open('$REPO_ROOT/data_preprocessing/embedllm-sampled/prompts_subset.jsonl') as f:
    for line in f:
        cats.add(json.loads(line)['category'])
for c in sorted(cats):
    print(c)
")

N_CATS=$(echo "$CATEGORIES" | wc -l | tr -d ' ')
echo "=== EmbedLLM Evaluation: submitting $N_CATS category jobs ==="
echo "  partition=$PARTITION  gres=$GRES  time=$TIME  mem=$MEM"
echo "  backend=$BACKEND  batch_size=$BATCH_SIZE"
echo ""

SUBMITTED=0
for CAT in $CATEGORIES; do
    JOB_NAME="embedllm-${CAT}"
    LOG_FILE="$SLURM_LOGS/${CAT}_%j.out"

    SBATCH_CMD="sbatch \
        --job-name=$JOB_NAME \
        --partition=$PARTITION \
        --gres=$GRES \
        --time=$TIME \
        --mem=$MEM \
        --cpus-per-task=$CPUS \
        --output=$LOG_FILE \
        --error=$LOG_FILE \
        --wrap=\"cd $REPO_ROOT && python embedllm_eval/run_category.py \
            --category $CAT \
            --backend $BACKEND \
            --batch-size $BATCH_SIZE \
            --max-new-tokens $MAX_NEW_TOKENS\""

    if $DRY_RUN; then
        echo "[DRY-RUN] $SBATCH_CMD"
    else
        eval "$SBATCH_CMD"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete. $N_CATS jobs would be submitted."
else
    echo "Submitted $SUBMITTED jobs."
    echo "Monitor with: squeue -u \$USER"
    echo "After completion, run: python embedllm_eval/combine_results.py"
fi
