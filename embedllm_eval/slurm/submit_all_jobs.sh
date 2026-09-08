#!/usr/bin/env bash
# Submit one SLURM job per (model, category) pair for the EmbedLLM evaluation.
# Each job runs run_one_job.py: inference with the candidate model, then
# judging with Qwen2.5-32B. Results land in embedllm_eval/results/<model>/<category>.json
#
# Usage:
#   bash embedllm_eval/submit_all_jobs.sh                        # submit all
#   bash embedllm_eval/submit_all_jobs.sh --dry-run              # preview without submitting
#   bash embedllm_eval/submit_all_jobs.sh --model qwen3-4b-thinking-2507  # one model only
#   bash embedllm_eval/submit_all_jobs.sh --category gpqa_main_n_shot     # one category only
#
# After all jobs finish:
#   python embedllm_eval/stitch_results.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
mkdir -p "$SLURM_LOGS"

# --------------------------------------------------------------------------
# Defaults — override via env vars or CLI flags
# --------------------------------------------------------------------------
PARTITION="${PARTITION:-all}"
ACCOUNT="${ACCOUNT:-allcs}"
GRES="${GRES:-gpu:a6000:2}"  # must specify GPU type — gtx/rtx nodes fail (too old/small for bfloat16)
                              # other valid options: gpu:l40:2  gpu:a40:2  gpu:a100:2
TIME="${TIME:-04:00:00}"
MEM="${MEM:-128G}"
CPUS="${CPUS:-8}"
BACKEND="${BACKEND:-auto}"
BATCH_SIZE="${BATCH_SIZE:-3}"
FILTER_MODEL=""
FILTER_CATEGORY=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --account)    ACCOUNT="$2"; shift 2 ;;
        --gres)       GRES="$2"; shift 2 ;;
        --time)       TIME="$2"; shift 2 ;;
        --mem)        MEM="$2"; shift 2 ;;
        --backend)    BACKEND="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --model)      FILTER_MODEL="$2"; shift 2 ;;
        --category)   FILTER_CATEGORY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Models list (must match models.yaml ids)
# --------------------------------------------------------------------------
ALL_MODELS=(
    "qwen3-4b-thinking-2507"
    "deepseek-r1-distill-llama-8b"
    "medgemma-4b-it"
    "llama-3.1-8b-instruct"
    "qwen3-30b-a3b"
    "phi-4-mini-instruct"
    "mistral-7b-instruct-v0.3"
    "qwen1.5-0.5b-chat"
    "mathstral-7b-v0.1"
    "llama-3.1-nemotron-nano-8b-v1"
)

if [[ -n "$FILTER_MODEL" ]]; then
    ALL_MODELS=("$FILTER_MODEL")
fi

# --------------------------------------------------------------------------
# Discover categories from the dataset
# --------------------------------------------------------------------------
CATEGORIES=$(python3 -c "
import json
cats = set()
with open('$REPO_ROOT/data_preprocessing/embedllm-sampled/prompts_subset.jsonl') as f:
    for line in f:
        cats.add(json.loads(line)['category'])
for c in sorted(cats):
    print(c)
")

if [[ -n "$FILTER_CATEGORY" ]]; then
    CATEGORIES="$FILTER_CATEGORY"
fi

N_MODELS=${#ALL_MODELS[@]}
N_CATS=$(echo "$CATEGORIES" | wc -l | tr -d ' ')
N_JOBS=$(( N_MODELS * N_CATS ))

echo "=== EmbedLLM Evaluation — submitting ${N_MODELS} models × ${N_CATS} categories = ${N_JOBS} jobs ==="
echo "  partition=${PARTITION}  account=${ACCOUNT}  gres=${GRES}  time=${TIME}  mem=${MEM}"
echo "  backend=${BACKEND}  batch_size=${BATCH_SIZE}"
echo ""

SUBMITTED=0
SKIPPED=0

for MODEL_ID in "${ALL_MODELS[@]}"; do
    for CATEGORY in $CATEGORIES; do
        # Skip if result already exists and is non-empty
        RESULT_FILE="$REPO_ROOT/embedllm_eval/results/$MODEL_ID/$CATEGORY.json"
        if [[ -f "$RESULT_FILE" && -s "$RESULT_FILE" ]]; then
            echo "[SKIP] $MODEL_ID / $CATEGORY — result already exists"
            SKIPPED=$(( SKIPPED + 1 ))
            continue
        fi

        JOB_NAME="emb-${MODEL_ID:0:12}-${CATEGORY:0:16}"
        LOG_FILE="$SLURM_LOGS/${MODEL_ID}__${CATEGORY}_%j.out"

        # DeepSeek-R1 generates very long CoT chains — needs more time
        JOB_TIME="$TIME"
        if [[ "$MODEL_ID" == "deepseek-r1-distill-llama-8b" ]]; then
            JOB_TIME="08:00:00"
        fi

        CMD="cd $REPO_ROOT \
            && HF_HUB_OFFLINE=1 \
            XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
            /n/fs/scratch/dl3533/conda_envs/MARIO/bin/python embedllm_eval/run_one_job.py \
            --model-id $MODEL_ID \
            --category $CATEGORY \
            --backend $BACKEND \
            --batch-size $BATCH_SIZE"

        SBATCH_CMD="sbatch \
            --job-name=$JOB_NAME \
            --partition=$PARTITION \
            --account=$ACCOUNT \
            --gres=$GRES \
            --time=$JOB_TIME \
            --mem=$MEM \
            --cpus-per-task=$CPUS \
            --output=$LOG_FILE \
            --error=$LOG_FILE \
            --wrap=\"$CMD\""

        if $DRY_RUN; then
            echo "[DRY-RUN] $MODEL_ID / $CATEGORY"
            echo "          $CMD"
            echo ""
        else
            eval "$SBATCH_CMD"
            SUBMITTED=$(( SUBMITTED + 1 ))
        fi
    done
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete. Would submit $((N_JOBS - SKIPPED)) jobs ($SKIPPED already done)."
else
    echo "Submitted $SUBMITTED jobs, skipped $SKIPPED already-complete."
    echo "Monitor with: squeue -u \$USER"
    echo "After all jobs finish, run: python embedllm_eval/stitch_results.py"
fi
