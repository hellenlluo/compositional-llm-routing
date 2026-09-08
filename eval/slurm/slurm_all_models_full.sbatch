#!/bin/bash
#SBATCH --job-name=cos484-full
#SBATCH --array=0-6
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/full-%A_%a.out
#SBATCH --error=outputs/logs/full-%A_%a.err

# Full-dataset inference, 7 dense models (Qwen3-30B-A3B handled in a
# separate vLLM tp=2 job). Data-parallel 2 shards per sub-task for ~2x
# speedup on the 8B-class models where subtask decode time is the bottleneck.
#
# If any sub-task hits the 12h time limit mid-run, just `sbatch` this again —
# the SQLite cache is idempotent and the next run resumes from where it stopped.

set -euo pipefail

MODELS=(
    qwen1.5-0.5b-chat              # 0
    phi-4-mini-instruct            # 1
    qwen3-4b-thinking-2507         # 2
    medgemma-4b-it                 # 3
    mistral-7b-instruct-v0.3       # 4
    llama-3.1-8b-instruct          # 5
    deepseek-r1-distill-llama-8b   # 6
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
if [ ! -x /scratch/ez2103/conda_envs/MARIO/bin/python ]; then
    echo "installing MARIO on $(hostname)..."
    bash ~/cos484-final-project/setup_env.sh
fi
conda activate /scratch/ez2103/conda_envs/MARIO

export HF_HOME=/scratch/ez2103/.cache/huggingface
mkdir -p "$HF_HOME"
cd ~/cos484-final-project

NUM_SHARDS=2
BATCH_SIZE=16

echo "=== $(date) : model=$MODEL  host=$(hostname)  array_task=$SLURM_ARRAY_TASK_ID  shards=$NUM_SHARDS ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for DATASET in musique morehopqa stepcot; do
    for PHASE in full subtask; do
        echo
        echo "=== $(date) : $MODEL / $DATASET / $PHASE ==="
        PIDS=()
        for ((s=0; s<NUM_SHARDS; s++)); do
            LOG="outputs/logs/full-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${MODEL}-${DATASET}-${PHASE}-shard${s}.log"
            CUDA_VISIBLE_DEVICES=$s \
            python -m eval.run_inference \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --phase "$PHASE" \
                --no-limit \
                --shard-idx "$s" \
                --num-shards "$NUM_SHARDS" \
                --backend hf \
                --batch-size "$BATCH_SIZE" \
                > "$LOG" 2>&1 &
            PIDS+=($!)
        done
        echo "  launched ${#PIDS[@]} shards: ${PIDS[*]}"
        FAILED=0
        for pid in "${PIDS[@]}"; do
            wait "$pid" || { echo "  WARN: pid $pid exited non-zero"; FAILED=$((FAILED+1)); }
        done
        echo "  $DATASET/$PHASE done ($FAILED shard failures)"
    done
done

echo
echo "=== $(date) : exporting $MODEL to JSONL ==="
python -m eval.export_runs --model "$MODEL"

echo "=== $(date) : done for $MODEL ==="
