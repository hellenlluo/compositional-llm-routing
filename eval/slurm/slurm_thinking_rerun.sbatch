#!/bin/bash
#SBATCH --job-name=cos484-think
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/think-%A_%a.out
#SBATCH --error=outputs/logs/think-%A_%a.err
#SBATCH --requeue

# Rerun the 2 thinking models with max_new_tokens=2048 (was 512).
# Uses a SEPARATE cache DB so old 512-token responses are preserved.
# After inference, strips <think> blocks before export.

set -euo pipefail

MODELS=(
    qwen3-4b-thinking-2507         # 0
    qwen3-30b-a3b                  # 1
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

REPO=~/cos484-final-project
cd "$REPO"

# preflight ECC
BAD=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null \
    | awk '{if($1 != "[N/A]" && $1+0 > 0) print "bad"}')
if [ -n "$BAD" ]; then
    scontrol requeue $SLURM_JOB_ID
    exit 1
fi

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
if [ ! -x /scratch/ez2103/conda_envs/MARIO/bin/python ]; then
    bash "$REPO/setup_env.sh"
fi
conda activate /scratch/ez2103/conda_envs/MARIO

export HF_HOME=/scratch/ez2103/.cache/huggingface
mkdir -p "$HF_HOME"
TOKEN="${HF_TOKEN:-}"
for CAND in /u/ez2103/.cache/huggingface/token /auto/u/ez2103/.cache/huggingface/token "$HOME/.cache/huggingface/token"; do
    [ -n "$TOKEN" ] && break
    TOKEN=$(cat "$CAND" 2>/dev/null || true)
done
if [ -n "$TOKEN" ]; then
    printf '%s' "$TOKEN" > "$HF_HOME/token"
    chmod 600 "$HF_HOME/token"
    export HF_TOKEN="$TOKEN"
fi

# Fresh cache — do NOT seed from home (we want new 2048-token generations)
CACHE_DB=/scratch/ez2103/cache_${MODEL}_2048.db

# For 30B-A3B, need vLLM tp=2 but we only have 1 GPU here.
# Use vLLM tp=1 for 4B-thinking. 30B needs its own 2-GPU job.
if [ "$MODEL" = "qwen3-30b-a3b" ]; then
    echo "SKIP: $MODEL needs 2 GPUs (tensor-parallel). Use slurm_30b_rerun.sbatch instead."
    exit 0
fi

BATCH_SIZE=16

echo "=== $(date) : model=$MODEL  max_new_tokens=2048  cache=$CACHE_DB ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for DATASET in musique morehopqa stepcot; do
    for PHASE in full subtask; do
        echo
        echo "=== $(date) : $MODEL / $DATASET / $PHASE ==="
        python -m eval.run_inference \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --phase "$PHASE" \
            --no-limit \
            --backend vllm \
            --batch-size "$BATCH_SIZE" \
            --cache-db "$CACHE_DB" \
            --max-new-tokens 2048
    done
done

echo
echo "=== $(date) : exporting $MODEL to runs_2048/ ==="
python -m eval.export_runs --model "$MODEL" --cache-db "$CACHE_DB" --out-dir "$REPO/outputs/runs_2048"

echo "=== $(date) : done for $MODEL ==="
