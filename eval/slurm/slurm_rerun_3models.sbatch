#!/bin/bash
#SBATCH --job-name=cos484-rerun
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/rerun-%A_%a.out
#SBATCH --error=outputs/logs/rerun-%A_%a.err
#SBATCH --requeue

set -euo pipefail

MODELS=(
    deepseek-r1-distill-llama-8b   # 0
    medgemma-4b-it                 # 1
    llama-3.1-8b-instruct          # 2
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

REPO=~/cos484-final-project
cd "$REPO"

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

# Fresh cache — no seeding from old 512-token results
CACHE_DB=/scratch/ez2103/cache_${MODEL}_2048.db

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
            --batch-size 16 \
            --cache-db "$CACHE_DB" \
            --max-new-tokens 2048
    done
done

echo
echo "=== $(date) : exporting $MODEL to runs_2048/ ==="
python -m eval.export_runs --model "$MODEL" --cache-db "$CACHE_DB" --out-dir "$REPO/outputs/runs_2048"

echo "=== $(date) : done for $MODEL ==="
