#!/bin/bash
#SBATCH --job-name=cos484-4k
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/4k-%A_%a.out
#SBATCH --error=outputs/logs/4k-%A_%a.err
#SBATCH --requeue

# Rerun with max_new_tokens=4000 AND guided-decoding regex that forces every
# response to end with "ANSWER: <answer>".
# qwen3-30b-a3b handled separately (needs tp=2).

set -euo pipefail

MODELS=(
    llama-3.1-8b-instruct          # 0
    medgemma-4b-it                 # 1
    deepseek-r1-distill-llama-8b   # 2  (also truncates heavily)
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
if ! /scratch/ez2103/conda_envs/MARIO/bin/python -c "import vllm" 2>/dev/null; then echo "MARIO missing or incomplete, wiping and reinstalling"; rm -rf /scratch/ez2103/conda_envs/MARIO
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

CACHE_DB=/scratch/ez2103/cache_${MODEL}_4k.db
REGEX='[\s\S]{0,14000}\nANSWER: [^\n]{1,500}'

echo "=== $(date) : model=$MODEL  max_tokens=4000  guided_regex=$REGEX ==="
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
            --max-new-tokens 4000 \
            --force-answer-regex "$REGEX"
    done
done

echo
echo "=== $(date) : exporting $MODEL to runs_4k/ ==="
python -m eval.export_runs --model "$MODEL" --cache-db "$CACHE_DB" --out-dir "$REPO/outputs/runs_4k"
echo "=== $(date) : done ==="
