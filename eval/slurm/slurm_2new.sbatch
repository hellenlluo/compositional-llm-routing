#!/bin/bash
#SBATCH --job-name=cos484-2new
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/2new-%A_%a.out
#SBATCH --error=outputs/logs/2new-%A_%a.err
#SBATCH --requeue

set -euo pipefail

MODELS=(
    mathstral-7b                     # 0
    llama-3.1-nemotron-nano-8b       # 1
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
if ! /scratch/ez2103/conda_envs/MARIO/bin/python -c "import vllm" 2>/dev/null; then
    echo "MARIO missing or incomplete, wiping and reinstalling"
    rm -rf /scratch/ez2103/conda_envs/MARIO
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

echo "=== $(date) : model=$MODEL  max_tokens=4000 + guided regex ==="
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
python -m eval.export_runs --model "$MODEL" --cache-db "$CACHE_DB" --out-dir "$REPO/outputs/runs_4k"
echo "=== $(date) : done ==="
