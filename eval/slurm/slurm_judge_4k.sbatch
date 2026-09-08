#!/bin/bash
#SBATCH --job-name=judge-4k
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/judge4k-%A_%a.out
#SBATCH --error=outputs/logs/judge4k-%A_%a.err
#SBATCH --requeue

# Judge the 4k-guided model runs with Prometheus.
# Reads outputs/runs_4k/  → writes outputs/judgments_4k/

set -euo pipefail

MODELS=(
    phi-4-mini-instruct            # 0
    llama-3.1-8b-instruct          # 1
    qwen3-30b-a3b                  # 2
    mistral-7b-instruct-v0.3       # 3
    medgemma-4b-it                 # 4
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
REPO=~/cos484-final-project
cd "$REPO"

BAD=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null \
    | awk '{if($1 != "[N/A]" && $1+0 > 0) print "bad"}')
if [ -n "$BAD" ]; then
    scontrol requeue $SLURM_JOB_ID; exit 1
fi

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
if ! /scratch/ez2103/conda_envs/MARIO/bin/python -c "import vllm" 2>/dev/null; then
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

echo "=== $(date) : judging $MODEL 4k-guided outputs ==="
python -m eval.run_judge \
    --model "$MODEL" \
    --runs-dir "$REPO/outputs/runs_4k" \
    --judgments-dir "$REPO/outputs/judgments_4k" \
    --backend vllm \
    --batch-size 16

echo "=== $(date) : done ==="
