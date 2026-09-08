#!/bin/bash
#SBATCH --job-name=judge-2048
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/judge2048-%A_%a.out
#SBATCH --error=outputs/logs/judge2048-%A_%a.err
#SBATCH --requeue

# Judge the 3 recovered 2048-token models.
# Reads from outputs/runs_2048/, writes to outputs/judgments_2048/.

set -euo pipefail

MODELS=(
    llama-3.1-8b-instruct          # 0
    medgemma-4b-it                 # 1
    qwen3-30b-a3b                  # 2
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

echo "=== $(date) : judging $MODEL (2048-token rerun) on $(hostname) ==="
python -m eval.run_judge \
    --model "$MODEL" \
    --runs-dir "$REPO/outputs/runs_2048" \
    --judgments-dir "$REPO/outputs/judgments_2048" \
    --backend vllm \
    --batch-size 16

echo "=== $(date) : done ==="
