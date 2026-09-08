#!/bin/bash
#SBATCH --job-name=judge-qwen
#SBATCH --array=0-6
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/judgeqwen-%A_%a.out
#SBATCH --error=outputs/logs/judgeqwen-%A_%a.err
#SBATCH --requeue

# Re-judge with Qwen2.5-32B-Instruct (stronger than Prometheus-7B).
# Writes to outputs/judgments_4k_qwen/qwen2.5-32b/<dataset>/<model>/<phase>.jsonl

set -euo pipefail

MODELS=(
    phi-4-mini-instruct            # 0
    llama-3.1-8b-instruct          # 1
    qwen3-30b-a3b                  # 2
    mistral-7b-instruct-v0.3       # 3
    medgemma-4b-it                 # 4
    qwen1.5-0.5b-chat              # 5
    deepseek-r1-distill-llama-8b   # 6
    qwen3-4b-thinking-2507         # 7
    mathstral-7b                   # 8
    llama-3.1-nemotron-nano-8b     # 9
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

echo "=== $(date) : judging $MODEL with Qwen2.5-32B ==="
python -m eval.run_judge \
    --model "$MODEL" \
    --runs-dir "$REPO/outputs/runs_4k" \
    --judgments-dir "$REPO/outputs/judgments_4k_qwen" \
    --judge-config "$REPO/eval/configs/judge_qwen.yaml" \
    --backend vllm \
    --batch-size 16

echo "=== $(date) : done ==="
