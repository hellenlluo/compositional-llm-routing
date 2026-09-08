#!/bin/bash
#SBATCH --job-name=cos484-judge
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/judge-%A_%a.out
#SBATCH --error=outputs/logs/judge-%A_%a.err
#SBATCH --requeue

# Run Prometheus-7B-v2.0 (eval/configs/judge.yaml) over each model's exported
# JSONLs. One array sub-task per model — re-runnable; run_judge.py skips rows
# already in outputs/judgments/<judge_id>/.../*.jsonl.

set -euo pipefail

MODELS=(
    qwen1.5-0.5b-chat              # 0
    phi-4-mini-instruct            # 1
    qwen3-4b-thinking-2507         # 2
    medgemma-4b-it                 # 3
    mistral-7b-instruct-v0.3       # 4
    llama-3.1-8b-instruct          # 5
    deepseek-r1-distill-llama-8b   # 6
    qwen3-30b-a3b                  # 7
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

REPO=~/cos484-final-project
cd "$REPO"

# preflight ECC
BAD=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null \
    | awk '{if($1 != "[N/A]" && $1+0 > 0) print "bad"}')
if [ -n "$BAD" ]; then
    echo "PREFLIGHT: bad ECC on $(hostname) — requeueing"
    scontrol requeue $SLURM_JOB_ID
    exit 1
fi

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
if [ ! -x /scratch/ez2103/conda_envs/MARIO/bin/python ]; then
    bash "$REPO/setup_env.sh"
fi
conda activate /scratch/ez2103/conda_envs/MARIO

# HF token (Prometheus is ungated but the cli still wants creds for cache)
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

echo "=== $(date) : judging $MODEL on $(hostname) ==="

# Confirm there are JSONLs for this model. If not (e.g. inference still
# running for medgemma/llama/deepseek), exit 0 — nothing to do yet.
JSONL_COUNT=$(find outputs/runs -path "*/$MODEL/*.jsonl" 2>/dev/null | wc -l)
if [ "$JSONL_COUNT" -eq 0 ]; then
    echo "no JSONLs for $MODEL yet — exiting cleanly. Resubmit this index later."
    exit 0
fi
echo "found $JSONL_COUNT JSONL files for $MODEL"

python -m eval.run_judge \
    --model "$MODEL" \
    --backend vllm \
    --batch-size 16

echo "=== $(date) : done judging $MODEL ==="
ls -la outputs/judgments/prometheus-7b-v2/*/${MODEL}/*.jsonl 2>/dev/null
