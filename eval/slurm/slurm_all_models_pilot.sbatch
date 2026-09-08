#!/bin/bash
#SBATCH --job-name=cos484-pilot
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/pilot-%A_%a.out
#SBATCH --error=outputs/logs/pilot-%A_%a.err

# Slurm array job: one sub-task per model. Each sub-task runs the PILOT
# (N=200 per dataset) across all 3 datasets × both phases.
#
# Array index maps to one of the 8 models via the MODELS list below. Bump
# --time or --array size as we add models / scale up.
#
# Submit:   sbatch eval/slurm_all_models_pilot.sbatch
# Watch :   squeue -u $USER      ;  tail -f outputs/logs/pilot-<jobid>_<idx>.out

set -euo pipefail

MODELS=(
    qwen1.5-0.5b-chat
    phi-4-mini-instruct
    qwen3-4b-thinking-2507
    medgemma-4b-it
    mistral-7b-instruct-v0.3
    llama-3.1-8b-instruct
    deepseek-r1-distill-llama-8b
    qwen3-30b-a3b
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

echo "=== $(date) : model=$MODEL  host=$(hostname)  array_task=$SLURM_ARRAY_TASK_ID ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Qwen3-30B-A3B is 60GB bf16 — won't fit on one 46GB L40. Skip with a warning
# for now; we'll handle it separately with tp=2.
if [ "$MODEL" = "qwen3-30b-a3b" ]; then
    echo "SKIP: $MODEL needs tensor-parallel-size=2 on L40. Handle in a separate job."
    exit 0
fi

BATCH_SIZE=16

for DATASET in musique morehopqa stepcot; do
    for PHASE in full subtask; do
        echo
        echo "=== $(date) : $MODEL / $DATASET / $PHASE ==="
        python -m eval.run_inference \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --phase "$PHASE" \
            --backend hf \
            --batch-size "$BATCH_SIZE"
    done
done

echo
echo "=== $(date) : exporting $MODEL to JSONL ==="
python -m eval.export_runs --model "$MODEL"

echo "=== $(date) : done for $MODEL ==="
