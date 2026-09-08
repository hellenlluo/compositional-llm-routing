#!/bin/bash
#SBATCH --job-name=cos484-30b
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/qwen30b-%j.out
#SBATCH --error=outputs/logs/qwen30b-%j.err

# Qwen3-30B-A3B with tensor-parallel-size=2 (60 GB bf16 fits across 2x L40
# 46 GB). Uses vLLM for real throughput since 30B on HF would be unusably slow.
#
# If vLLM startup hangs the way it did with 8 parallel instances on the login
# node, one instance alone on a compute node should be fine (the bug was about
# 8 vLLM engines on one host racing on IPC).

set -euo pipefail

MODEL="qwen3-30b-a3b"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
if [ ! -x /scratch/ez2103/conda_envs/MARIO/bin/python ]; then
    bash ~/cos484-final-project/setup_env.sh
fi
conda activate /scratch/ez2103/conda_envs/MARIO

export HF_HOME=/scratch/ez2103/.cache/huggingface
mkdir -p "$HF_HOME"
cd ~/cos484-final-project

echo "=== $(date) : $MODEL on $(hostname) ==="
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
            --batch-size 32
    done
done

echo
python -m eval.export_runs --model "$MODEL"
echo "=== $(date) : done ==="
