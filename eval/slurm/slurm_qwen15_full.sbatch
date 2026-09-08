#!/bin/bash
#SBATCH --job-name=qwen15-full
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/qwen15-full-%j.out
#SBATCH --error=outputs/logs/qwen15-full-%j.err

# Run qwen1.5-0.5b-chat on the ENTIRE dataset (all 3 datasets, both phases).
# Single GPU, HF backend (most reliable). Idempotent — restarts pick up where
# the cache.db left off.
#
# Submit with:   sbatch eval/slurm_qwen15_full.sbatch

set -euo pipefail

# Activate MARIO on this compute node. If scratch doesn't have the env, install first.
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
if [ ! -x /scratch/ez2103/conda_envs/MARIO/bin/python ]; then
    echo "MARIO not on this node's scratch — installing first..."
    bash ~/cos484-final-project/setup_env.sh
fi
conda activate /scratch/ez2103/conda_envs/MARIO

export HF_HOME=/scratch/ez2103/.cache/huggingface
mkdir -p "$HF_HOME"
cd ~/cos484-final-project

MODEL="qwen1.5-0.5b-chat"
BATCH_SIZE=32
BACKEND=hf

echo "=== $(date) : starting $MODEL full-dataset run on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

for DATASET in musique morehopqa stepcot; do
    for PHASE in full subtask; do
        echo
        echo "=== $(date) : $MODEL / $DATASET / $PHASE ==="
        python -m eval.run_inference \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --phase "$PHASE" \
            --no-limit \
            --backend "$BACKEND" \
            --batch-size "$BATCH_SIZE"
    done
done

echo
echo "=== $(date) : exporting runs to JSONL tree ==="
python -m eval.export_runs --model "$MODEL"

echo
echo "=== $(date) : all done ==="
sqlite3 outputs/cache.db <<EOF
.headers on
.mode column
SELECT dataset, subtask_idx>-1 as is_subtask, COUNT(*) as rows
FROM runs WHERE model_id='$MODEL' GROUP BY 1,2;
EOF
