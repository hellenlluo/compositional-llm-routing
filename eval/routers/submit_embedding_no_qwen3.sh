#!/usr/bin/env bash
# Cosine-similarity embedding router — no_qwen3 model set (8 models).
# Excludes qwen3-30b-a3b and qwen3-4b-thinking-2507.
# Runs full-task + subtask for morehopqa and musique.
# Results saved to outputs/routing/no_qwen3/
#
# Usage:
#   bash eval/routers/submit_embedding_no_qwen3.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="$REPO_ROOT/eval/routers/logs"
OUT_DIR="$REPO_ROOT/outputs/routing/no_qwen3"
mkdir -p "$LOGS_DIR" "$OUT_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
EXCLUDED="qwen3-30b-a3b qwen3-4b-thinking-2507"

for DATASET in morehopqa musique; do
    for LEVEL in full subtask; do
        JOB="embrouter-no-qwen3-${DATASET}-${LEVEL}"
        sbatch \
            --job-name="$JOB" \
            --account=allcs \
            --partition=pvl \
            --gres=gpu:a6000:1 \
            --time=1:00:00 \
            --mem=16G \
            --cpus-per-task=4 \
            --output="$LOGS_DIR/${JOB}_%j.out" \
            --error="$LOGS_DIR/${JOB}_%j.out" \
            --wrap="cd $REPO_ROOT && \
                XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
                HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
                HF_HUB_DISABLE_XET=1 \
                $CONDA_PYTHON -m eval.routers.embedding_router \
                    --dataset $DATASET \
                    --level $LEVEL \
                    --model-set no_qwen3 \
                    --exclude-models $EXCLUDED"
        echo "Submitted $JOB. Logs: $LOGS_DIR/${JOB}_<jobid>.out"
    done
done

echo ""
echo "Results will be written to: $OUT_DIR/"
echo "  embedding_morehopqa_full.json"
echo "  embedding_morehopqa_subtask.json"
echo "  embedding_musique_full.json"
echo "  embedding_musique_subtask.json"
