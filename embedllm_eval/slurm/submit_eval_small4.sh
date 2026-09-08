#!/usr/bin/env bash
# Evaluate the small4 EmbedLLM checkpoint (full-task + subtask).
# Results saved to: embedllm_eval/eval_results/small4/
#
# Usage:
#   bash embedllm_eval/submit_eval_small4.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
OUT_DIR="$REPO_ROOT/embedllm_eval/eval_results/small4"
mkdir -p "$SLURM_LOGS" "$OUT_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
CKPT="/n/fs/scratch/dl3533/models/embedclassifier_small4.pt"

sbatch \
    --job-name=embedllm_eval-small4 \
    --partition=pvl \
    --account=allcs \
    --gres=gpu:a6000:1 \
    --time=01:00:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/eval_small4_%j.out" \
    --error="$SLURM_LOGS/eval_small4_%j.out" \
    --wrap="cd $REPO_ROOT && \
        XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        $CONDA_PYTHON embedllm_eval/eval_specialized.py \
            --ckpt $CKPT \
            --out-dir $OUT_DIR && \
        $CONDA_PYTHON embedllm_eval/eval_subtasks.py \
            --ckpt $CKPT \
            --out-dir $OUT_DIR"

echo "Submitted embedllm_eval-small4. Logs: $SLURM_LOGS/eval_small4_<jobid>.out"
echo "Results: $OUT_DIR/"
echo "  Full-task:       $OUT_DIR/summary.json"
echo "  Overall subtask: $OUT_DIR/overall_subtask_summary.json"
echo "  Chained subtask: $OUT_DIR/subtask_summary.json"
echo "  Predictions:     $OUT_DIR/{morehopqa,musique,stepcot}_predictions.json"
