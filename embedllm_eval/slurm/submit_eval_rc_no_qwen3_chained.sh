#!/usr/bin/env bash
# Chained subtask evaluation for EmbedLLM on R+C questions (GPU required).
# Runs eval_subtasks.py with --qid-filter to route each subtask independently.
# Saves to embedllm_eval/eval_results/rc_questions/no_qwen3/subtask_summary.json
#
# Prerequisites:
#   1. python router_analysis/generate_rc_qids.py   (outputs/rc_qids.json)
#   2. submit_eval_rc_no_qwen3.sh  (CPU job for full-task + overall subtask)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
OUT_DIR="$REPO_ROOT/embedllm_eval/eval_results/rc_questions/no_qwen3"
QID_FILTER="$REPO_ROOT/outputs/rc_qids.json"
mkdir -p "$SLURM_LOGS" "$OUT_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
CKPT="/n/fs/scratch/dl3533/models/embedclassifier_no_qwen3.pt"

sbatch \
    --job-name=embedllm-rc-chain-nq \
    --partition=pvl \
    --account=allcs \
    --gres=gpu:a6000:1 \
    --time=0:30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/eval_rc_no_qwen3_chained_%j.out" \
    --error="$SLURM_LOGS/eval_rc_no_qwen3_chained_%j.out" \
    --wrap="cd $REPO_ROOT && \
        XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        $CONDA_PYTHON embedllm_eval/eval_subtasks.py \
            --ckpt $CKPT \
            --qid-filter $QID_FILTER \
            --out-dir $OUT_DIR"

echo "Submitted embedllm-rc-chain-nq. Logs: $SLURM_LOGS/eval_rc_no_qwen3_chained_<jobid>.out"
echo "Results: $OUT_DIR/subtask_summary.json  (chained subtask routing)"
