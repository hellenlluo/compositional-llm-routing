#!/usr/bin/env bash
# Evaluate the trained accuracy-augmented EmbedLLM router on held-out datasets.
#
# Requires:
#   /n/fs/scratch/dl3533/models/perf_router/perf_router_best.pt  (train first)
#   perf_router/results/accuracy_vectors.json
#
# Usage:
#   bash perf_router/training/submit_eval_perf_router.sh
#   bash perf_router/training/submit_eval_perf_router.sh --ckpt /path/to/other.pt
#   bash perf_router/training/submit_eval_perf_router.sh --datasets musique morehopqa

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_LOGS="$REPO_ROOT/perf_router/logs"
mkdir -p "$SLURM_LOGS"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"
DEFAULT_CKPT="/n/fs/scratch/dl3533/models/perf_router/perf_router_best.pt"
EXTRA_ARGS="${*}"

if [[ ! -f "$DEFAULT_CKPT" ]] && [[ "$*" != *"--ckpt"* ]]; then
    echo "ERROR: checkpoint not found at $DEFAULT_CKPT"
    echo "Train first with submit_train_perf_router.sh"
    exit 1
fi

sbatch \
    --job-name="eval_perf_router" \
    --account=allcs \
    --gres=gpu:a6000:1 \
    --time=0:30:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/eval_perf_router_%j.out" \
    --error="$SLURM_LOGS/eval_perf_router_%j.out" \
    --wrap="cd $REPO_ROOT && \
        HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface \
        HF_HUB_DISABLE_XET=1 \
        $CONDA_PYTHON perf_router/eval/eval_perf_router.py \
            $EXTRA_ARGS"

echo "Submitted eval_perf_router. Logs: $SLURM_LOGS/eval_perf_router_<jobid>.out"
echo "Results will be saved to: $REPO_ROOT/perf_router/results/qwen3/"
