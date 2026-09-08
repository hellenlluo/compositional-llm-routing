#!/usr/bin/env bash
# Derive EmbedLLM R+C metrics from existing predictions — no GPU needed.
# Full-task + overall subtask saved to embedllm_eval/eval_results/rc_questions/small4/
# Chained subtask (GPU) is run in a separate job via submit_eval_rc_small4_chained.sh.
#
# Prerequisites:
#   1. python router_analysis/generate_rc_qids.py   (outputs/rc_qids.json)
#   2. bash embedllm_eval/submit_eval_small4.sh     (produces {dataset}_predictions.json)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_LOGS="$REPO_ROOT/embedllm_eval/slurm-logs"
PREDS_DIR="$REPO_ROOT/embedllm_eval/eval_results/small4"
OUT_DIR="$REPO_ROOT/embedllm_eval/eval_results/rc_questions/small4"
QID_FILTER="$REPO_ROOT/outputs/rc_qids.json"
mkdir -p "$SLURM_LOGS" "$OUT_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

sbatch \
    --job-name=embedllm-rc-small4 \
    --partition=pvl \
    --account=allcs \
    --time=0:10:00 \
    --mem=8G \
    --cpus-per-task=2 \
    --output="$SLURM_LOGS/eval_rc_small4_%j.out" \
    --error="$SLURM_LOGS/eval_rc_small4_%j.out" \
    --wrap="cd $REPO_ROOT && \
        $CONDA_PYTHON embedllm_eval/eval_from_preds.py \
            --preds-dir $PREDS_DIR \
            --out-dir $OUT_DIR \
            --qid-filter $QID_FILTER"

echo "Submitted embedllm-rc-small4 (CPU). Logs: $SLURM_LOGS/eval_rc_small4_<jobid>.out"
echo "Results: $OUT_DIR/"
echo "  Full-task:       $OUT_DIR/summary.json"
echo "  Overall subtask: $OUT_DIR/overall_subtask_summary.json"
echo "  (Chained subtask still requires GPU — run submit_eval_rc_small4_chained.sh)"
