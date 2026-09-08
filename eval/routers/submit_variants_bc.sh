#!/usr/bin/env bash
# Submit SIMROUTE Variant B (example descriptions) and Variant C (strength prior)
# for both model sets (no_qwen3, small4) and both datasets (morehopqa, musique).
#
# Variant B: full + subtask level
# Variant C: full level only (λ tuned on dev split)
#
# Usage:
#   bash eval/routers/submit_variants_bc.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS_DIR="$REPO_ROOT/eval/routers/logs"
mkdir -p "$LOGS_DIR"

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

EXCLUDED_NO_QWEN3="qwen3-30b-a3b qwen3-4b-thinking-2507"
EXCLUDED_SMALL4="deepseek-r1-distill-llama-8b llama-3.1-8b-instruct medgemma-4b-it mathstral-7b qwen3-30b-a3b qwen3-4b-thinking-2507"

SBATCH_COMMON=(
    --account=allcs
    --partition=pvl
    --gres=gpu:a6000:1
    --time=1:00:00
    --mem=16G
    --cpus-per-task=4
)

ENV_VARS="XDG_CACHE_HOME=/n/fs/scratch/dl3533/.cache HF_HOME=/n/fs/scratch/dl3533/.cache/huggingface HF_HUB_DISABLE_XET=1"

echo "=== Submitting Variant B (example descriptions) ==="
for MODEL_SET in no_qwen3 small4; do
    if [ "$MODEL_SET" = "no_qwen3" ]; then
        EXCLUDED="$EXCLUDED_NO_QWEN3"
    else
        EXCLUDED="$EXCLUDED_SMALL4"
    fi
    mkdir -p "$REPO_ROOT/outputs/routing/$MODEL_SET"
    for DATASET in morehopqa musique; do
        for LEVEL in full subtask; do
            JOB="simB-${MODEL_SET}-${DATASET}-${LEVEL}"
            sbatch \
                "${SBATCH_COMMON[@]}" \
                --job-name="$JOB" \
                --output="$LOGS_DIR/${JOB}_%j.out" \
                --error="$LOGS_DIR/${JOB}_%j.out" \
                --wrap="cd $REPO_ROOT && $ENV_VARS \
                    $CONDA_PYTHON -m eval.routers.example_router \
                        --dataset $DATASET \
                        --level $LEVEL \
                        --model-set $MODEL_SET \
                        --exclude-models $EXCLUDED"
            echo "  Submitted $JOB"
        done
    done
done

echo ""
echo "=== Submitting Variant C (strength prior) ==="
for MODEL_SET in no_qwen3 small4; do
    if [ "$MODEL_SET" = "no_qwen3" ]; then
        EXCLUDED="$EXCLUDED_NO_QWEN3"
    else
        EXCLUDED="$EXCLUDED_SMALL4"
    fi
    for DATASET in morehopqa musique; do
        JOB="simC-${MODEL_SET}-${DATASET}"
        sbatch \
            "${SBATCH_COMMON[@]}" \
            --job-name="$JOB" \
            --output="$LOGS_DIR/${JOB}_%j.out" \
            --error="$LOGS_DIR/${JOB}_%j.out" \
            --wrap="cd $REPO_ROOT && $ENV_VARS \
                $CONDA_PYTHON -m eval.routers.strength_prior_router \
                    --dataset $DATASET \
                    --model-set $MODEL_SET \
                    --exclude-models $EXCLUDED"
        echo "  Submitted $JOB"
    done
done

echo ""
echo "Logs: $LOGS_DIR/simB-*  and  $LOGS_DIR/simC-*"
echo "Results: outputs/routing/{no_qwen3,small4}/{example,strength_prior}_*.json"
