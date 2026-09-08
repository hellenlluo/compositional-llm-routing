#!/usr/bin/env bash
# Data-parallel full inference for ONE model across all 3 datasets, both phases.
# Launches NUM_GPUS workers per (dataset, phase), each on its own GPU via
# CUDA_VISIBLE_DEVICES, sharded by record index.
#
# Usage:
#   bash eval/run_full_single_model.sh <model_id> [num_gpus] [backend]
# Example:
#   bash eval/run_full_single_model.sh qwen1.5-0.5b-chat 8 hf
#
# The SQLite cache handles concurrent writes via WAL mode, so re-runs are
# idempotent and crash-safe.

set -euo pipefail

MODEL="${1:-qwen1.5-0.5b-chat}"
NUM_GPUS="${2:-8}"
BACKEND="${3:-hf}"           # hf or vllm. vllm is faster but has longer startup per worker.
BATCH_SIZE="${4:-16}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/outputs/logs/$MODEL"
mkdir -p "$LOG_DIR"

echo "=== full-dataset run: model=$MODEL, gpus=$NUM_GPUS, backend=$BACKEND ==="
echo "logs: $LOG_DIR/"

for DATASET in musique morehopqa stepcot; do
    for PHASE in full subtask; do
        echo
        echo "--- $DATASET / $PHASE ---"
        PIDS=()
        for ((i=0; i<NUM_GPUS; i++)); do
            LOG="$LOG_DIR/${DATASET}_${PHASE}_gpu${i}.log"
            CUDA_VISIBLE_DEVICES=$i \
            python -m eval.run_inference \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --phase "$PHASE" \
                --no-limit \
                --shard-idx "$i" \
                --num-shards "$NUM_GPUS" \
                --backend "$BACKEND" \
                --batch-size "$BATCH_SIZE" \
                > "$LOG" 2>&1 &
            PIDS+=($!)
        done
        echo "  launched ${#PIDS[@]} workers: ${PIDS[*]}"
        # Wait for all shards of this combo to finish before starting the next.
        for pid in "${PIDS[@]}"; do
            wait "$pid" || echo "  WARN: pid $pid exited non-zero (see its log)"
        done
        echo "  $DATASET/$PHASE complete."
    done
done

echo
echo "=== all combos done for $MODEL ==="
echo "cache: $REPO_ROOT/outputs/cache.db"
echo "run  : sqlite3 $REPO_ROOT/outputs/cache.db 'SELECT dataset, subtask_idx>-1 as is_subtask, COUNT(*) FROM runs WHERE model_id=\"$MODEL\" GROUP BY 1,2;'"
