#!/usr/bin/env bash
# Run heterogeneous_accuracy.py restricted to the 4 small/weak models where
# subtask routing (B) outperforms full-task routing (A) on morehopqa.
#
# Models: mistral-7b-instruct-v0.3, qwen1.5-0.5b-chat,
#         phi-4-mini-instruct, llama-3.1-nemotron-nano-8b
#
# "mathstral-7b" "medgemma-4b-it" "deepseek-r1-distill-llama-8b" "llama-3.1-8b-instruct"
# Run both oracle variants so you can see strict vs soft side-by-side.
# Usage: bash router_analysis/heterogeneous_accuracy_small_models.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

MODELS=(
    "mistral-7b-instruct-v0.3"
    "qwen1.5-0.5b-chat"
    "phi-4-mini-instruct"
    "llama-3.1-nemotron-nano-8b"
)

for ORACLE in strict soft; do
    echo ""
    echo "################################################################"
    echo "  oracle = ${ORACLE}"
    echo "################################################################"
    "$PYTHON" "$SCRIPT_DIR/heterogeneous_accuracy.py" \
        --models "${MODELS[@]}" \
        --oracle "$ORACLE"
done
