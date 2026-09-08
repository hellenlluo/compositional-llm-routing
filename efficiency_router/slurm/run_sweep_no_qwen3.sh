#!/usr/bin/env bash
# Like run_sweep.sh but excludes qwen3-30b-a3b and qwen3-4b-thinking-2507.
# Called by submit_sweep_no_qwen3.sh as the SLURM job script.

set -euo pipefail

SCRIPT_DIR="$1"
SWEEP_DIR="$2"
shift 2
if [ "$#" -gt 0 ]; then
    OFFSETS=("$@")
fi

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

export HF_HOME="/n/fs/scratch/dl3533/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$SCRIPT_DIR"

OFFSETS=(1 2 3 4 5 6 7 8 9 10)
EXCLUDED="qwen3-30b-a3b qwen3-4b-thinking-2507"

ALL_RESULTS=()

for OFFSET in "${OFFSETS[@]}"; do
    TAG="offset_${OFFSET}"
    CKPT="$SCRIPT_DIR/checkpoints/effrouter_no_qwen3_${TAG}.pt"
    OUT_DIR="$SWEEP_DIR/${TAG}"
    RESULT="$OUT_DIR/eval_results.json"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  TRAIN  offset=$OFFSET  (no qwen3)"
    echo "════════════════════════════════════════════════════"
    "$CONDA_PYTHON" train.py \
        --offset "$OFFSET" \
        --exclude-models $EXCLUDED \
        --ckpt-name "effrouter_no_qwen3_${TAG}"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  EVAL   offset=$OFFSET  (no qwen3)"
    echo "════════════════════════════════════════════════════"
    "$CONDA_PYTHON" eval.py \
        --ckpt "$CKPT" \
        --out  "$RESULT" \
        --subtask-out "$OUT_DIR/subtask_results.json" \
        --predictions-out "$OUT_DIR/predictions.json"

    ALL_RESULTS+=("$OFFSET:$RESULT")
done

# Build summary.json
SUMMARY_JSON="$SWEEP_DIR/summary.json"
echo ""
echo "Building summary → $SUMMARY_JSON"

SWEEP_SUMMARY="$SUMMARY_JSON" \
"$CONDA_PYTHON" - "${ALL_RESULTS[@]}" <<'PYEOF'
import json, os, sys, pathlib

entries = sys.argv[1:]
summary = []
for entry in entries:
    offset_str, path = entry.split(":", 1)
    offset = float(offset_str)
    try:
        data = json.loads(open(path).read())
        for r in data:
            r["offset"] = offset
        summary.extend(data)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}", flush=True)

summary_path = pathlib.Path(os.environ["SWEEP_SUMMARY"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2))
print(f"Summary written to {summary_path}  ({len(summary)} rows)", flush=True)

print(f"\n{'offset':>8}  {'dataset':<12}  {'routing_acc':>12}  "
      f"{'nECS':>8}  {'mean_B':>8}  {'savings%':>9}")
print("-" * 66)
for r in sorted(summary, key=lambda x: (x["dataset"], x["offset"])):
    print(f"{r['offset']:>8.2f}  {r['dataset']:<12}  {r['routing_accuracy']:>12.4f}  "
          f"{r['nECS']:>8.4f}  {r['mean_active_params_B']:>8.3f}  "
          f"{r['cost_savings_pct']:>8.1f}%")
PYEOF

echo ""
echo "Sweep complete. Results in $SWEEP_DIR/"
