#!/usr/bin/env bash
# Called by submit_sweep.sh as the SLURM job script.
# Trains + evaluates one EfficiencyRouter per --offset value, then writes
# efficiency_router/results/sweep/summary.json aggregating all runs.

set -euo pipefail

SCRIPT_DIR="$1"
SWEEP_DIR="$2"
shift 2
# Remaining args are the offset values to run (allows sharding across jobs).
# If none provided, fall back to full grid 1-10.
if [ "$#" -gt 0 ]; then
    OFFSETS=("$@")
fi

CONDA_PYTHON="/n/fs/scratch/dl3533/conda_envs/MARIO/bin/python"

export HF_HOME="/n/fs/scratch/dl3533/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$SCRIPT_DIR"

# ── Sweep grid ───────────────────────────────────────────────────────────────
# offset controls the efficiency vs correctness trade-off in soft labels:
#   1  = strong efficiency preference (~2.7x ratio between 0.5B and 8B)
#   10 = near-uniform (correctness-heavy, tiny efficiency preference)
OFFSETS=(1 2 3 4 5 6 7 8 9 10)
# ─────────────────────────────────────────────────────────────────────────────

ALL_RESULTS=()

for OFFSET in "${OFFSETS[@]}"; do
    TAG="offset_${OFFSET}"
    CKPT="$SCRIPT_DIR/checkpoints/effrouter_${TAG}.pt"
    OUT_DIR="$SWEEP_DIR/${TAG}"
    RESULT="$OUT_DIR/eval_results.json"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  TRAIN  offset=$OFFSET"
    echo "════════════════════════════════════════════════════"
    "$CONDA_PYTHON" train.py \
        --offset "$OFFSET" \
        --ckpt-name "effrouter_${TAG}"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  EVAL   offset=$OFFSET"
    echo "════════════════════════════════════════════════════"
    "$CONDA_PYTHON" eval.py \
        --ckpt "$CKPT" \
        --out  "$RESULT"

    ALL_RESULTS+=("$OFFSET:$RESULT")
done

# ── Build summary.json ───────────────────────────────────────────────────────
SUMMARY_JSON="$SWEEP_DIR/summary.json"
echo ""
echo "Building summary → $SUMMARY_JSON"

SWEEP_SUMMARY="$SUMMARY_JSON" \
"$CONDA_PYTHON" - "${ALL_RESULTS[@]}" <<'PYEOF'
import json, os, sys, pathlib

entries = sys.argv[1:]   # "offset:path" pairs
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

# Quick comparison table
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
