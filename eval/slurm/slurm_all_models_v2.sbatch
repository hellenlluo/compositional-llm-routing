#!/bin/bash
#SBATCH --job-name=cos484-full-v2
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/v2-%A_%a.out
#SBATCH --error=outputs/logs/v2-%A_%a.err
#SBATCH --requeue

# Full-dataset inference v2. Fixes from v1:
#  - cache.db on per-node scratch (avoid NFS/SQLite concurrent-writer bug)
#  - HF token copied into node's HF_HOME so gated repos authenticate
#  - preflight ECC check requeues bad-GPU nodes
#  - vLLM backend everywhere (avoids transformers dynamic_module incompat;
#    Phi-4-mini's modeling_phi3.py uses LossKwargs removed in transformers 4.57)
#
# Before submitting, make sure you've run `huggingface-cli login` on the login
# node — gated models (medgemma/llama/mistral) will 401 without a token.

set -euo pipefail

MODELS=(
    qwen1.5-0.5b-chat              # 0
    phi-4-mini-instruct            # 1
    qwen3-4b-thinking-2507         # 2
    medgemma-4b-it                 # 3
    mistral-7b-instruct-v0.3       # 4
    llama-3.1-8b-instruct          # 5
    deepseek-r1-distill-llama-8b   # 6
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

REPO=~/cos484-final-project
cd "$REPO"

# --- preflight: reject GPUs with uncorrected ECC errors ---
BAD=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null \
    | awk '{if($1 != "[N/A]" && $1+0 > 0) print "bad"}')
if [ -n "$BAD" ]; then
    echo "PREFLIGHT: uncorrected ECC errors on $(hostname) — requeueing"
    scontrol requeue $SLURM_JOB_ID
    exit 1
fi
echo "preflight OK: $(hostname), GPUs clean"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
if [ ! -x /scratch/ez2103/conda_envs/MARIO/bin/python ]; then
    echo "installing MARIO on $(hostname)..."
    bash "$REPO/setup_env.sh"
fi
conda activate /scratch/ez2103/conda_envs/MARIO

# --- HF token: try multiple home paths (autofs can hide $HOME on first access) ---
export HF_HOME=/scratch/ez2103/.cache/huggingface
mkdir -p "$HF_HOME"
TOKEN=""
if [ -n "${HF_TOKEN:-}" ]; then
    TOKEN="$HF_TOKEN"
fi
for CAND in /u/ez2103/.cache/huggingface/token /auto/u/ez2103/.cache/huggingface/token "$HOME/.cache/huggingface/token"; do
    [ -n "$TOKEN" ] && break
    TOKEN=$(cat "$CAND" 2>/dev/null || true)
done
if [ -n "$TOKEN" ]; then
    printf '%s' "$TOKEN" > "$HF_HOME/token"
    chmod 600 "$HF_HOME/token"
    export HF_TOKEN="$TOKEN"
    echo "HF token loaded (${#TOKEN} chars)"
else
    echo "WARN: no HF token found. Gated models will 401."
fi

# --- per-node local cache on scratch (avoids NFS+SQLite concurrent-writer bug) ---
CACHE_DB=/scratch/ez2103/cache_${MODEL}.db
mkdir -p "$(dirname "$CACHE_DB")"
# seed cache from home if we already have inference for this model (e.g. qwen1.5)
HOME_CACHE=$REPO/outputs/cache.db
if [ -f "$HOME_CACHE" ] && [ ! -f "$CACHE_DB" ]; then
    echo "seeding $CACHE_DB from $HOME_CACHE"
    sqlite3 "$HOME_CACHE" ".backup '$CACHE_DB'" 2>/dev/null || true
fi

BATCH_SIZE=16

echo "=== $(date) : model=$MODEL  host=$(hostname)  cache=$CACHE_DB ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for DATASET in musique morehopqa stepcot; do
    for PHASE in full subtask; do
        echo
        echo "=== $(date) : $MODEL / $DATASET / $PHASE ==="
        python -m eval.run_inference \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --phase "$PHASE" \
            --no-limit \
            --backend vllm \
            --batch-size "$BATCH_SIZE" \
            --cache-db "$CACHE_DB"
    done
done

echo
echo "=== $(date) : exporting $MODEL JSONL (home-backed, persistent) ==="
python -m eval.export_runs --model "$MODEL" --cache-db "$CACHE_DB"

echo
echo "=== $(date) : merging scratch cache into home cache ==="
# Use Python since sqlite3 ATTACH on NFS home can flake; atomic per-row insert-or-ignore.
python - <<PY
import sqlite3
src = sqlite3.connect("$CACHE_DB")
dst = sqlite3.connect("$HOME_CACHE")
dst.execute("""CREATE TABLE IF NOT EXISTS runs (
  dataset TEXT, question_id TEXT, subtask_idx INTEGER, model_id TEXT, prompt_hash TEXT,
  prompt TEXT, response TEXT, gold TEXT, gold_aliases TEXT, gen_kwargs TEXT,
  wall_ms INTEGER, created_at TEXT, correct INTEGER, judge_id TEXT, judge_reason TEXT,
  PRIMARY KEY (dataset, question_id, subtask_idx, model_id, prompt_hash))""")
cols = "dataset,question_id,subtask_idx,model_id,prompt_hash,prompt,response,gold,gold_aliases,gen_kwargs,wall_ms,created_at,correct,judge_id,judge_reason"
n = 0
for row in src.execute(f"SELECT {cols} FROM runs WHERE model_id=?", ("$MODEL",)):
    dst.execute(f"INSERT OR IGNORE INTO runs ({cols}) VALUES ({','.join(['?']*15)})", row)
    n += 1
dst.commit()
print(f"merged {n} rows into home cache")
PY

echo "=== $(date) : done for $MODEL ==="
