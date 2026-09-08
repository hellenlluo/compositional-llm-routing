#!/bin/bash
#SBATCH --job-name=cos484-30b-v2
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/logs/30b-v2-%j.out
#SBATCH --error=outputs/logs/30b-v2-%j.err
#SBATCH --requeue

# Qwen3-30B-A3B with vLLM tensor-parallel-size=2 (set in models.yaml).
# Same reliability fixes as the v2 array script.

set -euo pipefail

MODEL="qwen3-30b-a3b"
REPO=~/cos484-final-project
cd "$REPO"

# preflight ECC
BAD=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null \
    | awk '{if($1 != "[N/A]" && $1+0 > 0) print "bad"}')
if [ -n "$BAD" ]; then
    echo "PREFLIGHT: uncorrected ECC on $(hostname) — requeueing"
    scontrol requeue $SLURM_JOB_ID
    exit 1
fi

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
if [ ! -x /scratch/ez2103/conda_envs/MARIO/bin/python ]; then
    bash "$REPO/setup_env.sh"
fi
conda activate /scratch/ez2103/conda_envs/MARIO

export HF_HOME=/scratch/ez2103/.cache/huggingface
mkdir -p "$HF_HOME"
TOKEN=""
[ -n "${HF_TOKEN:-}" ] && TOKEN="$HF_TOKEN"
for CAND in /u/ez2103/.cache/huggingface/token /auto/u/ez2103/.cache/huggingface/token "$HOME/.cache/huggingface/token"; do
    [ -n "$TOKEN" ] && break
    TOKEN=$(cat "$CAND" 2>/dev/null || true)
done
if [ -n "$TOKEN" ]; then
    printf '%s' "$TOKEN" > "$HF_HOME/token"
    chmod 600 "$HF_HOME/token"
    export HF_TOKEN="$TOKEN"
fi

CACHE_DB=/scratch/ez2103/cache_${MODEL}.db
HOME_CACHE=$REPO/outputs/cache.db
if [ -f "$HOME_CACHE" ] && [ ! -f "$CACHE_DB" ]; then
    sqlite3 "$HOME_CACHE" ".backup '$CACHE_DB'" 2>/dev/null || true
fi

echo "=== $(date) : $MODEL on $(hostname) ==="
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
            --batch-size 32 \
            --cache-db "$CACHE_DB"
    done
done

echo
python -m eval.export_runs --model "$MODEL" --cache-db "$CACHE_DB"

python - <<PY
import sqlite3
src = sqlite3.connect("$CACHE_DB"); dst = sqlite3.connect("$HOME_CACHE")
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
dst.commit(); print(f"merged {n} rows")
PY

echo "=== $(date) : done ==="
