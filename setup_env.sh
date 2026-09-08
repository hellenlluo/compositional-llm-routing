#!/usr/bin/env bash
# Create and populate the `MARIO` conda env for cos484 baseline evaluations.
# The env lives on scratch so it doesn't fill the 16 GB home partition.
# NOTE: scratch is per-node on this cluster. Re-run this script on each new
# compute node you `srun` into (the pip install step is ~10 min fresh).
#
# Usage: bash setup_env.sh
# After it finishes:  conda activate /n/fs/scratch/dl3533/conda_envs/MARIO

set -euo pipefail

ENV_PREFIX="/n/fs/scratch/dl3533/conda_envs/MARIO"
PY_VERSION="3.11"
SCRATCH_CACHE="/n/fs/scratch/dl3533/.cache/huggingface"

mkdir -p "$(dirname "$ENV_PREFIX")" "$SCRATCH_CACHE"

# Make `conda activate` work inside this script.
eval "$(conda shell.bash hook)"

# If an empty/leftover MARIO exists in $HOME, warn the user and skip it —
# prefix envs take precedence and avoid the home-disk issue.
if [ -d "$HOME/.conda/envs/MARIO" ]; then
    echo "NOTE: $HOME/.conda/envs/MARIO also exists (ignore it — scratch env is primary)."
fi

if [ -d "$ENV_PREFIX" ] && [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "scratch env already exists at $ENV_PREFIX — reusing."
else
    conda create -p "$ENV_PREFIX" "python=$PY_VERSION" -y
fi

conda activate "$ENV_PREFIX"

pip install --upgrade pip

# Core inference stack. transformers >=4.50 is required for Gemma-3 / MedGemma-1.5.
pip install \
    "torch>=2.6" \
    "transformers>=4.50" \
    "tokenizers>=0.20" \
    accelerate \
    sentencepiece \
    protobuf \
    pyyaml \
    pandas

# vLLM (fast inference). If it fails (CUDA mismatch), the HF backend still works.
pip install "vllm>=0.7.0" || echo "WARN: vllm install failed; will fall back to HF backend."

# Persist HF cache to scratch for this env.
conda env config vars set \
    HF_HOME="$SCRATCH_CACHE" \
    TRANSFORMERS_CACHE="$SCRATCH_CACHE" \
    -p "$ENV_PREFIX"

echo
echo "=== done ==="
du -sh "$ENV_PREFIX" || true
echo
echo "Next:"
echo "  conda activate $ENV_PREFIX"
echo "  huggingface-cli login   # needed for Llama-3.1, Gemma-3, MedGemma, Mistral (gated)"
echo "  python -m eval.run_all --limit 10   # smoke test"
