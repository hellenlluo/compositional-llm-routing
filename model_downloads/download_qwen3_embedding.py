#!/usr/bin/env python3
"""Download Qwen3-Embedding-0.6B into the shared models directory.

Run this once on the cluster before training the efficiency router with the
Qwen3 encoder. The local copy avoids hitting HuggingFace Hub on every job.
"""

import os
from sentence_transformers import SentenceTransformer

MODELS_DIR = "/n/fs/scratch/dl3533/models"
HUB_NAME   = "Qwen/Qwen3-Embedding-0.6B"
LOCAL_DIR  = os.path.join(MODELS_DIR, "Qwen3-Embedding-0.6B")

print(f"Downloading {HUB_NAME} -> {LOCAL_DIR}")
model = SentenceTransformer(HUB_NAME)
model.save(LOCAL_DIR)
print("Done.")
