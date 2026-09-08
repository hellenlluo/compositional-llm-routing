#!/usr/bin/env python3
"""Download all-mpnet-base-v2 into this directory."""

import os
from sentence_transformers import SentenceTransformer

SAVE_DIR = "/n/fs/scratch/dl3533/models/all-mpnet-base-v2"

print(f"Downloading all-mpnet-base-v2 -> {SAVE_DIR}")
model = SentenceTransformer("all-mpnet-base-v2")
model.save(SAVE_DIR)
print("Done.")
