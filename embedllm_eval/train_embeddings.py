#!/usr/bin/env python3
"""
Train a per-model embedding + linear classifier on top of frozen MPNet embeddings.

Architecture (per sample):
  q768 = MPNet(prompt)               -- frozen, 768-dim
  q    = Projection(q768)            -- learned Linear(768 -> embed_dim), default 232-dim
  p    = ModelEmbedding[model_id]    -- learned, embed_dim-dim
  x    = q * p                       -- element-wise product, embed_dim-dim
  y    = w · x + b                   -- Linear(embed_dim -> 1), scalar
  out  = sigmoid(y)                  -- binary probability
  loss = BCELoss(out, label)

Data: embedllm_eval/results/{model}/all_results.json for all 10 models.
Prompts are encoded once with MPNet then cached to disk for fast reuse.
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MODELS_DIR  = "/n/fs/scratch/dl3533/models"
MPNET_PATH  = os.path.join(MODELS_DIR, "all-mpnet-base-v2")
EMBED_CACHE = os.path.join(MODELS_DIR, "prompt_embeddings_cache.pt")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_records(results_dir: str, exclude: set[str] | None = None) -> list[dict]:
    skip = exclude or set()
    records = []
    for model_dir in sorted(os.listdir(results_dir)):
        if model_dir in skip:
            print(f"  Skipping {model_dir} (excluded)")
            continue
        path = os.path.join(results_dir, model_dir, "all_results.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            records.extend(json.load(f))
    return records


def build_vocab(records: list[dict]) -> tuple[dict, dict]:
    """Return (model_name -> id, prompt_id -> index) mappings."""
    model_names = sorted({r["model_name"] for r in records})
    prompt_ids  = sorted({r["prompt_id"]   for r in records})
    model2id  = {m: i for i, m in enumerate(model_names)}
    prompt2idx = {p: i for i, p in enumerate(prompt_ids)}
    return model2id, prompt2idx


def get_prompt_texts(records: list[dict]) -> dict[int, str]:
    """Return {prompt_id: prompt_text} (deduplicated)."""
    return {r["prompt_id"]: r["prompt"] for r in records}


# ---------------------------------------------------------------------------
# MPNet embeddings (cached)
# ---------------------------------------------------------------------------

MPNET_MAX_TOKENS = 384  # all-mpnet-base-v2 hard limit
MPNET_CHARS_PER_TOKEN = 4  # conservative estimate for tail-truncation
TAIL_CHARS = MPNET_MAX_TOKENS * MPNET_CHARS_PER_TOKEN  # 1536 chars


def tail_truncate(text: str) -> str:
    """Keep the last TAIL_CHARS characters so the actual question (which
    appears at the end of n-shot prompts) always falls within MPNet's
    384-token window rather than being cut off by front-truncation."""
    return text[-TAIL_CHARS:] if len(text) > TAIL_CHARS else text


def build_or_load_prompt_embeddings(
    prompt2idx: dict[int, int],
    prompt_texts: dict[int, str],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Returns a float32 tensor of shape (num_prompts, 768) on CPU.
    Computes once and caches to EMBED_CACHE.
    """
    if os.path.isfile(EMBED_CACHE):
        print(f"Loading cached prompt embeddings from {EMBED_CACHE}")
        return torch.load(EMBED_CACHE, map_location="cpu")

    print(f"Encoding {len(prompt2idx)} prompts with MPNet (this may take a while)...")
    mpnet_path = MPNET_PATH if os.path.isdir(MPNET_PATH) else "all-mpnet-base-v2"
    encoder = SentenceTransformer(mpnet_path, device=str(device))

    ordered_ids = sorted(prompt2idx, key=lambda pid: prompt2idx[pid])
    texts = [tail_truncate(prompt_texts[pid]) for pid in ordered_ids]

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    os.makedirs(os.path.dirname(EMBED_CACHE), exist_ok=True)
    torch.save(tensor, EMBED_CACHE)
    print(f"Saved prompt embeddings to {EMBED_CACHE}")
    return tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    def __init__(self, records: list[dict], model2id: dict, prompt2idx: dict):
        self.model_ids  = torch.tensor([model2id[r["model_name"]] for r in records], dtype=torch.long)
        self.prompt_idxs = torch.tensor([prompt2idx[r["prompt_id"]] for r in records], dtype=torch.long)
        self.labels     = torch.tensor([float(r["score"]) for r in records], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.model_ids[i], self.prompt_idxs[i], self.labels[i]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EmbedClassifier(nn.Module):
    """
    q768 = prompt_emb[prompt_idx]          (768-dim, frozen MPNet)
    q    = projection(q768)                (embed_dim-dim, learned)
    p    = model_emb[model_id]             (embed_dim-dim, learned)
    x    = p * q                           (element-wise)
    out  = sigmoid(Linear(embed_dim->1)(x))
    """
    def __init__(self, num_models: int, prompt_embeddings: torch.Tensor, embed_dim: int = 232):
        super().__init__()
        mpnet_dim = prompt_embeddings.shape[1]  # 768

        # Learned projection: MPNet 768-dim -> embed_dim
        self.projection = nn.Linear(mpnet_dim, embed_dim)

        # Learned model embeddings
        self.model_emb = nn.Embedding(num_models, embed_dim)
        nn.init.normal_(self.model_emb.weight, mean=0.0, std=0.01)

        # Frozen prompt embeddings (768-dim MPNet features)
        self.prompt_emb = nn.Embedding.from_pretrained(
            prompt_embeddings, freeze=True
        )

        # Linear classifier: w (embed_dim,) + b (scalar)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, model_ids: torch.Tensor, prompt_idxs: torch.Tensor) -> torch.Tensor:
        q768 = self.prompt_emb(prompt_idxs)      # (B, 768) frozen
        q    = self.projection(q768)              # (B, embed_dim) learned
        p    = self.model_emb(model_ids)          # (B, embed_dim) learned
        x    = p * q                              # element-wise, (B, embed_dim)
        y    = self.classifier(x).squeeze(1)      # (B,)
        return torch.sigmoid(y)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for model_ids, prompt_idxs, labels in loader:
            model_ids   = model_ids.to(device)
            prompt_idxs = prompt_idxs.to(device)
            labels      = labels.to(device)

            preds = model(model_ids, prompt_idxs)
            loss  = criterion(preds, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * len(labels)
            total_correct += ((preds >= 0.5) == labels.bool()).sum().item()
            total         += len(labels)

    return total_loss / total, total_correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=2048)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--embed-dim",   type=int,   default=232,
                   help="Dimension of the learned model embeddings and projection output.")
    p.add_argument("--save-path",   type=str,
                   default=os.path.join(MODELS_DIR, "embedclassifier.pt"))
    p.add_argument("--exclude-models", nargs="*", default=[],
                   help="Model names to exclude from training "
                        "(e.g. qwen3-30b-a3b qwen3-4b-thinking-2507)")
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load records ---
    exclude_set = set(args.exclude_models)
    if exclude_set:
        print(f"Excluding models: {sorted(exclude_set)}")
    print("Loading records...")
    records = load_all_records(RESULTS_DIR, exclude_set)
    print(f"  {len(records)} total records")

    model2id, prompt2idx = build_vocab(records)
    prompt_texts = get_prompt_texts(records)
    print(f"  {len(model2id)} models, {len(prompt2idx)} unique prompts")

    # --- Prompt embeddings ---
    prompt_embeddings = build_or_load_prompt_embeddings(
        prompt2idx, prompt_texts, device
    )

    # --- Use all data for training ---
    train_ds = PairDataset(records, model2id, prompt2idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"  Training on all {len(records)} records")

    # --- Model ---
    print(f"  embed_dim={args.embed_dim} (projection: 768 -> {args.embed_dim})")
    net = EmbedClassifier(
        num_models=len(model2id),
        prompt_embeddings=prompt_embeddings,
        embed_dim=args.embed_dim,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(net, train_loader, criterion, optimizer, device, train=True)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={tr_loss:.4f}  acc={tr_acc:.4f}")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": net.state_dict(),
        "model2id": model2id,
        "prompt2idx": prompt2idx,
        "embed_dim": args.embed_dim,
    }, args.save_path)
    print(f"\nDone. Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
