#!/usr/bin/env python3
"""
Train the EfficiencyRouter on MMLU training pairs.

Training data:
  modelsat-baseline/modelsat-data/training_pairs.jsonl   (untrimmed, 3000 questions)
  Each record: {model_name, prompt_id, prompt, score, ...}
  We have correctness labels for all 10 models on every question.

Soft label construction (per question):
  raw_m  = (offset + 1 / params_active_b_m)        if score_m == 1, else 0
  label  = raw / sum(raw)                          if sum(raw) > 0, else skip

  offset=0   → pure efficiency, 16x ratio between 0.5B and 8B models
  offset=1   → moderate efficiency preference, ~2.7x ratio (default)
  offset=10  → near-uniform, correctness dominates over efficiency
  offset=∞   → equivalent to binary uniform-over-correct labels

Loss: soft cross-entropy + optional pairwise margin ranking loss
  total = soft_ce + rank_weight * pairwise_margin_loss

Encoder modes:
  --encoder PATH                  pick any sentence-transformers model
                                  (default: all-mpnet-base-v2)
  --finetune-encoder              joint encoder + routing head training,
                                  unfreezes the last --finetune-layers blocks
                                  of the encoder. Otherwise the encoder is
                                  frozen and prompt embeddings are cached.

Checkpoint saved to efficiency_router/checkpoints/<ckpt-name>.pt
"""

import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

from model import (
    EfficiencyRouter,
    EfficiencyRouterFinetune,
    soft_ce_loss,
    pairwise_margin_loss,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = "/n/fs/scratch/dl3533/models"

# Default encoder: Qwen3-Embedding-0.6B (June 2025).
# Top open-source MTEB model in its size class, 1024-dim, 32K context.
# Falls back to the HF hub name if the local copy isn't present.
DEFAULT_ENC_LOCAL = os.path.join(MODELS_DIR, "Qwen3-Embedding-0.6B")
DEFAULT_ENC_HUB   = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_ENC = DEFAULT_ENC_LOCAL if os.path.isdir(DEFAULT_ENC_LOCAL) else DEFAULT_ENC_HUB

PAIRS_FILE   = os.path.join(PROJECT_DIR, "modelsat-baseline", "modelsat-data", "training_pairs.jsonl")
CKPT_DIR     = os.path.join(SCRIPT_DIR, "checkpoints")
EMBED_CACHE_DIR = MODELS_DIR

# Active parameter counts (billions) for all 10 training models
MODEL_PARAMS_ACTIVE_B: dict[str, float] = {
    "deepseek-r1-distill-llama-8b":   8.0,
    "llama-3.1-8b-instruct":          8.0,
    "llama-3.1-nemotron-nano-8b":     8.0,
    "mathstral-7b":                   7.0,
    "medgemma-4b-it":                 4.0,
    "mistral-7b-instruct-v0.3":       7.2,
    "phi-4-mini-instruct":            3.8,
    "qwen1.5-0.5b-chat":              0.5,
    "qwen3-30b-a3b":                  3.0,
    "qwen3-4b-thinking-2507":         4.0,
}

# Conservative tail-truncation in characters. Most modern encoders accept
# 512 tokens (BGE, GTE) or more (Snowflake, Jina) but encoding speed grows
# linearly with sequence length, so we keep prompts to ~1500 chars.
PROMPT_MAX_CHARS = 1536


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_pairs(
    path: str,
    exclude_models: set[str] | None = None,
) -> tuple[dict, dict, list[str]]:
    """
    Returns:
      prompt_texts:  {prompt_id: prompt_text}
      model_scores:  {prompt_id: {model_name: score}}
      model_names:   sorted list of all model names (after exclusions)
    """
    exclude = exclude_models or set()
    prompt_texts: dict[int, str] = {}
    model_scores: dict[int, dict[str, int]] = defaultdict(dict)

    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["model_name"] in exclude:
                continue
            prompt_texts[r["prompt_id"]] = r["prompt"]
            model_scores[r["prompt_id"]][r["model_name"]] = r["score"]

    model_names = sorted(m for m in MODEL_PARAMS_ACTIVE_B if m not in exclude)
    return prompt_texts, dict(model_scores), model_names


def build_soft_labels(
    model_scores: dict[int, dict[str, int]],
    model_names:  list[str],
    offset:       float,
) -> dict[int, "torch.Tensor | None"]:
    """
    raw_m = (offset + 1 / params_active_b_m) if score_m == 1, else 0
    label = raw / sum(raw)  (None if sum(raw) == 0)
    """
    labels: dict[int, "torch.Tensor | None"] = {}
    for pid, scores in model_scores.items():
        raw = torch.zeros(len(model_names))
        for i, m in enumerate(model_names):
            if scores.get(m, 0) > 0:
                raw[i] = offset + 1.0 / MODEL_PARAMS_ACTIVE_B[m]
        total = raw.sum().item()
        labels[pid] = None if total == 0 else raw / total
    return labels


def truncate(text: str) -> str:
    return text[-PROMPT_MAX_CHARS:] if len(text) > PROMPT_MAX_CHARS else text


# ---------------------------------------------------------------------------
# Cached embeddings (frozen encoder mode only)
# ---------------------------------------------------------------------------

def _encoder_cache_name(encoder_path: str) -> str:
    """Stable cache filename derived from the encoder model name."""
    base = os.path.basename(os.path.normpath(encoder_path)).replace("/", "_")
    return f"effrouter_train_embeddings__{base}.pt"


def build_or_load_embeddings(
    encoder_path: str,
    prompt2idx:   dict[int, int],
    prompt_texts: dict[int, str],
    device:       torch.device,
    batch_size:   int = 32,
) -> torch.Tensor:
    cache_path = os.path.join(EMBED_CACHE_DIR, _encoder_cache_name(encoder_path))
    if os.path.isfile(cache_path):
        print(f"Loading cached train embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"Encoding {len(prompt2idx)} prompts with {encoder_path}...")
    encoder = SentenceTransformer(encoder_path, device=str(device))
    ordered_ids = sorted(prompt2idx, key=lambda pid: prompt2idx[pid])
    texts = [truncate(prompt_texts[pid]) for pid in ordered_ids]
    embs = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    )
    tensor = torch.tensor(embs, dtype=torch.float32)
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    torch.save(tensor, cache_path)
    print(f"Saved to {cache_path}")
    return tensor


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class FrozenDataset(Dataset):
    """Each item is (prompt_idx, soft_label) for the frozen-encoder path."""

    def __init__(
        self,
        soft_labels:  dict[int, "torch.Tensor | None"],
        prompt2idx:   dict[int, int],
    ):
        self.items = [
            (prompt2idx[pid], label)
            for pid, label in soft_labels.items()
            if label is not None
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        prompt_idx, label = self.items[i]
        return torch.tensor(prompt_idx, dtype=torch.long), label


class FinetuneDataset(Dataset):
    """Each item is (prompt_text, soft_label) for the joint-encoder path."""

    def __init__(
        self,
        soft_labels:  dict[int, "torch.Tensor | None"],
        prompt_texts: dict[int, str],
    ):
        self.items = [
            (truncate(prompt_texts[pid]), label)
            for pid, label in soft_labels.items()
            if label is not None
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def finetune_collate(batch):
    """Stack texts as a list and labels as a tensor."""
    texts  = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch], dim=0)
    return texts, labels


# ---------------------------------------------------------------------------
# Training loop (shared between modes)
# ---------------------------------------------------------------------------

def total_loss(
    logits:       torch.Tensor,
    soft_targets: torch.Tensor,
    rank_weight:  float,
    margin:       float,
) -> tuple[torch.Tensor, float, float]:
    """Returns (loss, ce_value, rank_value) where ce/rank are float scalars."""
    ce = soft_ce_loss(logits, soft_targets)
    if rank_weight > 0:
        rank = pairwise_margin_loss(logits, soft_targets, margin=margin)
        loss = ce + rank_weight * rank
        return loss, ce.item(), rank.item()
    return ce, ce.item(), 0.0


def run_epoch_frozen(
    model:       EfficiencyRouter,
    loader:      DataLoader,
    optimizer:   "torch.optim.Optimizer | None",
    device:      torch.device,
    rank_weight: float,
    margin:      float,
    train:       bool,
) -> tuple[float, float]:
    model.train(train)
    total, total_correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for prompt_idxs, soft_targets in loader:
            prompt_idxs  = prompt_idxs.to(device)
            soft_targets = soft_targets.to(device)
            logits = model(prompt_idxs)
            loss, _, _ = total_loss(logits, soft_targets, rank_weight, margin)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds   = logits.argmax(dim=-1)
            targets = soft_targets.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total        += loss.item() * len(prompt_idxs)
            n            += len(prompt_idxs)
    return total / n, total_correct / n


def run_epoch_finetune(
    model:       EfficiencyRouterFinetune,
    loader:      DataLoader,
    optimizer:   "torch.optim.Optimizer | None",
    device:      torch.device,
    rank_weight: float,
    margin:      float,
    train:       bool,
) -> tuple[float, float]:
    model.train(train)
    total, total_correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for texts, soft_targets in loader:
            soft_targets = soft_targets.to(device)
            logits = model(texts)
            loss, _, _ = total_loss(logits, soft_targets, rank_weight, margin)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds   = logits.argmax(dim=-1)
            targets = soft_targets.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total        += loss.item() * len(soft_targets)
            n            += len(soft_targets)
    return total / n, total_correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # --- Encoder ---
    p.add_argument("--encoder", type=str, default=DEFAULT_ENC,
                   help="HF/sentence-transformers model path or hub name "
                        "(e.g. BAAI/bge-large-en-v1.5)")
    p.add_argument("--finetune-encoder", action="store_true",
                   help="Train the encoder jointly with the routing head.")
    p.add_argument("--finetune-layers", type=int, default=4,
                   help="Number of last transformer blocks to unfreeze "
                        "(only used if --finetune-encoder).")
    p.add_argument("--encoder-lr", type=float, default=2e-5,
                   help="LR for encoder params (only used if --finetune-encoder).")

    # --- Routing head ---
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--lr",        type=float, default=1e-3,
                   help="LR for routing head (proj, model_emb, model_bias).")

    # --- Loss ---
    p.add_argument("--offset",      type=float, default=1.0,
                   help="Soft label offset: 0=pure efficiency (16x ratio), "
                        "1=moderate (~2.7x), 10=near-uniform")
    p.add_argument("--rank-weight", type=float, default=0.0,
                   help="Coefficient for pairwise margin ranking loss "
                        "(0 = soft CE only).")
    p.add_argument("--margin",      type=float, default=1.0,
                   help="Margin for pairwise ranking loss.")

    # --- Optimization ---
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=512,
                   help="Larger for frozen mode (fits easily); reduce for "
                        "fine-tune mode (e.g. 32-64) to fit encoder activations.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed",         type=int,   default=42)

    # --- Model filtering ---
    p.add_argument("--exclude-models", nargs="*", default=[],
                   help="Model names to exclude from training (e.g. qwen3-30b-a3b qwen3-4b-thinking-2507)")

    # --- I/O ---
    p.add_argument("--ckpt-name", type=str, default="effrouter")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:           {device}")
    print(f"Encoder:          {args.encoder}")
    print(f"Mode:             {'FINETUNE' if args.finetune_encoder else 'FROZEN'}")
    print(f"Soft label offset: {args.offset}")
    print(f"Rank weight:      {args.rank_weight} (margin={args.margin})")

    # --- Data ---
    exclude_set = set(args.exclude_models)
    if exclude_set:
        print(f"Excluding models: {sorted(exclude_set)}")
    print(f"\nLoading training pairs from {PAIRS_FILE}...")
    prompt_texts, model_scores, model_names = load_training_pairs(PAIRS_FILE, exclude_set)
    print(f"  {len(prompt_texts)} questions, {len(model_names)} models")

    soft_labels = build_soft_labels(model_scores, model_names, offset=args.offset)
    usable  = sum(1 for v in soft_labels.values() if v is not None)
    skipped = len(soft_labels) - usable
    print(f"  Usable (≥1 correct): {usable}  |  Skipped (all wrong): {skipped}")

    # ===================================================================
    # Path A — Frozen encoder (cached embeddings)
    # ===================================================================
    if not args.finetune_encoder:
        prompt2idx = {pid: i for i, pid in enumerate(sorted(prompt_texts))}
        prompt_embeddings = build_or_load_embeddings(
            args.encoder, prompt2idx, prompt_texts, device,
        )

        ds = FrozenDataset(soft_labels, prompt2idx)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print(f"\nTraining (frozen) on {len(ds)} questions, {len(model_names)} models")

        net = EfficiencyRouter(
            num_models=len(model_names),
            prompt_embeddings=prompt_embeddings,
            embed_dim=args.embed_dim,
        ).to(device)
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.epochs + 1):
            loss, acc = run_epoch_frozen(
                net, loader, optimizer, device,
                args.rank_weight, args.margin, train=True,
            )
            print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  hard_acc={acc:.4f}")

        save_payload = {
            "mode":             "frozen",
            "model_state_dict": net.state_dict(),
            "model_names":      model_names,
            "prompt2idx":       prompt2idx,
            "embed_dim":        args.embed_dim,
            "offset":           args.offset,
            "rank_weight":      args.rank_weight,
            "margin":           args.margin,
            "encoder":          args.encoder,
            "epochs":           args.epochs,
        }

    # ===================================================================
    # Path B — Joint encoder + head (fine-tune the encoder)
    # ===================================================================
    else:
        encoder = SentenceTransformer(args.encoder, device=str(device))
        net = EfficiencyRouterFinetune(
            num_models=len(model_names),
            encoder=encoder,
            embed_dim=args.embed_dim,
            finetune_layers=args.finetune_layers,
        ).to(device)

        # Report parameter counts
        head_params = (
            list(net.proj.parameters())
            + list(net.model_emb.parameters())
            + [net.model_bias]
        )
        encoder_trainable = [p for p in net.encoder.parameters() if p.requires_grad]
        n_head    = sum(p.numel() for p in head_params)
        n_encoder = sum(p.numel() for p in encoder_trainable)
        print(f"\nTrainable head params:    {n_head:,}")
        print(f"Trainable encoder params: {n_encoder:,}  "
              f"(unfrozen last {args.finetune_layers} layers)")

        ds = FinetuneDataset(soft_labels, prompt_texts)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=finetune_collate,
        )
        print(f"Training (finetune) on {len(ds)} questions, {len(model_names)} models")

        # Two LR groups: high LR for the head, low LR for the encoder.
        optimizer = torch.optim.Adam(
            [
                {"params": head_params,       "lr": args.lr},
                {"params": encoder_trainable, "lr": args.encoder_lr},
            ],
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.epochs + 1):
            loss, acc = run_epoch_finetune(
                net, loader, optimizer, device,
                args.rank_weight, args.margin, train=True,
            )
            print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  hard_acc={acc:.4f}")

        save_payload = {
            "mode":             "finetune",
            "model_state_dict": net.state_dict(),
            "model_names":      model_names,
            "embed_dim":        args.embed_dim,
            "offset":           args.offset,
            "rank_weight":      args.rank_weight,
            "margin":           args.margin,
            "encoder":          args.encoder,
            "finetune_layers":  args.finetune_layers,
            "epochs":           args.epochs,
        }

    # --- Save ---
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{args.ckpt_name}.pt")
    torch.save(save_payload, ckpt_path)
    print(f"\nCheckpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
