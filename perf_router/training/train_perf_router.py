#!/usr/bin/env python3
"""Train an accuracy-augmented EmbedLLM router.

Architecture (per sample):
  p_learned = ModelEmbedding[model_id]       -- learned, 1024-dim (like EmbedLLM)
  p_acc     = accuracy_vec[model_id]         -- fixed, 88-dim benchmark accuracy profile
  p_combined = concat([p_learned, p_acc])    -- 1112-dim
  p         = ModelProjector(p_combined)     -- learned Linear(1112 → 1024)
  q         = Qwen3Embedding(prompt)         -- frozen, 1024-dim
  x         = p * q                          -- element-wise product, 1024-dim
  out       = sigmoid(Linear(1024→1)(x))     -- binary probability
  loss      = BCELoss(out, label)

Compared to the original EmbedLLM (train_embeddings.py):
  - The learned model embedding is augmented with fixed benchmark accuracy features
    before the element-wise interaction with the question embedding.
  - This grounds the model representations in interpretable capability dimensions.

Training data: embedllm-eval/results/{model}/all_results.json (same 10-model set).
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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR     = os.path.dirname(SCRIPT_DIR)           # perf_router/
PROJECT_DIR  = os.path.dirname(EVAL_DIR)             # project root

RESULTS_DIR  = os.path.join(PROJECT_DIR, "embedllm-eval", "results")
PERF_VECTORS = os.path.join(EVAL_DIR, "results", "qwen3", "accuracy_vectors.json")
MODELS_DIR   = "/n/fs/scratch/dl3533/models"
QWEN_LOCAL   = os.path.join(MODELS_DIR, "Qwen3-Embedding-0.6B")
QWEN_HUB     = "Qwen/Qwen3-Embedding-0.6B"
QWEN_PATH    = QWEN_LOCAL if os.path.isdir(QWEN_LOCAL) else QWEN_HUB
EMBED_CACHE  = os.path.join(MODELS_DIR, "prompt_embeddings_qwen3_0.6b.pt")
CKPT_DIR     = os.path.join(MODELS_DIR, "perf_router")

EMBED_DIM = 1024  # Qwen3-Embedding-0.6B hidden size

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_records(results_dir: str, exclude: set[str] | None = None) -> list[dict]:
    skip = exclude or set()
    records: list[dict] = []
    for model_dir in sorted(os.listdir(results_dir)):
        if model_dir in skip:
            continue
        path = os.path.join(results_dir, model_dir, "all_results.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            records.extend(json.load(f))
    return records


def load_perf_vectors(path: str) -> tuple[list[str], dict[str, list[float]]]:
    """Return (feature_names, {model_id: [float, ...]})."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Performance vectors not found at {path}.\n"
            f"Run build_perf_vectors.py first."
        )
    with open(path) as f:
        data = json.load(f)
    return data["feature_names"], data["vectors"]


def build_vocab(
    records: list[dict],
    perf_model_ids: set[str],
) -> tuple[dict[str, int], dict[int, int]]:
    """Return (model_name → id, prompt_id → index).

    Only keeps models that have both training records AND perf vectors.
    """
    model_names = sorted(
        {r["model_name"] for r in records} & perf_model_ids
    )
    prompt_ids  = sorted({r["prompt_id"] for r in records})
    model2id   = {m: i for i, m in enumerate(model_names)}
    prompt2idx = {p: i for i, p in enumerate(prompt_ids)}
    return model2id, prompt2idx


def get_prompt_texts(records: list[dict]) -> dict[int, str]:
    return {r["prompt_id"]: r["prompt"] for r in records}


# ---------------------------------------------------------------------------
# Qwen3-Embedding-0.6B question embeddings (cached)
# ---------------------------------------------------------------------------

# Qwen3-Embedding-0.6B supports up to 32K tokens; cap at 512 for speed.
QWEN_MAX_TOKENS    = 512
QWEN_CHARS_PER_TOK = 4
TAIL_CHARS         = QWEN_MAX_TOKENS * QWEN_CHARS_PER_TOK


def tail_truncate(text: str) -> str:
    return text[-TAIL_CHARS:] if len(text) > TAIL_CHARS else text


def build_or_load_prompt_embeddings(
    prompt2idx: dict[int, int],
    prompt_texts: dict[int, str],
    device: torch.device,
    batch_size: int = 32,
    cache_path: str = EMBED_CACHE,
) -> torch.Tensor:
    """Encode all prompts with MPNet and cache to disk."""
    if os.path.isfile(cache_path):
        print(f"  Loading cached prompt embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"  Encoding {len(prompt2idx)} prompts with Qwen3-Embedding-0.6B...")
    encoder = SentenceTransformer(QWEN_PATH, device=str(device))

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
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(tensor, cache_path)
    print(f"  Saved prompt embeddings to {cache_path}")
    return tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContrastiveDataset(Dataset):
    """Per-question contrastive dataset.

    Each item represents one question with ALL model scores grouped together.
    The training loss is a per-question cross-entropy / soft-CE, forcing the
    router to rank correct models above incorrect ones rather than learning
    per-model binary thresholds in isolation (which collapses to always picking
    the globally strongest model).

    soft_label[m] = score[m] / sum(scores)   (uniform over correct models)
    """

    def __init__(
        self,
        records: list[dict],
        model2id: dict[str, int],
        prompt2idx: dict[int, int],
    ):
        all_models = sorted(model2id.keys())
        num_models = len(all_models)

        # Group by prompt_id → {model_name: score}
        by_prompt: dict[int, dict[str, int]] = {}
        for r in records:
            if r["model_name"] not in model2id or r["prompt_id"] not in prompt2idx:
                continue
            by_prompt.setdefault(r["prompt_id"], {})[r["model_name"]] = r["score"]

        model_id_vec = torch.tensor([model2id[m] for m in all_models], dtype=torch.long)

        self.prompt_idxs: list[torch.Tensor] = []
        self.model_id_vecs: list[torch.Tensor] = []
        self.soft_labels: list[torch.Tensor] = []

        for pid, m_scores in by_prompt.items():
            scores = torch.tensor(
                [float(m_scores.get(m, 0)) for m in all_models], dtype=torch.float32
            )
            total = scores.sum().item()
            if total == 0:
                continue  # skip all-wrong questions — no training signal
            soft = scores / total
            self.prompt_idxs.append(torch.tensor(prompt2idx[pid], dtype=torch.long))
            self.model_id_vecs.append(model_id_vec)
            self.soft_labels.append(soft)

    def __len__(self):
        return len(self.soft_labels)

    def __getitem__(self, i):
        # (scalar, (M,), (M,))
        return self.prompt_idxs[i], self.model_id_vecs[i], self.soft_labels[i]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AccuracyAugmentedEmbedRouter(nn.Module):
    """Accuracy-augmented EmbedLLM router.

    Model side:
      p_learned   = ModelEmbedding[model_id]     learned 1024-dim (like EmbedLLM)
      p_acc       = accuracy_matrix[model_id]    fixed   D-dim benchmark scores
      p_combined  = concat([p_learned, p_acc])   (1024+D)-dim
      p           = model_proj(p_combined)        learned Linear(1024+D → 1024)

    Question side:
      q = prompt_emb[prompt_idx]                 frozen Qwen3-Embedding-0.6B 1024-dim

    Score:
      x   = p * q                                element-wise product 1024-dim
      out = sigmoid(classifier(x))               Linear(1024→1) + sigmoid
    """

    def __init__(
        self,
        num_models: int,
        accuracy_matrix: torch.Tensor,    # (num_models, D) — fixed
        prompt_embeddings: torch.Tensor,  # (num_prompts, 1024) — frozen
        embed_dim: int = EMBED_DIM,
        dropout: float = 0.0,
    ):
        super().__init__()
        acc_dim = accuracy_matrix.shape[1]   # 88

        # Fixed accuracy vectors
        self.register_buffer("accuracy_matrix", accuracy_matrix.float())

        # Learned model embeddings (same as EmbedLLM)
        self.model_emb = nn.Embedding(num_models, embed_dim)
        nn.init.normal_(self.model_emb.weight, mean=0.0, std=0.01)

        # Frozen question embeddings
        self.prompt_emb = nn.Embedding.from_pretrained(prompt_embeddings, freeze=True)

        # Project concat([model_emb, acc_vec]) → embed_dim
        self.model_proj = nn.Linear(embed_dim + acc_dim, embed_dim)
        nn.init.kaiming_normal_(self.model_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.model_proj.bias)

        self.dropout = nn.Dropout(p=dropout)

        # Classifier: Linear(1024→1) same as EmbedLLM
        self.classifier = nn.Linear(embed_dim, 1)

    def logits(
        self,
        model_ids: torch.Tensor,    # (B,) or (B, M)
        prompt_idxs: torch.Tensor,  # (B,) or (B, M)
    ) -> torch.Tensor:
        """Raw scores (no sigmoid). Used for contrastive cross-entropy."""
        p_learned  = self.model_emb(model_ids)
        p_acc      = self.accuracy_matrix[model_ids]
        p_combined = torch.cat([p_learned, p_acc], dim=-1)
        p = self.dropout(self.model_proj(p_combined))
        q = self.prompt_emb(prompt_idxs)
        x = self.dropout(p * q)
        return self.classifier(x).squeeze(-1)

    def forward(
        self,
        model_ids: torch.Tensor,    # (B,)
        prompt_idxs: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Sigmoid-gated scores. Used at inference time."""
        return torch.sigmoid(self.logits(model_ids, prompt_idxs))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool, loss_type: str = "soft_ce"):
    """Per-question training.

    loss_type="soft_ce" (default): contrastive soft cross-entropy across all
        models per question — forces question-specific routing signal.
    loss_type="bce": independent binary cross-entropy per (question, model)
        pair — each model treated as a separate binary classifier.
    """
    import torch.nn.functional as F
    model.train(train)
    total_loss = total_correct = total = 0
    with torch.set_grad_enabled(train):
        for prompt_idx, model_id_vecs, soft_labels in loader:
            B, M = model_id_vecs.shape
            model_id_vecs = model_id_vecs.to(device)
            soft_labels   = soft_labels.to(device)

            prompt_idxs = prompt_idx.unsqueeze(1).expand(B, M).to(device)
            raw = model.logits(
                model_id_vecs.reshape(-1),
                prompt_idxs.reshape(-1),
            ).view(B, M)                               # (B, M)

            if loss_type == "soft_ce":
                log_probs = F.log_softmax(raw, dim=1)  # (B, M)
                loss = -(soft_labels * log_probs).sum(dim=1).mean()
            else:  # bce
                # soft_labels are in [0,1]; treat as binary targets per slot
                loss = F.binary_cross_entropy_with_logits(raw, soft_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            chosen = raw.argmax(dim=1)                 # (B,)
            correct = (soft_labels[torch.arange(B), chosen] > 0).sum().item()

            total_loss    += loss.item() * B
            total_correct += correct
            total         += B

    return total_loss / total, total_correct / total


# ---------------------------------------------------------------------------
# Routing evaluation
# ---------------------------------------------------------------------------

def evaluate_routing(
    model: "AccuracyAugmentedEmbedRouter",
    records: list[dict],
    model2id: dict[str, int],
    prompt2idx: dict[int, int],
    device: torch.device,
) -> dict[str, float]:
    """For each unique prompt, predict which model would be correct and pick the
    top-scoring model. Report routing accuracy (% of prompts where the chosen
    model answers correctly) and oracle accuracy (% of prompts where ANY model
    answers correctly, i.e. upper bound)."""
    model.eval()

    # Group records by prompt_id
    by_prompt: dict[int, list[dict]] = {}
    for r in records:
        if r["model_name"] not in model2id:
            continue
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    correct_routing = oracle_correct = 0
    n_prompts = 0
    with torch.no_grad():
        for prompt_id, recs in by_prompt.items():
            if prompt_id not in prompt2idx:
                continue
            pidx = torch.tensor([prompt2idx[prompt_id]] * len(recs), device=device)
            mids = torch.tensor([model2id[r["model_name"]] for r in recs], device=device)
            scores = model(mids, pidx)  # (num_models_for_prompt,)
            best_idx = scores.argmax().item()
            chosen = recs[best_idx]
            correct_routing += int(chosen["score"])
            oracle_correct   += int(any(r["score"] for r in recs))
            n_prompts        += 1

    return {
        "routing_accuracy": correct_routing / n_prompts if n_prompts else 0,
        "oracle_accuracy":  oracle_correct  / n_prompts if n_prompts else 0,
        "n_prompts":        n_prompts,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--embed-dim",    type=int,   default=EMBED_DIM,
                   help="Model embedding dimension (default: 1024, matches Qwen3-Embedding-0.6B)")
    p.add_argument("--dropout",      type=float, default=0.0,
                   help="Dropout rate applied after model_proj and interaction (default: 0.0)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--eval-every",   type=int,   default=5,
                   help="Run routing evaluation every N epochs (0 = off)")
    p.add_argument("--save-path",    type=str,
                   default=os.path.join(CKPT_DIR, "perf_router.pt"))
    p.add_argument("--perf-vectors", type=str, default=PERF_VECTORS,
                   help="Path to perf_vectors.json (from build_perf_vectors.py)")
    p.add_argument("--embed-cache",  type=str, default=EMBED_CACHE,
                   help="Path to cached Qwen3-Embedding-0.6B prompt embeddings")
    p.add_argument("--exclude-models", nargs="*", default=[],
                   help="Model names to exclude from training "
                        "(e.g. qwen3-30b-a3b qwen3-4b-thinking-2507)")
    p.add_argument("--loss", choices=["soft_ce", "bce"], default="soft_ce",
                   help="Training loss: soft_ce (contrastive per-question, default) "
                        "or bce (independent binary cross-entropy per model).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exclude_set = set(args.exclude_models)
    if exclude_set:
        print(f"Excluding models: {sorted(exclude_set)}")

    # --- Load accuracy vectors ---
    print("\nLoading accuracy vectors...")
    feat_names, vectors = load_perf_vectors(args.perf_vectors)
    # Drop excluded models from accuracy vectors
    for m in exclude_set:
        vectors.pop(m, None)
    print(f"  {len(feat_names)} features, {len(vectors)} models")

    model_ids_in_acc = set(vectors.keys())

    # Build accuracy matrix (rows = sorted model index, cols = feature dims)
    sorted_models = sorted(vectors.keys())
    accuracy_matrix = torch.tensor(
        [vectors[m] for m in sorted_models], dtype=torch.float32
    )
    # Z-score normalize per feature so accuracy vectors signal relative capability
    # (which models are unusually strong/weak at each category) rather than absolute
    # scores that always favour globally stronger models like qwen3-30b.
    acc_mean = accuracy_matrix.mean(dim=0, keepdim=True)
    acc_std  = accuracy_matrix.std(dim=0, keepdim=True).clamp(min=1e-6)
    accuracy_matrix = (accuracy_matrix - acc_mean) / acc_std
    print(f"  accuracy_matrix shape: {tuple(accuracy_matrix.shape)}  (z-score normalized)")

    # sorted_models[i] → index i in accuracy_matrix
    acc_model2id = {m: i for i, m in enumerate(sorted_models)}

    # --- Load training records ---
    print("\nLoading training records from embedllm-eval...")
    records = load_all_records(RESULTS_DIR, exclude_set)
    print(f"  {len(records)} total records")

    # Only keep models present in both records and accuracy vectors
    available = {r["model_name"] for r in records} & model_ids_in_acc
    model2id  = {m: acc_model2id[m] for m in available}
    prompt_ids   = sorted({r["prompt_id"] for r in records})
    prompt2idx   = {p: i for i, p in enumerate(prompt_ids)}
    prompt_texts = get_prompt_texts(records)

    print(f"  {len(model2id)} models with both records and accuracy vectors")
    print(f"  {len(prompt2idx)} unique prompts")

    # --- Prompt embeddings ---
    print("\nBuilding / loading prompt embeddings...")
    prompt_embeddings = build_or_load_prompt_embeddings(
        prompt2idx, prompt_texts, device,
        cache_path=args.embed_cache,
    )

    # --- Dataset ---
    # ContrastiveDataset groups all models per question; each batch item is one
    # question with soft labels over models.  batch_size here is in questions.
    train_ds = ContrastiveDataset(records, model2id, prompt2idx)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    print(f"\nTraining questions: {len(train_ds)}")

    # --- Model ---
    net = AccuracyAugmentedEmbedRouter(
        num_models=len(sorted_models),
        accuracy_matrix=accuracy_matrix,
        prompt_embeddings=prompt_embeddings,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Training loop ---
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_routing_acc = 0.0
    print(f"\nTraining for {args.epochs} epochs (loss={args.loss})...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(net, train_loader, optimizer, device, train=True, loss_type=args.loss)
        log = f"Epoch {epoch:3d}/{args.epochs}  loss={tr_loss:.4f}  acc={tr_acc:.4f}"

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            rmetrics = evaluate_routing(net, records, model2id, prompt2idx, device)
            routing_acc = rmetrics["routing_accuracy"]
            oracle_acc  = rmetrics["oracle_accuracy"]
            log += (f"  routing_acc={routing_acc:.4f}  oracle={oracle_acc:.4f}")
            if routing_acc > best_routing_acc:
                best_routing_acc = routing_acc
                ckpt = {
                    "epoch":            epoch,
                    "model_state_dict": net.state_dict(),
                    "model2id":         model2id,
                    "prompt2idx":       prompt2idx,
                    "feature_names":    feat_names,
                    "embed_dim":        args.embed_dim,
                    "num_models":       len(sorted_models),
                    "acc_dim":          accuracy_matrix.shape[1],
                    "acc_mean":         acc_mean.squeeze(0),
                    "acc_std":          acc_std.squeeze(0),
                    "routing_accuracy": routing_acc,
                    "oracle_accuracy":  oracle_acc,
                }
                best_path = args.save_path.replace(".pt", "_best.pt")
                torch.save(ckpt, best_path)
                log += "  [best saved]"

        print(log, flush=True)

    # Save final checkpoint
    torch.save({
        "epoch":            args.epochs,
        "model_state_dict": net.state_dict(),
        "model2id":         model2id,
        "prompt2idx":       prompt2idx,
        "feature_names":    feat_names,
        "embed_dim":        args.embed_dim,
        "num_models":       len(sorted_models),
        "acc_dim":          accuracy_matrix.shape[1],
    }, args.save_path)
    print(f"\nFinal model saved to: {args.save_path}")
    if best_routing_acc > 0:
        print(f"Best routing accuracy: {best_routing_acc:.4f}")


if __name__ == "__main__":
    main()
