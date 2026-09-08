"""
EfficiencyRouter: bilinear model that jointly scores all Nm candidate models
for a given question.

Two architectures supported:

  EfficiencyRouter
      Frozen-encoder version. Uses cached MPNet/BGE/etc. sentence embeddings
      that are pre-computed once for the training prompts. Only proj +
      model_emb + bias are trained.

  EfficiencyRouterFinetune
      Joint-encoder version. Wraps a SentenceTransformer encoder. Optionally
      unfreezes the last N transformer layers so the encoder is trained end
      to end with the routing head. Tokenization happens on the fly.

Two losses supported:

  soft_ce_loss
      Standard soft cross-entropy. Matches the predicted softmax distribution
      to the efficiency-weighted target distribution.

  pairwise_margin_loss
      Hinge ranking loss. For each (correct, wrong) model pair, enforces
      score[correct] >= score[wrong] + margin. Different gradient shape than
      soft CE — saturates outside the margin instead of decaying logarithmically.

The two losses can be combined: total = soft_ce + lam * pairwise_margin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Helpers
# ===========================================================================

def _find_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """
    Walk a HF transformer model and return its list of transformer blocks.

    Handles three common storage layouts:
      - BERT/RoBERTa/MPNet:  model.encoder.layer       (ModuleList of BertLayer)
      - Decoder LLMs (Qwen, Llama, Mistral): model.model.layers
      - T5/BART encoder-only wrappers: model.layers
    """
    candidates = [
        ("encoder.layer",        lambda m: getattr(m.encoder, "layer", None) if hasattr(m, "encoder") else None),
        ("encoder.layers",       lambda m: getattr(m.encoder, "layers", None) if hasattr(m, "encoder") else None),
        ("model.layers",         lambda m: getattr(m.model, "layers", None) if hasattr(m, "model") else None),
        ("layers",               lambda m: getattr(m, "layers", None)),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            layers = None
        if layers is not None and len(layers) > 0:
            return layers

    raise RuntimeError(
        f"Could not locate transformer layers in {type(model).__name__}. "
        f"Add this backbone's layer path to _find_transformer_layers()."
    )


# ===========================================================================
# Frozen-encoder router (cached sentence embeddings)
# ===========================================================================

class EfficiencyRouter(nn.Module):
    def __init__(
        self,
        num_models: int,
        prompt_embeddings: torch.Tensor,
        embed_dim: int = 256,
    ):
        super().__init__()
        text_dim = prompt_embeddings.shape[1]

        self.prompt_emb = nn.Embedding.from_pretrained(prompt_embeddings, freeze=True)
        self.proj       = nn.Linear(text_dim, embed_dim, bias=False)
        self.model_emb  = nn.Embedding(num_models, embed_dim)
        self.model_bias = nn.Parameter(torch.zeros(num_models))

        nn.init.normal_(self.model_emb.weight, std=0.01)

    def _score_from_q(self, q: torch.Tensor) -> torch.Tensor:
        return q @ self.model_emb.weight.T + self.model_bias

    def forward(self, prompt_idxs: torch.Tensor) -> torch.Tensor:
        q = self.proj(self.prompt_emb(prompt_idxs))
        return self._score_from_q(q)

    def score_embeddings(self, mpnet_embs: torch.Tensor) -> torch.Tensor:
        q = self.proj(mpnet_embs)
        return self._score_from_q(q)


# ===========================================================================
# Joint-encoder router (encoder fine-tuned end-to-end with routing head)
# ===========================================================================

class EfficiencyRouterFinetune(nn.Module):
    """
    Wraps a SentenceTransformer (e.g. MPNet, BGE-large, Snowflake-arctic).
    Tokenization runs on the fly so the encoder can backprop.

    finetune_layers controls how many of the encoder's last transformer
    blocks are trainable. 0 keeps the encoder frozen but still runs it
    on the fly (slower than caching; only useful as a sanity check).
    A small N (2-4) is usually enough — full fine-tuning of a 100M-param
    encoder on 2848 training questions tends to overfit.
    """

    def __init__(
        self,
        num_models:      int,
        encoder,                              # SentenceTransformer instance
        embed_dim:       int = 256,
        finetune_layers: int = 4,
    ):
        super().__init__()
        self.encoder    = encoder
        text_dim        = encoder.get_sentence_embedding_dimension()

        self.proj       = nn.Linear(text_dim, embed_dim, bias=False)
        self.model_emb  = nn.Embedding(num_models, embed_dim)
        self.model_bias = nn.Parameter(torch.zeros(num_models))

        nn.init.normal_(self.model_emb.weight, std=0.01)

        # Freeze everything in the encoder, then unfreeze the last N
        # transformer blocks (and the pooling layer if present).
        for p in self.encoder.parameters():
            p.requires_grad = False

        if finetune_layers > 0:
            layers = _find_transformer_layers(self.encoder[0].auto_model)
            for layer in layers[-finetune_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
            # Sentence-transformers usually has a pooling module at index 1.
            if len(self.encoder) > 1:
                for p in self.encoder[1].parameters():
                    p.requires_grad = True

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenize + forward through the encoder, respecting requires_grad.
        Returns (B, text_dim) sentence embeddings.

        We deliberately avoid `encoder.encode(...)` — it wraps the forward
        in torch.no_grad(), which silently freezes fine-tuning gradients.
        """
        device = next(self.parameters()).device
        features = self.encoder.tokenize(texts)
        features = {
            k: v.to(device) for k, v in features.items() if torch.is_tensor(v)
        }
        out = self.encoder(features)
        return out["sentence_embedding"]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, texts: list[str]) -> torch.Tensor:
        q_raw = self.encode(texts)              # (B, text_dim)
        q     = self.proj(q_raw)                # (B, embed_dim)
        return q @ self.model_emb.weight.T + self.model_bias


# ===========================================================================
# Losses
# ===========================================================================

def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy: -E_t[log softmax(logits)]."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def pairwise_margin_loss(
    logits:       torch.Tensor,
    soft_targets: torch.Tensor,
    margin:       float = 1.0,
) -> torch.Tensor:
    """
    Pairwise hinge ranking loss that enforces the ordering induced by
    soft_targets. For every pair (i, j) within a question where
    soft_targets[i] > soft_targets[j], we require:

        logits[i] >= logits[j] + margin

    For our soft labels = normalized (offset + 1/params_active_b), this means:

      • correct > wrong         (target_correct > 0, target_wrong = 0)
      • smaller_correct > larger_correct
            both correct, but smaller model has higher target.
            Example: qwen1.5-0.5b (target=0.73) ranked above llama-8b (target=0.27).

    Pairs where target[i] == target[j] (e.g. two equally large correct models,
    or two wrong models) impose no constraint.

    Loss per question is the mean hinge over all valid (i, j) pairs;
    final loss is averaged across the batch.
    """
    diff        = logits.unsqueeze(2)        - logits.unsqueeze(1)        # (B, M, M)
    target_diff = soft_targets.unsqueeze(2)  - soft_targets.unsqueeze(1)  # (B, M, M)

    # Constrain only the strict ordering target[i] > target[j].
    pair_mask = (target_diff > 0).float()                                 # (B, M, M)

    hinge = torch.clamp(margin - diff, min=0.0) * pair_mask               # (B, M, M)

    n_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)
    per_q   = hinge.sum(dim=(1, 2)) / n_pairs
    return per_q.mean()
