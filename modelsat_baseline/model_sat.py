"""
Model-SAT architecture (arXiv:2502.17282, Figure 4).

Components:
  - E5-Large (ψ): encodes capability string c^m → (1024,) vector
  - MLP connector (W): Linear(1024 → 3072), aligns capability embedding
    into Phi-3-Mini's token embedding space
  - Phi-3-Mini (φ): receives [e_{c^m}, e_{x_i}, e_p] as input embeddings,
    outputs Pr("Yes") at the last token position as the routing score

The capability string is never tokenized by Phi-3-Mini — it enters only
as the projected embedding vector prepended to the sequence.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

E5_LARGE_DIM   = 1024
PHI3_EMBED_DIM = 3072

INQUIRY_PROMPT = "Predict whether the model can handle the above instruction by indicating 'Yes' or 'No'."


def build_input_text(question: str) -> str:
    """
    Assembles the text portion of the capability instruction (x_i + p).
    The capability string c^m is handled separately through E5-Large.
    """
    return f"{question}\n\n{INQUIRY_PROMPT}"


class ModelSAT(nn.Module):
    def __init__(
        self,
        e5_path: str,
        phi3_path: str,
        device: torch.device,
    ):
        super().__init__()
        self.device = device

        # --- Capability encoder ψ: E5-Large ---
        # SentenceTransformer handles pooling; outputs a (1024,) vector per string.
        self.encoder = SentenceTransformer(e5_path, device=str(device)).to(torch.bfloat16)
        self.encoder_dim = E5_LARGE_DIM

        # --- MLP connector W: Linear(1024 → 3072) ---
        # The only component trained in Stage 1.
        self.connector = nn.Linear(self.encoder_dim, PHI3_EMBED_DIM).to(device=device, dtype=torch.bfloat16)

        # --- Router LLM φ: Phi-3-Mini ---
        self.tokenizer = AutoTokenizer.from_pretrained(phi3_path)
        for attn_impl, label in [
            ("flash_attention_2", "Flash Attention 2"),
            ("sdpa",              "SDPA (Triton-backed)"),
            (None,                "default attention"),
        ]:
            try:
                kwargs = dict(torch_dtype=torch.bfloat16, device_map=str(device))
                if attn_impl:
                    kwargs["attn_implementation"] = attn_impl
                self.llm = AutoModelForCausalLM.from_pretrained(phi3_path, **kwargs)
                print(f"[INFO] Using {label}", flush=True)
                break
            except Exception:
                continue
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # torch.compile with max-autotune is intentionally omitted here.
        # It conflicts with gradient checkpointing + dynamic token-sequence lengths:
        # max-autotune records a separate CUDAGraph per unique input shape, and the
        # CUDAGraph checkpoint allocator then collides with gradient-checkpointing's
        # own memory save/restore, producing:
        #   RuntimeError: Expected curr_block->next == nullptr to be true
        # It also adds 5-10 minutes of autotuning warmup on the first two steps,
        # which is wasteful for a training run. Eager mode is fast enough here.

        # Pre-compute the token IDs for " Yes" and " No" (leading space for BPE).
        self.yes_id = self.tokenizer(" Yes", add_special_tokens=False).input_ids[-1]
        self.no_id  = self.tokenizer(" No",  add_special_tokens=False).input_ids[-1]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_capabilities(self, capability_strings: list[str]) -> torch.Tensor:
        """
        Encode a batch of capability strings with E5-Large.

        We deliberately do NOT use ``self.encoder.encode(...)`` because
        SentenceTransformer.encode wraps its forward pass in
        ``torch.no_grad()``. That blocks gradients from reaching ψ when the
        encoder is unfrozen in Stage 2, which would silently turn the
        paper's "fine-tune all parameters" recipe into a no-op for the
        encoder. Instead we drive the underlying transformer directly via
        tokenize() + forward(), which respects ``requires_grad`` on the
        encoder parameters (frozen in Stage 1, trainable in Stage 2).

        Returns: (B, 1024) float32 on self.device.
        """
        features = self.encoder.tokenize(capability_strings)
        features = {
            k: v.to(self.device)
            for k, v in features.items()
            if torch.is_tensor(v)
        }
        out = self.encoder(features)
        return out["sentence_embedding"].to(torch.bfloat16)  # (B, 1024)

    def tokenize_questions(self, input_texts: list[str]) -> dict:
        """
        Tokenize the text portion [question + inquiry prompt].
        Returns a dict with input_ids and attention_mask on self.device.
        """
        enc = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        capability_strings: list[str],
        input_texts: list[str],
    ) -> torch.Tensor:
        """
        Args:
            capability_strings: list of M unique capability strings.
            input_texts:        list of B strings = build_input_text(question).
                                B must be a multiple of M; each capability
                                string is broadcast to k = B // M consecutive
                                input texts.

                                During contrastive training all batch_k items
                                in a group share the same c^m, so callers pass
                                M = num_groups and B = num_groups * batch_k —
                                this lets us run E5-Large only M times instead
                                of B times.

                                For inference where every (model, instruction)
                                pair is unique, simply pass M = B = len(...).

        Returns:
            yes_logit: (B,) tensor — raw logit for "Yes" at the last token
                position for each pair. Pass through softmax([yes, no]) to
                get Pr("Yes") for inference; use directly with cross_entropy
                for contrastive training (avoids double-softmax).
        """
        M = len(capability_strings)
        B = len(input_texts)
        if B % M != 0:
            raise ValueError(
                f"len(input_texts)={B} must be a multiple of "
                f"len(capability_strings)={M}"
            )
        k = B // M

        # Step 1: encode the M unique capability strings with E5-Large → (M, 1024)
        cap_vecs = self.encode_capabilities(capability_strings).to(self.device)

        # Step 2: project through MLP connector → (M, 1, 3072), then broadcast to B
        cap_emb = self.connector(cap_vecs).unsqueeze(1)          # (M, 1, PHI3_EMBED_DIM)
        if k > 1:
            cap_emb = cap_emb.repeat_interleave(k, dim=0)        # (B, 1, PHI3_EMBED_DIM)
        cap_emb = cap_emb.to(self.llm.dtype)

        # Step 3: tokenize text portion and get Phi-3-Mini token embeddings → (B, T, 3072)
        enc = self.tokenize_questions(input_texts)
        text_emb = self.llm.model.embed_tokens(enc["input_ids"]) # (B, T, PHI3_EMBED_DIM)

        # Step 4: prepend capability embedding → (B, T+1, 3072)
        full_emb = torch.cat([cap_emb, text_emb], dim=1)

        # Extend attention mask by 1 for the prepended capability token
        cap_mask  = torch.ones(B, 1, dtype=enc["attention_mask"].dtype, device=self.device)
        full_mask = torch.cat([cap_mask, enc["attention_mask"]], dim=1)  # (B, T+1)

        # Step 5: forward through Phi-3-Mini transformer layers
        out = self.llm(
            inputs_embeds=full_emb,
            attention_mask=full_mask,
        )

        # Step 6: extract Yes/No logits at the last token position.
        # Return the raw "Yes" logit (not softmaxed) so the contrastive
        # cross-entropy loss in the trainer can apply log_softmax exactly once
        # across the group. Returning a softmaxed probability here would cause
        # a double-softmax bug: cross_entropy internally applies log_softmax,
        # so passing probabilities collapses all group scores toward uniform and
        # drives loss to ln(batch_k) regardless of model quality.
        # For inference, convert to a probability externally: softmax([yes, no]).
        last_logits = out.logits[:, -1, :]                       # (B, vocab_size)
        yn_logits   = last_logits[:, [self.yes_id, self.no_id]]  # (B, 2)
        return yn_logits[:, 0]                                    # (B,) raw Yes logit

    # ------------------------------------------------------------------
    # Parameter group helpers (used by the trainer for staged training)
    # ------------------------------------------------------------------

    def connector_parameters(self):
        return list(self.connector.parameters())

    def encoder_parameters(self):
        return list(self.encoder.parameters())

    def llm_parameters(self):
        return list(self.llm.parameters())

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False

    def unfreeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = True
