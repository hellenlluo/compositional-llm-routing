#!/usr/bin/env python3
"""
Fine-tune the Model-SAT router on training_pairs.jsonl.

Training follows the two-stage recipe from arXiv:2502.17282:

  Stage 1 — train only the MLP connector (E5-Large + Phi-3-Mini frozen).
             Establishes the initial capability-to-instruction alignment.

  Stage 2 — unfreeze all parameters. Apply a higher LR to the encoder
             and connector, a lower LR to Phi-3-Mini.

Loss: in-batch contrastive cross-entropy (Homogeneous In-Batch Negative
Sampling). Each batch contains records from a single model. One positive
(score=1) and k-1 negatives (score=0) are sampled. The loss maximises
Pr("Yes") for the positive relative to all negatives in the batch.

Checkpoint saved to: modelsat_baseline/router_checkpoint/
"""

import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model_sat import ModelSAT, build_input_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = "/n/fs/scratch/dl3533/models"
E5_PATH      = os.path.join(MODELS_DIR, "e5-large-v2")
PHI3_PATH    = os.path.join(MODELS_DIR, "Phi-3-mini-128k-instruct")
PAIRS_FILE   = os.path.join(SCRIPT_DIR, "modelsat_data", "training_pairs_trimmed.jsonl")
CKPT_DIR     = os.path.join(SCRIPT_DIR, "router_checkpoint")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(
    path: str,
    lucky_guess_threshold: int = 1,
) -> dict[int, dict[str, list[dict]]]:
    """
    Load training_pairs.jsonl, demote single-model-correct positives, and
    group records by stage and then by model_name.

    Returns:
      {
        1: {model_name: [{"prompt", "capability_string", "score"}, ...]},  # in-domain
        2: {model_name: [{"prompt", "capability_string", "score"}, ...]},  # OOD
      }

    Circle test approximation (arXiv:2502.17282): for multiple-choice
    questions where only `lucky_guess_threshold` or fewer models answered
    correctly, the correct answer is likely a lucky guess rather than
    genuine capability. Those positives are demoted to score=0 to avoid
    training the router on unreliable signal.
    """
    # First pass: count how many models answered each prompt correctly
    all_records = []
    prompt_correct_count: dict[int, int] = defaultdict(int)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            all_records.append(rec)
            if rec["score"] == 1:
                prompt_correct_count[rec["prompt_id"]] += 1

    # Second pass: demote positives on prompts only a single model got right,
    # and bucket records by stage → model.
    n_demoted = 0
    by_stage: dict[int, dict[str, list[dict]]] = {1: defaultdict(list), 2: defaultdict(list)}
    for rec in all_records:
        score = rec["score"]
        if score == 1 and prompt_correct_count[rec["prompt_id"]] <= lucky_guess_threshold:
            score = 0
            n_demoted += 1
        stage = rec.get("stage", 2)  # default to OOD if older file lacks the field
        by_stage[stage][rec["model_name"]].append({
            "prompt":            rec["prompt"],
            "capability_string": rec["capability_string"],
            "score":             score,
        })

    print(f"  Circle test: demoted {n_demoted} positives (≤{lucky_guess_threshold} model(s) correct)")
    return {stage: dict(d) for stage, d in by_stage.items()}


class ContrastivePairDataset(Dataset):
    """
    For each item, returns one positive and (batch_k - 1) negatives
    from the same model, sampled at dataset construction time.
    Each __getitem__ returns a list of batch_k dicts with keys:
      capability_string, input_text, is_positive (0/1)

    Stronger models (higher pos_rate) are sampled more frequently
    during training via a WeightedRandomSampler, following the paper's
    guidance to "prioritize higher-ranked candidates in the training
    data by sampling with increased weight" (arXiv:2502.17282).
    The dataset itself is unmodified — weights are returned via
    sample_weights() and passed to WeightedRandomSampler in the loader.
    """
    def __init__(
        self,
        by_model: dict[str, list[dict]],
        batch_k:  int,
        seed:     int = 42,
    ):
        rng = random.Random(seed)
        self.samples      = []   # list of [(input_text, cap_str, is_positive), ...]
        self.model_labels = []   # which model each sample came from

        for model, records in by_model.items():
            positives = [r for r in records if r["score"] == 1]
            negatives = [r for r in records if r["score"] == 0]

            if not positives or len(negatives) < batch_k - 1:
                continue

            for pos in positives:
                negs = rng.sample(negatives, batch_k - 1)
                # Pre-cache input_text so __getitem__ is pure list lookup.
                group = [
                    {
                        "capability_string": r["capability_string"],
                        "input_text":        build_input_text(r["prompt"]),
                        "is_positive":       int(i == 0),
                    }
                    for i, r in enumerate([pos] + negs)
                ]
                self.samples.append(group)
                self.model_labels.append(model)

        # Shuffle together
        combined = list(zip(self.samples, self.model_labels))
        rng.shuffle(combined)
        self.samples, self.model_labels = zip(*combined)

    def sample_weights(self, by_model: dict[str, list[dict]]) -> list[float]:
        """
        Return a per-sample weight proportional to the model's accuracy.
        Passed to WeightedRandomSampler so stronger models are drawn more often.
        """
        pos_rates = {
            model: sum(r["score"] for r in records) / len(records)
            for model, records in by_model.items()
            if records
        }
        return [pos_rates[m] for m in self.model_labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Group is fully pre-built at construction time; pure list lookup.
        return self.samples[idx]


def collate_fn(batch_of_groups):
    """
    batch_of_groups: list of N groups, each group is a list of k dicts.
    All k items in a group share the same capability string c^m, so we
    emit only one cap per group (length N) and the flat list of N*k input
    texts. ModelSAT.forward broadcasts each cap to k consecutive items via
    repeat_interleave, so E5-Large runs once per group instead of once per
    item — an ~8x speedup on the encoder pass for batch_k=8.

    Groups are sorted by descending max input-text length so the longest
    sequences appear first.  Phi-3-Mini pads all items in a batch to the
    same length; front-loading the long sequences lets subsequent (shorter)
    mini-batches skip the padding computation entirely.

    Returns:
        group_caps:  list of N capability strings (one per group)
        input_texts: list of N*k input texts (group order, positive first)
        group_ids:   list of N*k group indices (kept for diagnostics)
        pos_indices: (N,) tensor of positive positions within each group
                     (always 0 — positives are at index 0 of each group)
    """
    # Sort groups by the length of their longest input text (desc) so that
    # batches of similar-length sequences are processed together, minimising
    # padding waste and eliminating the periodic 8s outlier steps.
    batch_of_groups = sorted(
        batch_of_groups,
        key=lambda g: max(len(item["input_text"]) for item in g),
        reverse=True,
    )

    group_caps, input_texts, group_ids = [], [], []
    for g_idx, group in enumerate(batch_of_groups):
        # All items in `group` share the same capability_string by construction.
        group_caps.append(group[0]["capability_string"])
        for item in group:
            input_texts.append(item["input_text"])
            group_ids.append(g_idx)
    pos_indices = torch.zeros(len(batch_of_groups), dtype=torch.long)
    return group_caps, input_texts, group_ids, pos_indices


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def contrastive_ce_loss(
    pr_yes:     torch.Tensor,
    group_ids:  list[int],
    pos_indices: torch.Tensor,
    batch_k:    int,
) -> torch.Tensor:
    """
    For each group of batch_k items, compute cross-entropy where the
    target is the positive item (index 0 within the group).

    pr_yes:      (N * batch_k,) routing scores
    group_ids:   which group each element belongs to
    pos_indices: (N,) — always 0 (positive is first in each group)
    """
    N = len(pos_indices)
    pr_yes_grouped = pr_yes.view(N, batch_k)           # (N, batch_k)
    # Cross-entropy: target class is 0 (the positive) for every group
    targets = torch.zeros(N, dtype=torch.long, device=pr_yes.device)
    loss = nn.functional.cross_entropy(pr_yes_grouped, targets)
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: "ModelSAT",
    ckpt_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    extra: dict | None = None,
) -> None:
    tmp = ckpt_path + ".tmp"
    payload = {
        "connector_state_dict": model.connector.state_dict(),
        "encoder_state_dict":   model.encoder.state_dict(),
        "llm_state_dict":       model.llm.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}", flush=True)


def run_stage(
    model:        "ModelSAT",
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    epochs:       int,
    stage:        int,
    batch_k:      int,
    device:       torch.device,
    ckpt_path:    str | None = None,
    resume_state: dict | None = None,
    ckpt_every_steps: int = 500,
    scheduler_t_max: int | None = None,
):
    """
    scheduler_t_max: total optimizer steps across *all* chained jobs.
    When chaining epochs across SLURM jobs, pass the sum of steps for
    every planned epoch so the cosine LR schedule decays over the full
    training run rather than reaching zero at the end of each single job.
    If None, defaults to len(loader) * epochs (single-job behaviour).
    """
    import time
    total_steps = len(loader) * epochs
    t_max = scheduler_t_max if scheduler_t_max is not None else total_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=0.0,
    )

    if resume_state is not None:
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            print("  Optimizer state restored from checkpoint.", flush=True)
        if "scheduler_state_dict" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            # The saved state dict carries the old T_max from the prior job.
            # Override it with the full planned T_max so the cosine curve
            # spans the complete training chain, not just the prior job.
            scheduler.T_max = t_max
            print(f"  Scheduler state restored (last_epoch={scheduler.last_epoch}, "
                  f"T_max={t_max}, lr={scheduler.get_last_lr()[0]:.2e}).", flush=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, n_correct, n_total = 0.0, 0, 0
        epoch_start = time.time()

        for step_idx, (cap_strings, input_texts, group_ids, pos_indices) in enumerate(loader):
            step_start = time.time()
            pos_indices = pos_indices.to(device)

            pr_yes = model(cap_strings, input_texts)  # (N * batch_k,)

            loss = contrastive_ce_loss(pr_yes, group_ids, pos_indices, batch_k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Accuracy: did the positive get the highest score in its group?
            N = len(pos_indices)
            pr_grouped = pr_yes.detach().view(N, batch_k)
            preds = pr_grouped.argmax(dim=1)   # should all be 0
            n_correct += (preds == 0).sum().item()
            n_total   += N
            total_loss += loss.item() * N

            steps_done = step_idx + 1
            # Print step timing for first 3 steps and every 50 steps
            if step_idx < 3 or steps_done % 50 == 0:
                step_time = time.time() - step_start
                steps_left = len(loader) * (epochs - epoch + 1) - steps_done
                eta_h = steps_left * step_time / 3600
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Stage {stage} | Epoch {epoch}/{epochs} | Step {steps_done}/{len(loader)} | "
                      f"loss={loss.item():.4f} | lr={cur_lr:.2e} | step={step_time:.1f}s | ETA={eta_h:.1f}h", flush=True)

            # Mid-epoch checkpoint so progress isn't lost if the job times out
            if ckpt_path and ckpt_every_steps > 0 and steps_done % ckpt_every_steps == 0:
                _save_checkpoint(model, ckpt_path, optimizer=optimizer, scheduler=scheduler,
                                 extra={"stage": stage, "epoch": epoch, "step": steps_done})
                print(f"  Mid-epoch checkpoint at step {steps_done}", flush=True)

        avg_loss = total_loss / max(n_total, 1)
        acc      = n_correct  / max(n_total, 1)
        epoch_time = time.time() - epoch_start
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  Stage {stage} | Epoch {epoch}/{epochs} | "
              f"loss={avg_loss:.4f} | acc={acc:.4f} | lr={cur_lr:.2e} | epoch_time={epoch_time/60:.1f}min", flush=True)

        if ckpt_path:
            _save_checkpoint(model, ckpt_path, optimizer=optimizer, scheduler=scheduler,
                             extra={"stage": stage, "epoch": epoch})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-epochs", type=int,   default=5)
    p.add_argument("--stage2-epochs", type=int,   default=5)
    p.add_argument("--batch-k",       type=int,   default=8,
                   help="positives + negatives per contrastive group")
    p.add_argument("--batch-groups",  type=int,   default=16,
                   help="number of contrastive groups per gradient step")
    p.add_argument("--lr-connector",  type=float, default=1e-3)
    p.add_argument("--lr-encoder",    type=float, default=1e-4)
    p.add_argument("--lr-llm",        type=float, default=1e-5)
    p.add_argument("--num-workers",           type=int,   default=2,
                   help="DataLoader worker processes for prefetching (0 = synchronous)")
    p.add_argument("--scheduler-t-max",       type=int,   default=None,
                   help="Override cosine LR schedule T_max (total optimizer steps across "
                        "all chained jobs). Set this when chaining epochs across multiple "
                        "SLURM submissions so LR decays smoothly over the full training run. "
                        "Example: epoch1 ran 7070 steps; 4 more epochs × 1768 steps = "
                        "14142 total → pass --scheduler-t-max 14142 for epochs 2-5.")
    p.add_argument("--seed",                  type=int,   default=42)
    p.add_argument("--lucky-guess-threshold", type=int,   default=1,
                   help="demote positives where ≤ this many models answered correctly")
    p.add_argument("--ckpt-name",             type=str,   default="model_sat",
                   help="checkpoint filename stem (saved as <ckpt-name>.pt)")
    p.add_argument("--ckpt-dir",              type=str,   default=None,
                   help="checkpoint directory (default: modelsat_baseline/router_checkpoint/)")
    p.add_argument("--resume-from",           type=str,   default=None,
                   help="path to a .pt checkpoint to warm-start from (e.g. stage 1 output)")
    return p.parse_args()


def build_loader(
    by_model: dict[str, list[dict]],
    batch_k: int,
    batch_groups: int,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, ContrastivePairDataset]:
    """Wrap a per-model record dict in the contrastive dataset + weighted loader."""
    dataset = ContrastivePairDataset(by_model, batch_k=batch_k, seed=seed)
    if len(dataset) == 0:
        return None, dataset
    weights = dataset.sample_weights(by_model)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_groups,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        # 'spawn' avoids inheriting the parent's CUDA state when forking
        # worker processes after the model has already been loaded onto GPU.
        multiprocessing_context=("spawn" if num_workers > 0 else None),
    )
    return loader, dataset


def _print_stage_summary(stage: int, by_model: dict[str, list[dict]]) -> None:
    print(f"\nStage {stage} data:", flush=True)
    if not by_model:
        print("  (empty)")
        return
    for m, recs in by_model.items():
        pos = sum(r["score"] for r in recs)
        print(f"  {m}: {len(recs)} pairs  ({pos} pos / {len(recs)-pos} neg)")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # --- Load data, split by stage ---
    print("Loading training pairs...", flush=True)
    by_stage = load_pairs(PAIRS_FILE, lucky_guess_threshold=args.lucky_guess_threshold)

    # Stage 1 sees only the in-domain (MMLU) pairs.
    # Stage 2 sees all pairs — both in-domain and out-of-domain — so the
    # router gets to consolidate the capability→instruction mapping over
    # the full training distribution once all params are unfrozen.
    by_model_s1 = by_stage.get(1, {})
    by_model_all: dict[str, list[dict]] = defaultdict(list)
    for stage_dict in by_stage.values():
        for m, recs in stage_dict.items():
            by_model_all[m].extend(recs)
    by_model_all = dict(by_model_all)

    _print_stage_summary(1, by_model_s1)
    _print_stage_summary(2, by_model_all)

    loader_s1, ds_s1 = build_loader(by_model_s1,  args.batch_k, args.batch_groups, args.seed, args.num_workers)
    loader_s2, ds_s2 = build_loader(by_model_all, args.batch_k, args.batch_groups, args.seed, args.num_workers)
    print(f"\n  Stage 1 contrastive groups: {len(ds_s1)}", flush=True)
    print(f"  Stage 2 contrastive groups: {len(ds_s2)}", flush=True)

    # --- Build model ---
    print("\nLoading ModelSAT...", flush=True)
    model = ModelSAT(E5_PATH, PHI3_PATH, device)

    resume_state: dict | None = None
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}", flush=True)
        ckpt = torch.load(args.resume_from, map_location=device)
        model.connector.load_state_dict(ckpt["connector_state_dict"])
        model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        # Strip torch.compile wrapper prefix if present before loading LLM weights
        llm_sd = ckpt["llm_state_dict"]
        if any(k.startswith("_orig_mod.") for k in llm_sd):
            llm_sd = {k.replace("_orig_mod.", ""): v for k, v in llm_sd.items()}
        model.llm.load_state_dict(llm_sd, strict=False)
        print("  Model weights loaded.", flush=True)
        # Carry optimizer + scheduler state into the first stage that actually runs
        resume_state = ckpt

    ckpt_dir = args.ckpt_dir if args.ckpt_dir else CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.ckpt_name}.pt")

    # ==================================================================
    # Stage 1: train only the MLP connector on the in-domain (MMLU) data
    # ==================================================================
    print("\n=== Stage 1: connector only (in-domain / MMLU) ===", flush=True)
    if loader_s1 is None or args.stage1_epochs <= 0:
        print("  Skipping Stage 1 (no data or epochs=0).", flush=True)
    else:
        model.freeze_encoder()
        model.freeze_llm()

        opt1 = torch.optim.AdamW(
            model.connector_parameters(),
            lr=args.lr_connector,
            fused=True,
        )
        run_stage(model, loader_s1, opt1, args.stage1_epochs,
                  stage=1, batch_k=args.batch_k, device=device,
                  ckpt_path=ckpt_path, resume_state=resume_state)
        resume_state = None  # consumed; don't reuse for stage 2

    # ==================================================================
    # Stage 2: unfreeze all on the full data (in-domain + out-of-domain) —
    #          higher LR for encoder+connector, lower for LLM
    # ==================================================================
    print("\n=== Stage 2: all parameters (all pairs: in-domain + OOD) ===", flush=True)
    if loader_s2 is None or args.stage2_epochs <= 0:
        print("  Skipping Stage 2 (no data or epochs=0).", flush=True)
    else:
        model.unfreeze_encoder()
        model.unfreeze_llm()

        opt2 = torch.optim.AdamW([
            {"params": model.connector_parameters(), "lr": args.lr_connector},
            {"params": model.encoder_parameters(),   "lr": args.lr_encoder},
            {"params": model.llm_parameters(),       "lr": args.lr_llm},
        ], fused=True)
        run_stage(model, loader_s2, opt2, args.stage2_epochs,
                  stage=2, batch_k=args.batch_k, device=device,
                  ckpt_path=ckpt_path, resume_state=resume_state,
                  scheduler_t_max=args.scheduler_t_max)

    print(f"\nFinal checkpoint at: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
