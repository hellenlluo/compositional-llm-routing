#!/usr/bin/env python3
"""
Evaluate the trained EfficiencyRouter on the three test oracle matrices.

Supports both checkpoint modes:
  mode=frozen    -> load EfficiencyRouter, encode test questions with the
                    saved encoder name, look up scores via score_embeddings.
  mode=finetune  -> load EfficiencyRouterFinetune (encoder weights included
                    in the checkpoint), encode test questions on the fly
                    with the fine-tuned encoder.

Reported per dataset:
  Routing accuracy        — fraction of routed predictions that are correct
  Efficiency-weighted     — mean per_active_b of the routed model
  Oracle binary (UB)      — fraction of questions with ≥1 correct model
  Oracle eff (UB)         — mean max(per_active_b) per question
  Best single-model acc   — accuracy of the strongest fixed model
  Routing distribution    — counts per model

Two training-only models (mathstral-7b, llama-3.1-nemotron-nano-8b) are
masked out at inference; routing is restricted to the 8 oracle models.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from model import EfficiencyRouter, EfficiencyRouterFinetune

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = "/n/fs/scratch/dl3533/models"

# Fallback default if a checkpoint somehow lacks the encoder field.
DEFAULT_ENC_LOCAL = os.path.join(MODELS_DIR, "Qwen3-Embedding-0.6B")
DEFAULT_ENC_HUB   = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_ENC = DEFAULT_ENC_LOCAL if os.path.isdir(DEFAULT_ENC_LOCAL) else DEFAULT_ENC_HUB

MATRICES_DIR = Path(PROJECT_DIR) / "outputs" / "updated-matrices"
CLEANED_DIR  = Path(PROJECT_DIR) / "data_preprocessing" / "cleaned_trimmed_data"
CKPT_DIR     = os.path.join(SCRIPT_DIR, "checkpoints")
EMBED_CACHE_DIR = Path(MODELS_DIR)

DATASETS = ["morehopqa", "musique", "stepcot"]

TEST_MODELS = [
    "deepseek-r1-distill-llama-8b",
    "llama-3.1-8b-instruct",
    "llama-3.1-nemotron-nano-8b",
    "mathstral-7b",
    "medgemma-4b-it",
    "mistral-7b-instruct-v0.3",
    "phi-4-mini-instruct",
    "qwen1.5-0.5b-chat",
    "qwen3-30b-a3b",
    "qwen3-4b-thinking-2507",
]

PROMPT_MAX_CHARS = 1536


# ---------------------------------------------------------------------------
# Cleaned data loading
# ---------------------------------------------------------------------------

def load_cleaned(dataset: str) -> dict[str, str]:
    path = CLEANED_DIR / {
        "morehopqa": "morehopqa_cleaned.json",
        "musique":   "musique_cleaned.jsonl",
        "stepcot":   "stepcot_cleaned.json",
    }[dataset]

    if path.suffix == ".jsonl":
        with path.open() as f:
            entries = [json.loads(l) for l in f]
    else:
        entries = json.loads(path.read_text())

    result = {}
    for e in entries:
        qid = e["id"]
        if dataset == "stepcot":
            report  = e.get("report", "")
            chain   = e.get("vqa_chain", [])
            first_q = chain[0]["question"] if chain else ""
            text = f"{report} {first_q}".strip()
        else:
            text = e.get("question", "")
        result[qid] = text

    return result


def truncate(text: str) -> str:
    return text[-PROMPT_MAX_CHARS:] if len(text) > PROMPT_MAX_CHARS else text


# ---------------------------------------------------------------------------
# Frozen-encoder embedding cache (per dataset, per encoder)
# ---------------------------------------------------------------------------

def _eval_cache_name(encoder_path: str, dataset: str) -> str:
    base = os.path.basename(os.path.normpath(encoder_path)).replace("/", "_")
    return f"effrouter_eval__{base}__{dataset}.pt"


def encode_questions_cached(
    encoder_path: str,
    qid2text: dict[str, str],
    ordered_qids: list[str],
    dataset: str,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    cache_path = EMBED_CACHE_DIR / _eval_cache_name(encoder_path, dataset)
    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path.name}")
        return torch.load(str(cache_path), map_location="cpu")

    print(f"  Encoding {len(ordered_qids)} questions with {encoder_path}...")
    encoder = SentenceTransformer(encoder_path, device=str(device))
    texts = [truncate(qid2text.get(qid, "")) for qid in ordered_qids]
    embs = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    )
    # Free encoder weights from GPU before returning so subsequent datasets
    # don't OOM when a fresh SentenceTransformer is loaded for them.
    del encoder
    torch.cuda.empty_cache()

    tensor = torch.tensor(embs, dtype=torch.float32)
    torch.save(tensor, str(cache_path))
    print(f"  Saved to {cache_path.name}")
    return tensor


# ---------------------------------------------------------------------------
# Oracle matrix loading
# ---------------------------------------------------------------------------

def load_matrix(dataset: str, variant: str) -> tuple[list[str], list[str], np.ndarray]:
    path = MATRICES_DIR / f"{dataset}_{variant}.json"
    data = json.loads(path.read_text())
    qids   = data["rows"]
    models = data["cols"]
    values = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in data["values"]],
        dtype=np.float64,
    )
    return qids, models, values


def load_subtasks(dataset: str) -> dict:
    """Return {question_id: {cols, binary}} from subtasks matrix."""
    path = MATRICES_DIR / f"{dataset}_subtasks.json"
    if not path.exists():
        raise FileNotFoundError(f"Subtasks matrix not found: {path}")
    return json.loads(path.read_text())


def load_subtask_texts_cleaned(dataset: str) -> dict[str, list[str]]:
    """Return {qid: [subtask_text_0, subtask_text_1, ...]} from cleaned data."""
    if dataset == "morehopqa":
        items = json.loads((CLEANED_DIR / "morehopqa_cleaned.json").read_text())
        return {item["id"]: [s["question"] for s in item["question_decomposition"]]
                for item in items}
    elif dataset == "musique":
        result: dict[str, list[str]] = {}
        with open(CLEANED_DIR / "musique_cleaned.jsonl") as f:
            for line in f:
                item = json.loads(line)
                result[item["id"]] = [s["question"] for s in item["question_decomposition"]]
        return result
    elif dataset == "stepcot":
        items = json.loads((CLEANED_DIR / "stepcot_cleaned.json").read_text())
        result = {}
        for item in items:
            texts = []
            for step in item["vqa_chain"]:
                opts = "\n".join(step["options"])
                texts.append(f"{step['question']}\n{opts}")
            result[item["id"]] = texts
        return result
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_test_questions(
    net,                                    # EfficiencyRouter or EfficiencyRouterFinetune
    mode:           str,
    qid2text:       dict[str, str],
    ordered_qids:   list[str],
    dataset:        str,
    encoder_path:   str,
    device:         torch.device,
    batch_size:     int,
) -> np.ndarray:
    """Returns logits array of shape (N, num_train_models) on CPU."""
    net.eval()
    all_logits = []

    if mode == "frozen":
        # Cap encoding batch size: stepcot texts are long (report + question,
        # up to 1536 chars / ~500 tokens) and the Qwen3 KV tensors scale as
        # O(batch × seq²), so batch_size=256 blows up the 10 GB GPU.
        encode_bs = min(batch_size, 32)
        embs = encode_questions_cached(
            encoder_path, qid2text, ordered_qids, dataset, device, encode_bs,
        )
        with torch.no_grad():
            for start in range(0, len(embs), batch_size):
                batch = embs[start : start + batch_size].to(device)
                all_logits.append(net.score_embeddings(batch).cpu())

    elif mode == "finetune":
        texts = [truncate(qid2text.get(qid, "")) for qid in ordered_qids]
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                all_logits.append(net(batch_texts).cpu())

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return torch.cat(all_logits, dim=0).numpy()


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset:           str,
    net,
    mode:              str,
    train_model_names: list[str],
    encoder_path:      str,
    device:            torch.device,
    batch_size:        int,
    qid_filter:        dict | None = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")

    qids_bin, cols_bin, bin_vals = load_matrix(dataset, "full_binary")
    qids_eff, cols_eff, eff_vals = load_matrix(dataset, "full_per_active_b")
    assert qids_bin == qids_eff, "Row order mismatch"
    assert cols_bin == cols_eff, "Column order mismatch"
    # Allow the eval to work with a subset of models (e.g. no-qwen3 checkpoint).
    # Only route over models that are present in BOTH the oracle matrix AND the
    # trained checkpoint; the oracle upper-bound is also restricted to this set.
    model2train_idx = {m: i for i, m in enumerate(train_model_names)}
    shared_cols = [m for m in cols_bin if m in model2train_idx]
    if set(shared_cols) != set(cols_bin):
        excluded = sorted(set(cols_bin) - set(shared_cols))
        print(f"  Note: {len(excluded)} oracle model(s) not in checkpoint, excluded: {excluded}")

    oracle_cols  = shared_cols
    oracle_qids  = qids_bin

    # Restrict binary/eff matrices to shared columns
    shared_col_idxs = np.array([cols_bin.index(m) for m in shared_cols], dtype=np.int64)
    bin_vals = bin_vals[:, shared_col_idxs]
    eff_vals = eff_vals[:, shared_col_idxs]

    # Optionally restrict to a question-ID allow-list (e.g. R+C questions only)
    if qid_filter is not None and dataset in qid_filter:
        allowed = set(qid_filter[dataset])
        row_mask = [i for i, q in enumerate(oracle_qids) if q in allowed]
        oracle_qids = [oracle_qids[i] for i in row_mask]
        bin_vals    = bin_vals[row_mask, :]
        eff_vals    = eff_vals[row_mask, :]
        print(f"  qid_filter: keeping {len(oracle_qids)} questions")

    qid2text     = load_cleaned(dataset)

    logits_full = score_test_questions(
        net, mode, qid2text, oracle_qids, dataset,
        encoder_path, device, batch_size,
    )

    test_train_idxs = np.array([model2train_idx[m] for m in oracle_cols], dtype=np.int64)
    logits_test     = logits_full[:, test_train_idxs]      # (N, num_oracle_models)
    routed_local    = logits_test.argmax(axis=1)           # (N,) into oracle_cols

    n = len(oracle_qids)

    routed_binary = np.array([bin_vals[i, routed_local[i]] for i in range(n)])
    routed_eff    = np.array([eff_vals[i, routed_local[i]] for i in range(n)])

    routed_binary_nonan = np.nan_to_num(routed_binary, nan=0.0)
    routed_eff_nonan    = np.nan_to_num(routed_eff,    nan=0.0)

    # -----------------------------------------------------------------------
    # Derive active parameter counts from the eff matrix (eff[i,j] = 1/B_j
    # when model j is correct on question i, 0 otherwise).  For each model
    # column j, the non-zero values should all equal 1/B_j, so we take the
    # mean of the positive entries to recover B_j.
    # -----------------------------------------------------------------------
    K = len(oracle_cols)
    params_per_col = np.zeros(K)
    for j in range(K):
        col = eff_vals[:, j]
        valid = col[col > 0]
        if len(valid) > 0:
            params_per_col[j] = 1.0 / float(np.nanmean(valid))

    # Fallback: if a model has zero correct answers, assign it the max params
    # so it never looks cheaper than it is.
    max_params = params_per_col[params_per_col > 0].max() if np.any(params_per_col > 0) else 1.0
    params_per_col[params_per_col == 0] = max_params
    min_params = params_per_col.min()

    # -----------------------------------------------------------------------
    # BINARY MATRIX METRICS
    # routing_accuracy : fraction of routed questions where the model is correct
    # random_acc       : expected accuracy of a uniform random router
    # oracle_binary    : fraction answerable by at least one model (UB)
    # best_single_acc  : accuracy of the single best model (alternative UB)
    # -----------------------------------------------------------------------
    routing_acc        = routed_binary_nonan.mean()
    oracle_binary_mean = np.nanmean(np.nanmax(bin_vals, axis=1))
    best_model_acc     = np.nanmax(np.nanmean(bin_vals, axis=0))
    random_acc         = float(np.nanmean(bin_vals))   # avg per-model accuracy

    # -----------------------------------------------------------------------
    # EFFICIENCY MATRIX METRICS
    #
    # nECS (Normalized Efficiency-Correctness Score):
    #   nECS_i = correct_i × (min_params / params_routed_i)
    #          = routed_eff_nonan[i] × min_params
    #   Range [0, 1]: 1.0 = correct AND chose the smallest model.
    #   Penalizes both wrong answers (score 0) and overshooting model size.
    #
    # mean_active_params : average active-parameter count (B) of the routed
    #   model across questions (lower = more efficient).
    #
    # cost_savings_pct : relative savings vs always using the largest model.
    #   = (max_params - mean_params_routed) / max_params
    #
    # raw eff_score : mean(1/params if correct else 0) — unnormalised,
    #   kept for backward compatibility.
    # -----------------------------------------------------------------------
    eff_score_mean  = routed_eff_nonan.mean()
    oracle_eff_mean = np.nanmean(np.nanmax(eff_vals, axis=1))

    nECS        = routed_eff_nonan * min_params   # (N,), in [0, 1]
    nECS_mean   = float(nECS.mean())
    oracle_nECS = float(oracle_eff_mean * min_params)

    params_routed     = np.array([params_per_col[routed_local[i]] for i in range(n)])
    mean_params       = float(params_routed.mean())
    cost_savings_pct  = float((max_params - mean_params) / max_params * 100)

    # Oracle mean params: per question, the smallest model that is correct.
    oracle_params_per_q = np.array([
        params_per_col[np.nanargmin(
            np.where(bin_vals[i] > 0, params_per_col,
                     np.full(K, np.inf))
        )]
        if np.any(bin_vals[i] > 0) else max_params
        for i in range(n)
    ])
    oracle_mean_params   = float(np.nanmean(oracle_params_per_q))
    oracle_savings_pct   = float((max_params - oracle_mean_params) / max_params * 100)
    random_mean_params   = float(params_per_col.mean())
    random_savings_pct   = float((max_params - random_mean_params) / max_params * 100)

    model_counts = np.bincount(routed_local, minlength=K)
    model_pct    = model_counts / n

    # -----------------------------------------------------------------------
    # Print
    # -----------------------------------------------------------------------
    print(f"  N questions: {n}")

    print(f"\n  ── Binary matrix ──")
    print(f"  Routing accuracy:          {routing_acc:.4f}"
          f"  (oracle UB: {oracle_binary_mean:.4f} | best-single: {best_model_acc:.4f} | random: {random_acc:.4f})")

    print(f"\n  ── Efficiency matrix (active params weighted) ──")
    print(f"  nECS [0→1, ↑better]:       {nECS_mean:.4f}"
          f"  (oracle: {oracle_nECS:.4f})")
    print(f"  Mean active params [B, ↓]: {mean_params:.3f} B"
          f"  (oracle: {oracle_mean_params:.3f} B | random: {random_mean_params:.3f} B)")
    print(f"  Cost savings vs largest:   {cost_savings_pct:.1f}%"
          f"  (oracle: {oracle_savings_pct:.1f}% | random: {random_savings_pct:.1f}%)")
    print(f"  Raw eff score (1/B, ↑):    {eff_score_mean:.4f}"
          f"  (oracle: {oracle_eff_mean:.4f})")

    print(f"\n  Model routing distribution (active params shown):")
    col_params = {m: params_per_col[j] for j, m in enumerate(oracle_cols)}
    for m, pct, count in sorted(zip(oracle_cols, model_pct, model_counts), key=lambda x: -x[2]):
        print(f"    {m:<40} {col_params[m]:5.1f} B  {count:5d}  ({pct:.1%})")

    predictions = {oracle_qids[i]: oracle_cols[routed_local[i]] for i in range(n)}

    full_result = {
        "dataset":             dataset,
        "n":                   n,
        # binary
        "routing_accuracy":    float(routing_acc),
        "oracle_binary":       float(oracle_binary_mean),
        "best_single_model":   float(best_model_acc),
        "random_accuracy":     float(random_acc),
        # efficiency
        "nECS":                nECS_mean,
        "oracle_nECS":         oracle_nECS,
        "mean_active_params_B": mean_params,
        "oracle_mean_params_B": oracle_mean_params,
        "random_mean_params_B": random_mean_params,
        "cost_savings_pct":    cost_savings_pct,
        "oracle_savings_pct":  oracle_savings_pct,
        # raw (backward compat)
        "eff_score":           float(eff_score_mean),
        "oracle_eff":          float(oracle_eff_mean),
        "routing_distribution":     {m: int(c) for m, c in zip(oracle_cols, model_counts)},
        "routing_distribution_pct": {m: round(float(p), 6) for m, p in zip(oracle_cols, model_pct)},
    }

    # -----------------------------------------------------------------------
    # Subtask metrics (same routing decision, evaluated at subtask granularity)
    # -----------------------------------------------------------------------
    subtask_result = None
    try:
        subtasks_data = load_subtasks(dataset)
    except FileNotFoundError as e:
        print(f"\n  [subtask] skipped: {e}")
        return full_result, None, predictions

    subtask_router_correct  = 0
    subtask_oracle_correct  = 0
    total_subtasks          = 0
    question_router_correct = 0
    question_oracle_correct = 0
    per_model_sub_correct   = {m: 0 for m in oracle_cols}
    n_with_subs             = 0

    for i, qid in enumerate(oracle_qids):
        if qid not in subtasks_data:
            continue
        entry      = subtasks_data[qid]
        sub_cols   = entry["cols"]
        binary     = entry["binary"]   # (n_subtasks, n_models)
        n_subs     = len(binary)
        if n_subs == 0:
            continue
        n_with_subs    += 1
        total_subtasks += n_subs
        sub_col2idx     = {m: j for j, m in enumerate(sub_cols)}

        # Router-chosen model for this question
        chosen = oracle_cols[routed_local[i]]
        chosen_j = sub_col2idx.get(chosen)
        if chosen_j is not None:
            chosen_correct = sum(binary[s][chosen_j] for s in range(n_subs))
            subtask_router_correct  += chosen_correct
            question_router_correct += int(chosen_correct == n_subs)

        # Oracle: best model per question for subtasks
        best_subs = 0
        for m in oracle_cols:
            j = sub_col2idx.get(m)
            if j is None:
                continue
            m_correct = sum(binary[s][j] for s in range(n_subs))
            best_subs = max(best_subs, m_correct)
            per_model_sub_correct[m] += m_correct
        subtask_oracle_correct  += best_subs
        question_oracle_correct += int(best_subs == n_subs)

    if total_subtasks > 0:
        per_model_sub_acc = {m: per_model_sub_correct[m] / total_subtasks
                             for m in oracle_cols}
        best_model_sub    = max(per_model_sub_acc, key=per_model_sub_acc.get)

        print(f"\n  ── Subtask metrics (overall: one model per question) ──")
        print(f"  Subtask router accuracy:   {subtask_router_correct / total_subtasks:.4f}"
              f"  (oracle: {subtask_oracle_correct / total_subtasks:.4f})")
        print(f"  Question router accuracy:  {question_router_correct / n_with_subs:.4f}"
              f"  (oracle: {question_oracle_correct / n_with_subs:.4f})")

        subtask_result = {
            "dataset":                  dataset,
            "n_questions":              n_with_subs,
            "n_subtasks":               total_subtasks,
            "oracle_subtask_accuracy":  round(subtask_oracle_correct  / total_subtasks, 6),
            "subtask_router_accuracy":  round(subtask_router_correct  / total_subtasks, 6),
            "subtask_router_correct":   subtask_router_correct,
            "oracle_question_accuracy": round(question_oracle_correct / n_with_subs, 6),
            "question_router_accuracy": round(question_router_correct / n_with_subs, 6),
            "question_router_correct":  question_router_correct,
            "best_single_model":        best_model_sub,
            "best_single_subtask_acc":  round(per_model_sub_acc[best_model_sub], 6),
            "avg_single_subtask_acc":   round(sum(per_model_sub_acc.values()) / len(per_model_sub_acc), 6),
            "per_model_subtask_accuracy": {m: round(v, 6) for m, v in per_model_sub_acc.items()},
        }

    # -----------------------------------------------------------------------
    # Chained subtask: route EACH subtask independently
    # -----------------------------------------------------------------------
    chained_fields: dict = {}
    try:
        sub_texts = load_subtask_texts_cleaned(dataset)
    except Exception as e:
        print(f"\n  [chained] skipped (cannot load subtask texts): {e}")
        return full_result, subtask_result, predictions

    # Flatten to (qid, sidx, text) for questions in both oracle and text source
    flat_items: list[tuple[str, int, str]] = []
    for qid in oracle_qids:
        if qid not in subtasks_data or qid not in sub_texts:
            continue
        entry   = subtasks_data[qid]
        n_rows  = len(entry["rows"])
        texts_q = sub_texts[qid]
        for sidx in range(min(len(texts_q), n_rows)):
            flat_items.append((qid, sidx, texts_q[sidx]))

    if flat_items:
        print(f"\n  ── Chained subtask routing (independent model per subtask) ──")
        print(f"  Encoding {len(flat_items)} subtask texts...")
        flat_texts = [truncate(t) for _, _, t in flat_items]

        if mode == "frozen":
            sub_encoder = SentenceTransformer(encoder_path, device=str(device))
            sub_embs_np = sub_encoder.encode(
                flat_texts, batch_size=min(batch_size, 32),
                show_progress_bar=True, convert_to_numpy=True,
                normalize_embeddings=False,
            )
            del sub_encoder
            torch.cuda.empty_cache()
            sub_embs = torch.tensor(sub_embs_np, dtype=torch.float32)
            all_logits_sub = []
            net.eval()
            with torch.no_grad():
                for start in range(0, len(sub_embs), batch_size):
                    batch = sub_embs[start : start + batch_size].to(device)
                    all_logits_sub.append(net.score_embeddings(batch).cpu())
        else:  # finetune
            all_logits_sub = []
            net.eval()
            with torch.no_grad():
                for start in range(0, len(flat_texts), batch_size):
                    all_logits_sub.append(net(flat_texts[start : start + batch_size]).cpu())

        all_logits_sub   = torch.cat(all_logits_sub, dim=0)               # (N_subs, M_train)
        logits_sub_test  = all_logits_sub[:, test_train_idxs]             # (N_subs, K_oracle)
        chosen_sub_local = logits_sub_test.argmax(dim=1).numpy()          # index into oracle_cols

        chained_router_correct = 0
        chained_micro_oracle_correct = 0
        q_sub_correct:        dict[str, int] = {}
        q_sub_total:          dict[str, int] = {}
        q_oracle_sub_correct: dict[str, int] = {}

        for idx, (qid, sidx, _) in enumerate(flat_items):
            entry       = subtasks_data[qid]
            sub_cols    = entry["cols"]
            binary      = entry["binary"]
            sub_col2idx = {m: j for j, m in enumerate(sub_cols)}

            chosen_model = oracle_cols[chosen_sub_local[idx]]
            j       = sub_col2idx.get(chosen_model)
            correct = int(binary[sidx][j]) if j is not None else 0
            chained_router_correct += correct

            slot_feasible = int(any(
                binary[sidx][sub_col2idx[m]] for m in oracle_cols if m in sub_col2idx
            ))
            chained_micro_oracle_correct += slot_feasible

            q_sub_correct[qid]        = q_sub_correct.get(qid, 0) + correct
            q_sub_total[qid]          = q_sub_total.get(qid, 0) + 1
            q_oracle_sub_correct[qid] = q_oracle_sub_correct.get(qid, 0) + slot_feasible

        n_chained_qs          = len(q_sub_correct)
        chained_q_all         = sum(1 for qid in q_sub_correct
                                    if q_sub_correct[qid] == q_sub_total[qid])
        oracle_chained_q_all  = sum(1 for qid in q_sub_total
                                    if q_oracle_sub_correct.get(qid, 0) == q_sub_total[qid])
        total_ch_subs         = len(flat_items)

        chained_sub_acc       = chained_router_correct       / total_ch_subs
        chained_micro_oracle  = chained_micro_oracle_correct / total_ch_subs
        chained_q_acc         = chained_q_all        / n_chained_qs if n_chained_qs else 0.0
        oracle_chained_q_acc  = oracle_chained_q_all / n_chained_qs if n_chained_qs else 0.0

        print(f"  Chained subtask accuracy:  {chained_sub_acc:.4f}"
              f"  (micro oracle: {chained_micro_oracle:.4f})")
        print(f"  Chained question accuracy: {chained_q_acc:.4f}"
              f"  ({chained_q_all}/{n_chained_qs} questions all-correct,"
              f"  oracle: {oracle_chained_q_acc:.4f})")

        chained_fields = {
            "chained_n_questions":            n_chained_qs,
            "chained_n_subtasks":             total_ch_subs,
            "chained_accuracy":               round(chained_q_acc, 6),
            "chained_correct":                chained_q_all,
            "oracle_chained_accuracy":        round(oracle_chained_q_acc, 6),
            "chained_micro_subtask_accuracy": round(chained_sub_acc, 6),
            "chained_micro_subtask_correct":  chained_router_correct,
            "oracle_micro_subtask_feasibility": round(chained_micro_oracle, 6),
        }

    if chained_fields:
        subtask_result.update(chained_fields)
    return full_result, subtask_result, predictions


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def build_net_from_ckpt(ckpt: dict, device: torch.device):
    """Reconstruct the trained network from a checkpoint dict."""
    mode = ckpt.get("mode", "frozen")
    train_model_names = ckpt["model_names"]
    embed_dim         = ckpt["embed_dim"]
    encoder_path      = ckpt.get("encoder", DEFAULT_ENC)

    if mode == "frozen":
        # Need a dummy prompt_embeddings tensor with the right encoder dim.
        # We probe the encoder to get its dim, but never actually run it
        # (test questions are encoded separately and fed via score_embeddings).
        encoder_dim = SentenceTransformer(
            encoder_path, device="cpu",
        ).get_sentence_embedding_dimension()
        dummy_embs = torch.zeros(1, encoder_dim)
        net = EfficiencyRouter(
            num_models=len(train_model_names),
            prompt_embeddings=dummy_embs,
            embed_dim=embed_dim,
        ).to(device)
        # The frozen training prompt table is not used for eval on new questions.
        # Test questions are encoded separately and passed through score_embeddings().
        state = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if not k.startswith("prompt_emb.")
        }
        net.load_state_dict(state, strict=False)

    elif mode == "finetune":
        finetune_layers = ckpt.get("finetune_layers", 4)
        encoder = SentenceTransformer(encoder_path, device=str(device))
        net = EfficiencyRouterFinetune(
            num_models=len(train_model_names),
            encoder=encoder,
            embed_dim=embed_dim,
            finetune_layers=finetune_layers,
        ).to(device)
        net.load_state_dict(ckpt["model_state_dict"])

    else:
        raise ValueError(f"Unknown mode in checkpoint: {mode}")

    return net, mode, train_model_names, encoder_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        type=str, default=os.path.join(CKPT_DIR, "effrouter.pt"))
    p.add_argument("--datasets",    nargs="+", default=DATASETS,
                   choices=DATASETS, metavar="DS")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--out",         type=str, default=None,
                   help="Path to save full-task results JSON")
    p.add_argument("--subtask-out",     type=str, default=None,
                   help="Path to save combined subtask results JSON (overall + chained)")
    p.add_argument("--predictions-out", type=str, default=None,
                   help="Path to save per-question routing predictions JSON")
    p.add_argument("--qid-filter", type=str, default=None,
                   help="Path to JSON {dataset: [qid, ...]} restricting evaluated questions")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)

    net, mode, train_model_names, encoder_path = build_net_from_ckpt(ckpt, device)
    net.eval()

    print(f"  Mode:     {mode}")
    print(f"  Encoder:  {encoder_path}")
    print(f"  embed_dim={ckpt['embed_dim']}, offset={ckpt.get('offset', 1.0)}, "
          f"rank_weight={ckpt.get('rank_weight', 0.0)}")
    print(f"  Training models ({len(train_model_names)}): {train_model_names}")

    qid_filter = None
    if args.qid_filter:
        with open(args.qid_filter) as _f:
            qid_filter = json.load(_f)
        print(f"  qid_filter loaded from {args.qid_filter}")

    all_results     = []
    subtask_results = []
    all_predictions = {}
    for ds in args.datasets:
        full_r, sub_r, preds = evaluate_dataset(
            ds, net, mode, train_model_names, encoder_path, device, args.batch_size,
            qid_filter=qid_filter,
        )
        all_results.append(full_r)
        if sub_r is not None:
            subtask_results.append(sub_r)
        all_predictions[ds] = preds

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Summary across all evaluated datasets (micro-average)")
        print(f"{'='*60}")
        total_n = sum(r["n"] for r in all_results)

        print(f"\n  ── Binary matrix ──")
        for key, label in [
            ("routing_accuracy",  "Routing accuracy          "),
            ("oracle_binary",     "Oracle binary (UB)        "),
            ("best_single_model", "Best single model         "),
            ("random_accuracy",   "Random baseline           "),
        ]:
            micro_avg = sum(r[key] * r["n"] for r in all_results) / total_n
            print(f"  {label}: {micro_avg:.4f}")

        print(f"\n  ── Efficiency matrix ──")
        for key, label, fmt in [
            ("nECS",                "nECS [0→1, ↑]             ", ".4f"),
            ("oracle_nECS",         "Oracle nECS               ", ".4f"),
            ("mean_active_params_B","Mean active params [B, ↓] ", ".3f"),
            ("oracle_mean_params_B","Oracle mean params [B]    ", ".3f"),
            ("random_mean_params_B","Random mean params [B]    ", ".3f"),
            ("cost_savings_pct",    "Cost savings vs largest % ", ".1f"),
            ("oracle_savings_pct",  "Oracle savings %          ", ".1f"),
            ("eff_score",           "Raw eff score (1/B)       ", ".4f"),
            ("oracle_eff",          "Oracle eff score (UB)     ", ".4f"),
        ]:
            micro_avg = sum(r[key] * r["n"] for r in all_results) / total_n
            print(f"  {label}: {micro_avg:{fmt}}")

        if subtask_results:
            print(f"\n  ── Subtask overall (micro-average) ──")
            total_subs = sum(r["n_subtasks"] for r in subtask_results)
            total_qs   = sum(r["n_questions"] for r in subtask_results)
            sub_router = sum(r["subtask_router_correct"]  for r in subtask_results) / total_subs
            sub_oracle = sum(r["oracle_subtask_accuracy"] * r["n_subtasks"]
                             for r in subtask_results) / total_subs
            q_router   = sum(r["question_router_correct"] for r in subtask_results) / total_qs
            q_oracle   = sum(r["oracle_question_accuracy"] * r["n_questions"]
                             for r in subtask_results) / total_qs
            print(f"  Subtask router accuracy:   {sub_router:.4f}  (oracle: {sub_oracle:.4f})")
            print(f"  Question router accuracy:  {q_router:.4f}  (oracle: {q_oracle:.4f})")

            chained_only = [r for r in subtask_results if "chained_n_questions" in r]
            if chained_only:
                print(f"\n  ── Subtask chained (micro-average) ──")
                total_ch_subs = sum(r["chained_n_subtasks"]  for r in chained_only)
                total_ch_qs   = sum(r["chained_n_questions"] for r in chained_only)
                ch_router = sum(r["chained_micro_subtask_correct"] for r in chained_only) / total_ch_subs
                ch_oracle = sum(r["oracle_chained_accuracy"] * r["chained_n_questions"]
                                for r in chained_only) / total_ch_qs
                ch_q      = sum(r["chained_correct"] for r in chained_only) / total_ch_qs
                print(f"  Chained accuracy (questions, all slots correct): {ch_q:.4f}  (oracle: {ch_oracle:.4f})")
                print(f"  Chained micro subtask accuracy: {ch_router:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull-task results saved to {args.out}")

    if args.subtask_out and subtask_results:
        os.makedirs(os.path.dirname(args.subtask_out), exist_ok=True)
        with open(args.subtask_out, "w") as f:
            json.dump(subtask_results, f, indent=2)
        print(f"Subtask results saved to {args.subtask_out}")

    if args.predictions_out:
        os.makedirs(os.path.dirname(args.predictions_out), exist_ok=True)
        with open(args.predictions_out, "w") as f:
            json.dump(all_predictions, f, indent=2)
        print(f"Predictions saved to {args.predictions_out}")


if __name__ == "__main__":
    main()
