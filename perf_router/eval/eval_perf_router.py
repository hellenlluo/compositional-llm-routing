#!/usr/bin/env python3
"""Evaluate the trained accuracy-augmented router on held-out test datasets.

Test datasets: musique, morehopqa, stepcot  (NOT used during training)

Produces output files compatible with embedllm-eval conventions (paths via flags):
  summary.json                     -- full-question routing metrics
  subtask_summary.json             -- subtask routing with ONE model per question
  chained_subtask_summary.json      -- independent route per subtask (skipped if unavailable)
  predictions.json                 -- per-dataset routed model per question

Chained scoring can be VRAM-heavy; use --chained-router-batch-size if you OOM.

Usage:
    python perf_router/eval/eval_perf_router.py
    python perf_router/eval/eval_perf_router.py --ckpt /path/to/checkpoint.pt
    python perf_router/eval/eval_perf_router.py --datasets musique morehopqa
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR    = os.path.dirname(SCRIPT_DIR)           # perf_router/
PROJECT_DIR = os.path.dirname(EVAL_DIR)             # project root

MATRICES_DIR  = os.path.join(PROJECT_DIR, "outputs", "updated-matrices")
RUNS_DIR      = os.path.join(PROJECT_DIR, "outputs", "runs_4k")
MODELS_DIR    = "/n/fs/scratch/dl3533/models"
QWEN_LOCAL    = os.path.join(MODELS_DIR, "Qwen3-Embedding-0.6B")
QWEN_HUB      = "Qwen/Qwen3-Embedding-0.6B"
QWEN_PATH     = QWEN_LOCAL if os.path.isdir(QWEN_LOCAL) else QWEN_HUB
DEFAULT_CKPT  = os.path.join(MODELS_DIR, "perf_router", "perf_router_best.pt")
RESULTS_DIR   = os.path.join(EVAL_DIR, "results", "qwen3")

DATASETS = ["morehopqa", "musique", "stepcot"]

QWEN_MAX_TOKENS    = 512
QWEN_CHARS_PER_TOK = 4
TAIL_CHARS         = QWEN_MAX_TOKENS * QWEN_CHARS_PER_TOK


def tail_truncate(text: str) -> str:
    return text[-TAIL_CHARS:] if len(text) > TAIL_CHARS else text


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_full_binary(dataset: str) -> tuple[list[str], list[str], list[list[int]]]:
    """Return (question_ids, model_ids, values) from full binary matrix."""
    path = os.path.join(MATRICES_DIR, f"{dataset}_full_binary.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Full binary matrix not found: {path}")
    with open(path) as f:
        mat = json.load(f)
    return mat["rows"], mat["cols"], mat["values"]


def load_subtasks(dataset: str) -> dict[str, dict]:
    """Return {question_id: {cols, binary}} from subtasks matrix."""
    path = os.path.join(MATRICES_DIR, f"{dataset}_subtasks.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Subtasks matrix not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_question_texts(dataset: str) -> dict[str, str]:
    """Return {question_id: question_text} from any model's full.jsonl."""
    dataset_dir = os.path.join(RUNS_DIR, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Runs dir not found: {dataset_dir}")
    for model_dir in sorted(os.listdir(dataset_dir)):
        jsonl_path = os.path.join(dataset_dir, model_dir, "full.jsonl")
        if not os.path.isfile(jsonl_path):
            continue
        texts: dict[str, str] = {}
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id", "")
                q   = rec.get("question", "")
                if qid and q:
                    texts[qid] = q
        if texts:
            return texts
    raise FileNotFoundError(f"No full.jsonl found in {dataset_dir}")


def load_subtask_texts(dataset: str) -> dict[str, list[str]]:
    """Return {question_id: [subtask_text_0, ...]} from any model's subtask.jsonl."""
    dataset_dir = os.path.join(RUNS_DIR, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Runs dir not found: {dataset_dir}")
    for model_dir in sorted(os.listdir(dataset_dir)):
        jsonl_path = os.path.join(dataset_dir, model_dir, "subtask.jsonl")
        if not os.path.isfile(jsonl_path):
            continue
        raw: dict[str, dict[int, str]] = {}
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec  = json.loads(line)
                qid  = rec.get("question_id", "")
                sidx = rec.get("subtask_idx", 0)
                q    = rec.get("question", "")
                if qid and q:
                    if qid not in raw:
                        raw[qid] = {}
                    raw[qid][sidx] = q
        if raw:
            return {qid: [t for _, t in sorted(subs.items())]
                    for qid, subs in raw.items()}
    raise FileNotFoundError(f"No subtask.jsonl found in {dataset_dir}")


# ---------------------------------------------------------------------------
# Router model (must match train_perf_router.py architecture exactly)
# ---------------------------------------------------------------------------

class AccuracyAugmentedEmbedRouter(nn.Module):
    def __init__(
        self,
        num_models: int,
        accuracy_matrix: torch.Tensor,
        embed_dim: int = 1024,
    ):
        super().__init__()
        acc_dim = accuracy_matrix.shape[1]
        self.register_buffer("accuracy_matrix", accuracy_matrix.float())
        self.model_emb  = nn.Embedding(num_models, embed_dim)
        self.model_proj = nn.Linear(embed_dim + acc_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def score_all(self, q_embs: torch.Tensor) -> torch.Tensor:
        """Return (N_questions, N_models) logit scores via vectorised form.

        classifier(p ⊙ q) = (p ⊙ w) · q + b
        """
        w = self.classifier.weight.squeeze(0)   # (embed_dim,)
        b = self.classifier.bias                # (1,)
        p_learned  = self.model_emb.weight
        p_combined = torch.cat([p_learned, self.accuracy_matrix], dim=-1)
        p = self.model_proj(p_combined)         # (M, embed_dim)
        p_w = p * w                             # (M, embed_dim)
        return q_embs @ p_w.T + b              # (N, M)


# ---------------------------------------------------------------------------
# Encode questions
# ---------------------------------------------------------------------------

def encode_questions(question_texts: list[str], device: torch.device) -> torch.Tensor:
    """Encode questions with Qwen3-Embedding-0.6B; return (N, 1024) on CPU."""
    encoder = SentenceTransformer(QWEN_PATH, device=str(device))
    truncated = [tail_truncate(t) for t in question_texts]
    embeddings = encoder.encode(
        truncated,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return torch.tensor(embeddings, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset: str,
    router: AccuracyAugmentedEmbedRouter,
    router_model_ids: list[str],
    device: torch.device,
    *,
    chained_sentence_batch_size: int = 16,
    chained_router_batch_size: int = 2048,
    qid_filter: dict | None = None,
) -> tuple[dict, dict | None, dict]:
    """Return (fulltask_result, subtask_result_with_chained_fields, predictions)."""
    print(f"\n--- {dataset} ---")

    # --- Load full binary matrix ---
    qids, matrix_models, values = load_full_binary(dataset)
    mat_model2col = {m: i for i, m in enumerate(matrix_models)}
    print(f"  {len(qids)} questions, {len(matrix_models)} models in matrix")

    # Models in both router and matrix (preserve router order)
    shared_models = [m for m in router_model_ids if m in mat_model2col]
    shared_rows   = [router_model_ids.index(m) for m in shared_models]
    print(f"  {len(shared_models)} models in both router and matrix")

    # --- Load question texts ---
    q_texts = load_question_texts(dataset)
    valid_qids = [qid for qid in qids if qid in q_texts]
    print(f"  {len(valid_qids)} / {len(qids)} questions have text")

    # Optionally restrict to a question-ID allow-list (e.g. R+C questions only)
    if qid_filter is not None and dataset in qid_filter:
        allowed = set(qid_filter[dataset])
        valid_qids = [q for q in valid_qids if q in allowed]
        print(f"  qid_filter: keeping {len(valid_qids)} questions")

    # --- Encode questions ---
    print("  Encoding questions with Qwen3-Embedding-0.6B...")
    q_embs = encode_questions([q_texts[qid] for qid in valid_qids], device).to(device)

    # --- Score all (question, model) pairs ---
    router.eval()
    with torch.no_grad():
        scores_all = router.score_all(q_embs)       # (N, num_router_models)
        scores     = scores_all[:, shared_rows]     # (N, n_shared)

    best_idxs     = scores.argmax(dim=1).cpu().tolist()
    chosen_models = [shared_models[i] for i in best_idxs]
    predictions   = {valid_qids[i]: chosen_models[i] for i in range(len(valid_qids))}

    # Free GPU memory before subtask oracle / chained phases (scores can be huge on GPU).
    del q_embs, scores_all, scores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    qids_index = {qid: idx for idx, qid in enumerate(qids)}

    # ---- Full-task metrics ------------------------------------------------
    routing_correct = oracle_correct = 0
    per_model_correct = {m: 0 for m in shared_models}
    n = len(valid_qids)

    for i, qid in enumerate(valid_qids):
        row_vals = values[qids_index[qid]]
        routing_correct += row_vals[mat_model2col[chosen_models[i]]]
        oracle_correct  += int(any(row_vals[mat_model2col[m]] for m in shared_models))
        for m in shared_models:
            per_model_correct[m] += row_vals[mat_model2col[m]]

    per_model_acc  = {m: per_model_correct[m] / n for m in shared_models}
    best_model     = max(per_model_acc, key=per_model_acc.get)
    avg_single_acc = sum(per_model_acc.values()) / len(per_model_acc)

    model_choice_counts = {}
    for m in chosen_models:
        model_choice_counts[m] = model_choice_counts.get(m, 0) + 1
    model_choice_pct = {m: round(model_choice_counts.get(m, 0) / n, 6)
                        for m in shared_models}

    full_result = {
        "dataset":           dataset,
        "n_questions":       n,
        "oracle_accuracy":   round(oracle_correct  / n, 6),
        "router_accuracy":   round(routing_correct / n, 6),
        "router_correct":    routing_correct,
        "best_single_model": best_model,
        "best_single_acc":   round(per_model_acc[best_model], 6),
        "avg_single_acc":    round(avg_single_acc, 6),
        "per_model_accuracy":    {m: round(v, 6) for m, v in per_model_acc.items()},
        "routing_distribution":  model_choice_counts,
        "routing_distribution_pct": model_choice_pct,
    }

    print(f"  [fulltask] router={full_result['router_accuracy']:.4f}  "
          f"oracle={full_result['oracle_accuracy']:.4f}  "
          f"best_single={full_result['best_single_acc']:.4f} ({best_model})")

    # ---- Subtask metrics (overall: one model per question) ---------------
    subtask_result = None
    try:
        subtasks_data = load_subtasks(dataset)
    except FileNotFoundError as e:
        print(f"  [subtask] skipped: {e}")
        return full_result, None, predictions

    subtask_router_correct  = 0
    subtask_oracle_correct  = 0
    total_subtasks          = 0
    question_router_correct = 0
    question_oracle_correct = 0
    per_model_sub_correct   = {m: 0 for m in shared_models}

    for i, qid in enumerate(valid_qids):
        if qid not in subtasks_data:
            continue
        entry      = subtasks_data[qid]
        sub_cols   = entry["cols"]
        binary     = entry["binary"]      # (n_subtasks, n_models)
        n_subs     = len(binary)
        total_subtasks += n_subs

        sub_col2idx = {m: j for j, m in enumerate(sub_cols)}
        chosen = chosen_models[i]

        chosen_j = sub_col2idx.get(chosen)
        if chosen_j is not None:
            chosen_subs_correct = sum(binary[s][chosen_j] for s in range(n_subs))
            subtask_router_correct += chosen_subs_correct
            question_router_correct += int(chosen_subs_correct == n_subs)

        best_subs = 0
        for m in shared_models:
            j = sub_col2idx.get(m)
            if j is None:
                continue
            m_correct = sum(binary[s][j] for s in range(n_subs))
            best_subs = max(best_subs, m_correct)
            per_model_sub_correct[m] += m_correct
        subtask_oracle_correct  += best_subs
        question_oracle_correct += int(best_subs == n_subs)

    n_with_subs       = sum(1 for qid in valid_qids if qid in subtasks_data)
    per_model_sub_acc = {m: per_model_sub_correct[m] / total_subtasks
                         for m in shared_models}
    best_model_sub    = max(per_model_sub_acc, key=per_model_sub_acc.get)

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

    print(f"  [subtask]  router={subtask_result['subtask_router_accuracy']:.4f}  "
          f"oracle={subtask_result['oracle_subtask_accuracy']:.4f}  "
          f"best_single={subtask_result['best_single_subtask_acc']:.4f} ({best_model_sub})")

    # ---- Chained subtask: route each subtask independently ---------------
    try:
        sub_texts = load_subtask_texts(dataset)
    except FileNotFoundError as e:
        print(f"  [chained] skipped: {e}")
        return full_result, subtask_result, predictions

    # Flatten to (qid, sidx, text) for questions present in both oracle and text source
    flat_items: list[tuple[str, int, str]] = []
    for qid in valid_qids:
        if qid not in subtasks_data or qid not in sub_texts:
            continue
        entry   = subtasks_data[qid]
        n_rows  = len(entry["rows"])
        texts_q = sub_texts[qid]
        for sidx in range(min(len(texts_q), n_rows)):
            flat_items.append((qid, sidx, texts_q[sidx]))

    if flat_items:
        print(f"  [chained] Encoding {len(flat_items)} subtask texts with Qwen3-Embedding-0.6B...")
        flat_texts = [tail_truncate(t) for _, _, t in flat_items]
        ch_encoder = SentenceTransformer(QWEN_PATH, device=str(device))
        ch_embs_np = ch_encoder.encode(
            flat_texts,
            batch_size=max(8, chained_sentence_batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        del ch_encoder
        torch.cuda.empty_cache()

        # Keep embeddings on CPU; score in chunks so we never materialize (N_subs × M_router) logits.
        ch_embs_cpu = torch.tensor(ch_embs_np, dtype=torch.float32)
        del ch_embs_np
        router.eval()
        ch_chosen_local: list[int] = []
        with torch.no_grad():
            for start in range(0, len(ch_embs_cpu), chained_router_batch_size):
                blob = ch_embs_cpu[start : start + chained_router_batch_size].to(device)
                ch_scores_all = router.score_all(blob)          # (chunk, num_router_models)
                ch_scores     = ch_scores_all[:, shared_rows]   # (chunk, n_shared)
                ch_chosen_local.extend(ch_scores.argmax(dim=1).cpu().tolist())
                del blob, ch_scores_all, ch_scores
        del ch_embs_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chained_router_correct   = 0
        chained_micro_oracle_correct = 0
        q_sub_correct:        dict[str, int] = {}
        q_sub_total:          dict[str, int] = {}
        q_oracle_sub_correct: dict[str, int] = {}

        for idx, (qid, sidx, _) in enumerate(flat_items):
            entry       = subtasks_data[qid]
            sub_cols    = entry["cols"]
            binary      = entry["binary"]
            sub_col2idx = {m: j for j, m in enumerate(sub_cols)}

            chosen_model = shared_models[ch_chosen_local[idx]]
            j       = sub_col2idx.get(chosen_model)
            correct = int(binary[sidx][j]) if j is not None else 0
            chained_router_correct += correct

            slot_feasible = int(any(
                binary[sidx][sub_col2idx[m]] for m in shared_models if m in sub_col2idx
            ))
            chained_micro_oracle_correct += slot_feasible

            q_sub_correct[qid]        = q_sub_correct.get(qid, 0) + correct
            q_sub_total[qid]          = q_sub_total.get(qid, 0) + 1
            q_oracle_sub_correct[qid] = q_oracle_sub_correct.get(qid, 0) + slot_feasible

        n_ch_qs              = len(q_sub_correct)
        ch_q_all             = sum(1 for qid in q_sub_correct
                                   if q_sub_correct[qid] == q_sub_total[qid])
        oracle_chained_q_all = sum(1 for qid in q_sub_total
                                   if q_oracle_sub_correct.get(qid, 0) == q_sub_total[qid])
        total_ch             = len(flat_items)

        ch_sub_acc           = chained_router_correct       / total_ch
        ch_micro_oracle_acc  = chained_micro_oracle_correct / total_ch
        ch_q_acc             = ch_q_all             / n_ch_qs if n_ch_qs else 0.0
        oracle_chained_q_acc = oracle_chained_q_all / n_ch_qs if n_ch_qs else 0.0

        print(f"  [chained]  accuracy (questions)={ch_q_acc:.4f}  "
              f"oracle_chained={oracle_chained_q_acc:.4f}  "
              f"micro_subtask={ch_sub_acc:.4f}  micro_oracle={ch_micro_oracle_acc:.4f}")

        chained_fields = {
            "chained_n_questions":              n_ch_qs,
            "chained_n_subtasks":               total_ch,
            "chained_accuracy":                 round(ch_q_acc, 6),
            "chained_correct":                  ch_q_all,
            "oracle_chained_accuracy":          round(oracle_chained_q_acc, 6),
            "chained_micro_subtask_accuracy":   round(ch_sub_acc, 6),
            "chained_micro_subtask_correct":    chained_router_correct,
            "oracle_micro_subtask_feasibility": round(ch_micro_oracle_acc, 6),
        }
        subtask_result.update(chained_fields)
    else:
        print(f"  [chained] skipped: flat_items empty (no overlap between "
              f"{dataset} valid_qids, subtasks oracle, subtask.jsonl texts)")

    return full_result, subtask_result, predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",     type=str, default=DEFAULT_CKPT,
                   help=f"Path to trained router checkpoint (default: {DEFAULT_CKPT})")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   help=f"Datasets to evaluate on (default: {DATASETS})")
    default_acc = os.path.join(EVAL_DIR, "results", "qwen3", "accuracy_vectors.json")
    p.add_argument(
        "--accuracy-vectors",
        type=str,
        default=default_acc,
        help=f"Per-model accuracy vectors JSON (default: {default_acc})",
    )
    p.add_argument(
        "--chained-sentence-batch-size",
        type=int,
        default=16,
        help="SentenceTransformer encode batch size for chained subtask texts "
             "(lower if encoder OOMs).",
    )
    p.add_argument(
        "--chained-router-batch-size",
        type=int,
        default=2048,
        help="Perf-router forward batch size for chained subtasks "
             "(lower if score_all OOMs).",
    )
    p.add_argument("--summary-out",  type=str,
                   default=os.path.join(RESULTS_DIR, "summary.json"))
    p.add_argument("--subtask-out",      type=str,
                   default=os.path.join(RESULTS_DIR, "subtask_summary.json"))
    p.add_argument("--predictions-out",  type=str,
                   default=os.path.join(RESULTS_DIR, "predictions.json"))
    p.add_argument("--qid-filter", type=str, default=None,
                   help="Path to JSON {dataset: [qid, ...]} restricting evaluated questions")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def ensure_parent(path: str) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt}\n"
            f"Train first with submit_train_perf_router.sh"
        )
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    model2id: dict[str, int] = ckpt["model2id"]
    feat_names: list[str]    = ckpt["feature_names"]
    embed_dim: int           = ckpt.get("embed_dim", 1024)
    num_models: int          = ckpt["num_models"]
    acc_dim: int             = ckpt["acc_dim"]

    router_model_ids = sorted(model2id, key=lambda m: model2id[m])
    print(f"Router: {len(router_model_ids)} models, {len(feat_names)} accuracy features")

    # Rebuild accuracy matrix from saved vectors
    acc_vec_path = os.path.abspath(args.accuracy_vectors)
    if not os.path.isfile(acc_vec_path):
        raise FileNotFoundError(
            f"accuracy vectors JSON not found: {acc_vec_path}\n"
            f"Train/build vectors first or pass --accuracy-vectors"
        )
    with open(acc_vec_path) as f:
        acc_data = json.load(f)
    all_vecs = acc_data["vectors"]
    accuracy_matrix = torch.tensor(
        [all_vecs[m] for m in router_model_ids], dtype=torch.float32
    )
    assert accuracy_matrix.shape == (num_models, acc_dim), (
        f"accuracy_matrix shape mismatch: got {tuple(accuracy_matrix.shape)}, "
        f"expected ({num_models}, {acc_dim})"
    )

    # Apply the same z-score normalization used during training
    if "acc_mean" in ckpt and "acc_std" in ckpt:
        acc_mean = ckpt["acc_mean"].to(device)
        acc_std  = ckpt["acc_std"].to(device)
        accuracy_matrix = (accuracy_matrix.to(device) - acc_mean) / acc_std
        accuracy_matrix = accuracy_matrix.cpu()
        print(f"  Applied z-score normalization to accuracy matrix (from checkpoint)")
    else:
        print(f"  WARNING: checkpoint has no acc_mean/acc_std; accuracy matrix NOT normalized")

    print(f"\nAccuracy vectors: {acc_vec_path}")

    router = AccuracyAugmentedEmbedRouter(
        num_models=num_models,
        accuracy_matrix=accuracy_matrix,
        embed_dim=embed_dim,
    ).to(device)
    # strict=False: checkpoint contains frozen prompt_emb not needed at eval time
    router.load_state_dict(ckpt["model_state_dict"], strict=False)
    router.eval()

    qid_filter = None
    if args.qid_filter:
        with open(args.qid_filter) as _f:
            qid_filter = json.load(_f)
        print(f"  qid_filter loaded from {args.qid_filter}")

    full_results    = []
    subtask_results = []
    all_predictions = {}

    for dataset in args.datasets:
        try:
            full_r, sub_r, preds = evaluate_dataset(
                dataset,
                router,
                router_model_ids,
                device,
                chained_sentence_batch_size=args.chained_sentence_batch_size,
                chained_router_batch_size=args.chained_router_batch_size,
                qid_filter=qid_filter,
            )
            full_results.append(full_r)
            if sub_r is not None:
                subtask_results.append(sub_r)
            all_predictions[dataset] = preds
        except FileNotFoundError as e:
            print(f"  [skip] {dataset}: {e}")

    # Summary table
    print("\n=== Full-task Summary ===")
    print(f"{'Dataset':<15} {'Router':>8} {'Oracle':>8} {'BestSingle':>12}")
    print("-" * 45)
    for r in full_results:
        print(f"{r['dataset']:<15} {r['router_accuracy']:>8.4f} "
              f"{r['oracle_accuracy']:>8.4f} {r['best_single_acc']:>12.4f}")

    if subtask_results:
        print("\n=== Subtask Summary ===")
        print(f"{'Dataset':<15} {'Router':>8} {'Oracle':>8} {'BestSingle':>12}")
        print("-" * 45)
        for r in subtask_results:
            print(f"{r['dataset']:<15} {r['subtask_router_accuracy']:>8.4f} "
                  f"{r['oracle_subtask_accuracy']:>8.4f} "
                  f"{r['best_single_subtask_acc']:>12.4f}")

    for out_path in (args.summary_out, args.subtask_out, args.predictions_out):
        ensure_parent(out_path)

    with open(args.summary_out, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull-task results → {args.summary_out}")

    if subtask_results:
        with open(args.subtask_out, "w") as f:
            json.dump(subtask_results, f, indent=2)
        print(f"Subtask results   → {args.subtask_out}")
        if not any("chained_n_questions" in r for r in subtask_results):
            print(
                "\nWARNING: Chained subtask metrics were not produced.\n"
                "Check log lines starting with [chained] (missing subtask.jsonl, empty "
                "flat_items, or an error/crash during chained encoding/scoring — often VRAM)."
            )

    if all_predictions:
        with open(args.predictions_out, "w") as f:
            json.dump(all_predictions, f, indent=2)
        print(f"Predictions       → {args.predictions_out}")


if __name__ == "__main__":
    main()
