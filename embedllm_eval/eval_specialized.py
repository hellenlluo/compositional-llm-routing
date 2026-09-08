#!/usr/bin/env python3
"""
Evaluate the trained EmbedClassifier on specialized datasets.

For each question, the router picks the model with the highest predicted
probability of correctness, then checks the oracle binary matrix.

Datasets:
  - morehopqa  (JSON, field: "question")
  - musique    (JSONL, field: "question")
  - stepcot    (JSON, uses vqa_chain step 7 formatted as MCQ)

Oracle matrices: outputs/updated-matrices/{dataset}_full_binary.json
  Structure: {"rows": [qid, ...], "cols": [model_name, ...], "values": [[0/1, ...], ...]}
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_DIR, "data_preprocessing", "cleaned_trimmed_data")
MATRIX_DIR   = os.path.join(PROJECT_DIR, "outputs", "updated-matrices")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
MODELS_DIR   = "/n/fs/scratch/dl3533/models"
MPNET_PATH   = os.path.join(MODELS_DIR, "all-mpnet-base-v2")
CKPT_PATH    = os.path.join(MODELS_DIR, "embedclassifier.pt")

EXCLUDE_MODELS: set[str] = set()

MPNET_MAX_TOKENS    = 384
MPNET_CHARS_PER_TOK = 4
TAIL_CHARS          = MPNET_MAX_TOKENS * MPNET_CHARS_PER_TOK  # 1536


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tail_truncate(text: str) -> str:
    return text[-TAIL_CHARS:] if len(text) > TAIL_CHARS else text



def load_oracle(dataset: str) -> tuple[list[str], list[str], list[list[int]]]:
    path = os.path.join(MATRIX_DIR, f"{dataset}_full_binary.json")
    with open(path) as f:
        d = json.load(f)
    return d["rows"], d["cols"], d["values"]


def load_subtask_oracle(dataset: str) -> dict | None:
    """Return {qid: {cols, binary}} or None if file missing."""
    path = os.path.join(MATRIX_DIR, f"{dataset}_subtasks.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset loaders  →  list of (qid, question_text)
# ---------------------------------------------------------------------------

def load_morehopqa() -> list[tuple[str, str]]:
    with open(os.path.join(DATA_DIR, "morehopqa_cleaned.json")) as f:
        items = json.load(f)
    return [(item["id"], item["question"]) for item in items]


def load_musique() -> list[tuple[str, str]]:
    pairs = []
    with open(os.path.join(DATA_DIR, "musique_cleaned.jsonl")) as f:
        for line in f:
            item = json.loads(line)
            pairs.append((item["id"], item["question"]))
    return pairs


def load_stepcot() -> list[tuple[str, str]]:
    with open(os.path.join(DATA_DIR, "stepcot_cleaned.json")) as f:
        items = json.load(f)
    pairs = []
    for item in items:
        step7 = item["vqa_chain"][6]          # 0-indexed; always the last step
        q = step7["question"]
        opts = "\n".join(step7["options"])
        text = f"{q}\n{opts}"
        pairs.append((item["id"], text))
    return pairs


# ---------------------------------------------------------------------------
# Model (weights-only inference, no embedding table lookup needed)
# ---------------------------------------------------------------------------

class EmbedClassifier(nn.Module):
    def __init__(self, num_models: int, prompt_embeddings: torch.Tensor, embed_dim: int = 232):
        super().__init__()
        mpnet_dim = prompt_embeddings.shape[1]  # 768
        self.projection = nn.Linear(mpnet_dim, embed_dim)
        self.model_emb  = nn.Embedding(num_models, embed_dim)
        self.prompt_emb = nn.Embedding.from_pretrained(prompt_embeddings, freeze=True)
        self.classifier = nn.Linear(embed_dim, 1)

    def score_new_prompts(
        self,
        prompt_vecs: torch.Tensor,   # (N, 768)  MPNet embeddings
        model_ids:   torch.Tensor,   # (N,)      int indices
    ) -> torch.Tensor:               # (N,)      sigmoid probabilities
        q = self.projection(prompt_vecs)  # (N, embed_dim) learned
        p = self.model_emb(model_ids)     # (N, embed_dim)
        x = p * q                         # element-wise
        y = self.classifier(x).squeeze(1)
        return torch.sigmoid(y)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    dataset_name: str,
    pairs: list[tuple[str, str]],
    device: torch.device,
    encoder: SentenceTransformer,
    clf: EmbedClassifier,
    model2id: dict[str, int],
    out_dir: str,
) -> dict:
    oracle_rows, oracle_cols, oracle_values = load_oracle(dataset_name)
    row_index = {qid: i for i, qid in enumerate(oracle_rows)}

    # Only route over models that are both in the oracle and in the trained model
    shared_models = [m for m in oracle_cols if m in model2id]
    if not shared_models:
        print(f"[{dataset_name}] No overlap between oracle cols and trained models!")
        return {}

    oracle_col_idx = {m: oracle_cols.index(m) for m in shared_models}
    print(f"\n[{dataset_name}]")
    print(f"  Oracle models:  {oracle_cols}")
    print(f"  Routing models: {shared_models}")

    # Filter to questions present in the oracle
    eval_pairs = [(qid, txt) for qid, txt in pairs if qid in row_index]
    missing = len(pairs) - len(eval_pairs)
    if missing:
        print(f"  Warning: {missing}/{len(pairs)} questions not found in oracle, skipped.")

    if not eval_pairs:
        print("  No questions to evaluate.")
        return {}

    # Encode all questions in one batch
    texts = [tail_truncate(txt) for _, txt in eval_pairs]
    print(f"  Encoding {len(texts)} questions with MPNet...")
    embeddings = encoder.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    prompt_vecs = torch.tensor(embeddings, dtype=torch.float32, device=device)

    # Score every candidate model for every question
    model_id_tensors = {
        m: torch.full((len(eval_pairs),), model2id[m], dtype=torch.long, device=device)
        for m in shared_models
    }

    clf.eval()
    with torch.no_grad():
        scores = {
            m: clf.score_new_prompts(prompt_vecs, model_id_tensors[m]).cpu()
            for m in shared_models
        }

    score_matrix = torch.stack([scores[m] for m in shared_models], dim=1)  # (N, |shared|)
    chosen_idxs  = score_matrix.argmax(dim=1).tolist()

    # Build per-question predictions and count correct
    correct = 0
    predictions = []
    for i, (qid, txt) in enumerate(eval_pairs):
        oracle_row   = row_index[qid]
        chosen_model = shared_models[chosen_idxs[i]]
        oracle_c     = oracle_col_idx[chosen_model]
        is_correct   = int(oracle_values[oracle_row][oracle_c])
        correct     += is_correct

        model_scores = {m: round(float(scores[m][i]), 6) for m in shared_models}
        predictions.append({
            "question_id":    qid,
            "question":       txt,
            "valid_models":   shared_models,
            "chosen_model":   chosen_model,
            "correct":        bool(is_correct),
            "model_scores":   model_scores,
        })

    n = len(eval_pairs)
    accuracy = correct / n if n > 0 else 0.0

    per_model_acc = {}
    for m in shared_models:
        c    = oracle_col_idx[m]
        hits = sum(oracle_values[row_index[qid]][c] for qid, _ in eval_pairs)
        per_model_acc[m] = hits / n

    # Oracle accuracy: fraction of questions where at least one model is correct
    oracle_hits = sum(
        1 for qid, _ in eval_pairs
        if any(oracle_values[row_index[qid]][oracle_col_idx[m]] == 1 for m in shared_models)
    )
    oracle_acc = oracle_hits / n

    best_single_acc  = max(per_model_acc.values())
    best_single_name = max(per_model_acc, key=per_model_acc.get)
    avg_acc          = sum(per_model_acc.values()) / len(per_model_acc)

    routing_dist     = {m: 0 for m in shared_models}
    for idx in chosen_idxs:
        routing_dist[shared_models[idx]] += 1
    routing_dist_pct = {m: round(routing_dist[m] / n, 6) for m in shared_models}

    print(f"\n  Results ({n} questions):")
    print(f"    Oracle accuracy:       {oracle_acc:.4f}  ({oracle_hits}/{n})  [upper bound: ≥1 model correct]")
    print(f"    Router accuracy:       {accuracy:.4f}  ({correct}/{n})")
    print(f"    Best single model:     {best_single_acc:.4f}  ({best_single_name})")
    print(f"    Average single model:  {avg_acc:.4f}")
    print(f"\n  Per-model accuracy:")
    for m, acc in sorted(per_model_acc.items(), key=lambda x: -x[1]):
        print(f"    {m:<40s}  {acc:.4f}")
    print(f"\n  Routing distribution:")
    for m, cnt in sorted(routing_dist.items(), key=lambda x: -x[1]):
        print(f"    {m:<40s}  {cnt:6d}  ({routing_dist_pct[m]:.2%})")

    # Save predictions JSON
    pred_path = os.path.join(out_dir, f"{dataset_name}_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\n  Predictions saved to: {pred_path}")

    summary = {
        "dataset":                  dataset_name,
        "n_questions":              n,
        "oracle_accuracy":          round(oracle_acc, 6),
        "router_accuracy":          round(accuracy, 6),
        "router_correct":           correct,
        "best_single_model":        best_single_name,
        "best_single_acc":          round(best_single_acc, 6),
        "avg_single_acc":           round(avg_acc, 6),
        "per_model_accuracy":       {m: round(a, 6) for m, a in per_model_acc.items()},
        "routing_distribution":     routing_dist,
        "routing_distribution_pct": routing_dist_pct,
    }
    return summary, predictions


def evaluate_overall_subtask(
    dataset_name: str,
    predictions: list[dict],
    shared_models: list[str],
) -> dict | None:
    """Overall subtask accuracy: one model chosen per question (same as full-task),
    evaluated against the subtask-level oracle."""
    subtask_oracle = load_subtask_oracle(dataset_name)
    if subtask_oracle is None:
        print(f"  [subtask] skipped: {dataset_name}_subtasks.json not found")
        return None

    total_subs = 0
    router_subs_correct = 0
    oracle_subs_correct = 0
    q_router_all = 0
    q_oracle_all = 0
    n_qs = 0
    per_model_correct: dict[str, int] = {m: 0 for m in shared_models}

    for pred in predictions:
        qid = pred["question_id"]
        if qid not in subtask_oracle:
            continue
        entry       = subtask_oracle[qid]
        sub_cols    = entry["cols"]
        binary      = entry["binary"]    # (n_subs, n_models)
        n_subs      = len(binary)
        sub_col2idx = {m: j for j, m in enumerate(sub_cols)}

        n_qs      += 1
        total_subs += n_subs

        chosen = pred["chosen_model"]
        j = sub_col2idx.get(chosen)
        if j is not None:
            chosen_correct = sum(binary[s][j] for s in range(n_subs))
            router_subs_correct += chosen_correct
            if chosen_correct == n_subs:
                q_router_all += 1

        # Oracle: best model for this question's subtasks
        best_subs = max(
            (sum(binary[s][sub_col2idx[m]] for s in range(n_subs))
             for m in shared_models if m in sub_col2idx),
            default=0,
        )
        oracle_subs_correct += best_subs
        if best_subs == n_subs:
            q_oracle_all += 1

        for m in shared_models:
            jm = sub_col2idx.get(m)
            if jm is not None:
                per_model_correct[m] += sum(binary[s][jm] for s in range(n_subs))

    if total_subs == 0 or n_qs == 0:
        return None

    per_model_acc = {m: per_model_correct[m] / total_subs for m in shared_models}
    best_model    = max(per_model_acc, key=per_model_acc.get)

    print(f"\n  Subtask (overall routing, {n_qs} questions, {total_subs} subtasks):")
    print(f"    Oracle subtask accuracy:  {oracle_subs_correct / total_subs:.4f}")
    print(f"    Subtask router accuracy:  {router_subs_correct / total_subs:.4f}"
          f"  ({router_subs_correct}/{total_subs})")
    print(f"    Oracle question accuracy: {q_oracle_all / n_qs:.4f}")
    print(f"    Question router accuracy: {q_router_all / n_qs:.4f}"
          f"  ({q_router_all}/{n_qs})")
    print(f"    Best single model:        {per_model_acc[best_model]:.4f}  ({best_model})")

    return {
        "dataset":                  dataset_name,
        "n_questions":              n_qs,
        "n_subtasks":               total_subs,
        "oracle_subtask_accuracy":  round(oracle_subs_correct / total_subs, 6),
        "subtask_router_accuracy":  round(router_subs_correct / total_subs, 6),
        "subtask_router_correct":   router_subs_correct,
        "oracle_question_accuracy": round(q_oracle_all / n_qs, 6),
        "question_router_accuracy": round(q_router_all / n_qs, 6),
        "question_router_correct":  q_router_all,
        "best_single_model":        best_model,
        "best_single_subtask_acc":  round(per_model_acc[best_model], 6),
        "avg_single_subtask_acc":   round(sum(per_model_acc.values()) / len(per_model_acc), 6),
        "per_model_subtask_accuracy": {m: round(v, 6) for m, v in per_model_acc.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["morehopqa", "musique", "stepcot"],
                        choices=["morehopqa", "musique", "stepcot"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "eval_results", "qwen3"),
                        help="Directory to save prediction and summary JSONs")
    parser.add_argument("--ckpt", default=CKPT_PATH,
                        help="Path to trained checkpoint (default: embedclassifier.pt)")
    parser.add_argument("--subtask-out", default=None,
                        help="Path to save overall subtask summary JSON "
                             "(default: <out-dir>/overall_subtask_summary.json)")
    parser.add_argument("--qid-filter", default=None,
                        help="Path to JSON {dataset: [qid, ...]} restricting evaluated questions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load checkpoint (which also contains the saved model2id)
    ckpt = torch.load(args.ckpt, map_location=device)
    model2id: dict[str, int] = ckpt["model2id"]
    num_models = len(model2id)
    embed_dim = ckpt.get("embed_dim", 768)  # backward-compatible default
    print(f"Trained models ({num_models}): {sorted(model2id)}")
    print(f"embed_dim={embed_dim}")

    dummy_emb = torch.zeros(1, 768)
    clf = EmbedClassifier(num_models, dummy_emb, embed_dim=embed_dim).to(device)
    state = ckpt["model_state_dict"]
    # prompt_emb is frozen training data; not needed for inference on new prompts
    state = {k: v for k, v in state.items() if not k.startswith("prompt_emb")}
    clf.load_state_dict(state, strict=False)
    clf.eval()
    print(f"Loaded checkpoint from {args.ckpt}")

    # Load MPNet encoder
    mpnet_path = MPNET_PATH if os.path.isdir(MPNET_PATH) else "all-mpnet-base-v2"
    encoder = SentenceTransformer(mpnet_path, device=str(device))

    loaders = {
        "morehopqa": load_morehopqa,
        "musique":   load_musique,
        "stepcot":   load_stepcot,
    }

    qid_filter = None
    if args.qid_filter:
        with open(args.qid_filter) as _f:
            qid_filter = json.load(_f)
        print(f"  qid_filter loaded from {args.qid_filter}")

    all_summaries    = []
    subtask_summaries = []
    for ds in args.datasets:
        pairs = loaders[ds]()
        print(f"\nLoaded {len(pairs)} questions from {ds}")
        if qid_filter and ds in qid_filter:
            allowed = set(qid_filter[ds])
            pairs = [(qid, txt) for qid, txt in pairs if qid in allowed]
            print(f"  qid_filter: keeping {len(pairs)} questions")
        summary, predictions = evaluate(ds, pairs, device, encoder, clf, model2id, args.out_dir)
        if summary:
            all_summaries.append(summary)
            # Overall subtask uses the same routing decisions already computed
            shared_models = list(model2id.keys())
            sub_summary = evaluate_overall_subtask(ds, predictions, shared_models)
            if sub_summary:
                subtask_summaries.append(sub_summary)

    # Save combined accuracy summary
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAccuracy summary saved to: {summary_path}")

    if subtask_summaries:
        subtask_out = args.subtask_out or os.path.join(args.out_dir, "overall_subtask_summary.json")
        with open(subtask_out, "w") as f:
            json.dump(subtask_summaries, f, indent=2)
        print(f"Overall subtask summary saved to: {subtask_out}")


if __name__ == "__main__":
    main()
