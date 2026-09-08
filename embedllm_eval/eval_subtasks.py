#!/usr/bin/env python3
"""
Evaluate the trained EmbedClassifier on chained per-subtask routing.

For each subtask of each question, the router picks the model with the highest
predicted probability of correctness for that subtask's text (independent choices
per subtask), then checks the subtask oracle binary matrix.

This is the EmbedLLM "chained subtask" setting (parallel to efficiency-router /
experiment_evals chained outputs).

Metrics (legacy names retained for backward compatibility):
  chained_accuracy        — PRIMARY: fraction of questions where every subtask was
                       routed to a correct model (perfect chain)
  chained_micro_subtask_accuracy — diagnostic per-slot rate (routed-correct slots / all slots)
  oracle_chained_accuracy — fraction of questions where a perfect chain is possible
                       (every subtask has ≥1 correct model in the pool)

Datasets / subtask text sources:
  morehopqa  question_decomposition[i]["question"]
  musique    question_decomposition[i]["question"]
  stepcot    vqa_chain[i] formatted as MCQ  (same as full-question eval step 7,
             but applied to every step)

Oracle subtask matrices:
  outputs/updated-matrices/{dataset}_subtasks.json
  Structure: { qid: {"rows": [0,1,...], "cols": [...], "binary": [[0/1,...], ...]} }
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
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data_preprocessing", "cleaned_trimmed_data")
MATRIX_DIR  = os.path.join(PROJECT_DIR, "outputs", "updated-matrices")
MODELS_DIR  = "/n/fs/scratch/dl3533/models"
MPNET_PATH  = os.path.join(MODELS_DIR, "all-mpnet-base-v2")
CKPT_PATH   = os.path.join(MODELS_DIR, "embedclassifier.pt")

MPNET_MAX_TOKENS    = 384
MPNET_CHARS_PER_TOK = 4
TAIL_CHARS          = MPNET_MAX_TOKENS * MPNET_CHARS_PER_TOK  # 1536


def tail_truncate(text: str) -> str:
    return text[-TAIL_CHARS:] if len(text) > TAIL_CHARS else text


# ---------------------------------------------------------------------------
# Subtask text loaders  →  {qid: [subtask_text, ...]}
# ---------------------------------------------------------------------------

def load_morehopqa_subtasks() -> dict[str, list[str]]:
    with open(os.path.join(DATA_DIR, "morehopqa_cleaned.json")) as f:
        items = json.load(f)
    return {
        item["id"]: [s["question"] for s in item["question_decomposition"]]
        for item in items
    }


def load_musique_subtasks() -> dict[str, list[str]]:
    result = {}
    with open(os.path.join(DATA_DIR, "musique_cleaned.jsonl")) as f:
        for line in f:
            item = json.loads(line)
            result[item["id"]] = [s["question"] for s in item["question_decomposition"]]
    return result


def load_stepcot_subtasks() -> dict[str, list[str]]:
    with open(os.path.join(DATA_DIR, "stepcot_cleaned.json")) as f:
        items = json.load(f)
    result = {}
    for item in items:
        texts = []
        for step in item["vqa_chain"]:
            opts = "\n".join(step["options"])
            texts.append(f"{step['question']}\n{opts}")
        result[item["id"]] = texts
    return result


# ---------------------------------------------------------------------------
# Model
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
        prompt_vecs: torch.Tensor,  # (N, 768)
        model_ids:   torch.Tensor,  # (N,)
    ) -> torch.Tensor:              # (N,)
        q = self.projection(prompt_vecs)  # (N, embed_dim) learned
        p = self.model_emb(model_ids)     # (N, embed_dim)
        x = p * q
        return torch.sigmoid(self.classifier(x).squeeze(1))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_subtasks(
    dataset_name: str,
    subtask_texts: dict[str, list[str]],
    device: torch.device,
    encoder: SentenceTransformer,
    clf: EmbedClassifier,
    model2id: dict[str, int],
    out_dir: str,
) -> dict:
    oracle_path = os.path.join(MATRIX_DIR, f"{dataset_name}_subtasks.json")
    with open(oracle_path) as f:
        oracle: dict = json.load(f)

    # Questions present in both the subtask text source and the oracle
    common_qids = [qid for qid in subtask_texts if qid in oracle]
    print(f"\n[{dataset_name}]")
    print(f"  Questions in subtask data: {len(subtask_texts)}")
    print(f"  Questions in oracle:       {len(oracle)}")
    print(f"  Common:                    {len(common_qids)}")

    if not common_qids:
        print("  No questions to evaluate.")
        return {}

    # Use the model set from the first oracle entry that has cols
    sample_cols = oracle[common_qids[0]]["cols"]
    shared_models = [m for m in sample_cols if m in model2id]
    print(f"  Oracle models:  {sample_cols}")
    print(f"  Routing models: {shared_models}")

    # Flatten all subtasks into a single list for batch encoding
    # flat_items: list of (qid, subtask_idx, text)
    flat_items: list[tuple[str, int, str]] = []
    for qid in common_qids:
        texts = subtask_texts[qid]
        n_rows = len(oracle[qid]["rows"])
        for i in range(min(len(texts), n_rows)):
            flat_items.append((qid, i, texts[i]))

    print(f"  Total subtasks to embed: {len(flat_items)}")

    # Encode all subtask texts
    raw_texts = [tail_truncate(txt) for _, _, txt in flat_items]
    print(f"  Encoding {len(raw_texts)} subtask texts with MPNet...")
    embeddings = encoder.encode(
        raw_texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    prompt_vecs = torch.tensor(embeddings, dtype=torch.float32, device=device)

    # Score each model for every subtask
    model_id_tensors = {
        m: torch.full((len(flat_items),), model2id[m], dtype=torch.long, device=device)
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

    # Build per-question results
    # subtask_results[qid] = list of per-subtask dicts
    subtask_results: dict[str, list[dict]] = {qid: [] for qid in common_qids}
    route_pick_counts = {m: 0 for m in shared_models}

    for flat_i, (qid, sub_idx, txt) in enumerate(flat_items):
        entry = oracle[qid]
        col_names = entry["cols"]
        binary    = entry["binary"]

        chosen_model = shared_models[chosen_idxs[flat_i]]
        route_pick_counts[chosen_model] += 1
        chosen_col   = col_names.index(chosen_model)
        is_correct   = int(binary[sub_idx][chosen_col])

        model_scores = {m: round(float(scores[m][flat_i]), 6) for m in shared_models}
        subtask_results[qid].append({
            "subtask_idx":   sub_idx,
            "question":      txt,
            "chosen_model":  chosen_model,
            "correct":       bool(is_correct),
            "model_scores":  model_scores,
        })

    # Aggregate metrics
    total_subtasks    = 0
    correct_subtasks  = 0
    total_questions   = len(common_qids)
    correct_questions = 0  # all subtasks routed correctly

    oracle_subtask_correct   = 0   # ≥1 model correct for this subtask
    oracle_question_correct  = 0   # ≥1 model correct for ALL subtasks

    per_model_subtask_correct: dict[str, int] = {m: 0 for m in shared_models}

    predictions = []
    for qid in common_qids:
        entry   = oracle[qid]
        col_names = entry["cols"]
        binary    = entry["binary"]
        n_sub     = len(entry["rows"])

        subs = subtask_results[qid]

        # Oracle per-subtask: any model correct?
        q_oracle_all = True
        q_all_correct = True

        for sub in subs:
            si = sub["subtask_idx"]
            row = binary[si]

            total_subtasks += 1
            if sub["correct"]:
                correct_subtasks += 1
            else:
                q_all_correct = False

            # Oracle for this subtask
            shared_col_indices = [col_names.index(m) for m in shared_models]
            if any(row[c] for c in shared_col_indices):
                oracle_subtask_correct += 1
            else:
                q_oracle_all = False

            # Per-model single-model accuracy
            for m in shared_models:
                c = col_names.index(m)
                per_model_subtask_correct[m] += int(row[c])

        if q_all_correct:
            correct_questions += 1
        if q_oracle_all:
            oracle_question_correct += 1

        predictions.append({
            "question_id":  qid,
            "all_correct":  q_all_correct,
            "subtasks":     subs,
        })

    subtask_acc   = correct_subtasks  / total_subtasks   if total_subtasks  else 0.0
    question_acc  = correct_questions / total_questions  if total_questions else 0.0
    oracle_st_acc = oracle_subtask_correct  / total_subtasks   if total_subtasks  else 0.0
    oracle_q_acc  = oracle_question_correct / total_questions  if total_questions else 0.0

    per_model_acc = {
        m: per_model_subtask_correct[m] / total_subtasks for m in shared_models
    }
    best_model_acc  = max(per_model_acc.values())
    best_model_name = max(per_model_acc, key=per_model_acc.get)
    avg_model_acc   = sum(per_model_acc.values()) / len(per_model_acc)

    n_route_picks      = len(flat_items)
    routing_dist       = {m: route_pick_counts[m] for m in shared_models}
    routing_dist_pct   = {
        m: round(route_pick_counts[m] / n_route_picks, 6)
        for m in shared_models
    } if n_route_picks else {m: 0.0 for m in shared_models}

    print(f"\n  Results ({total_questions} questions, {total_subtasks} subtasks):")
    print(f"    Oracle subtask accuracy:   {oracle_st_acc:.4f}  [≥1 model correct per subtask]")
    print(f"    Chained accuracy:          {question_acc:.4f}  ({correct_questions}/{total_questions})  [all subtasks correct — PRIMARY]")
    print(f"    Oracle chained accuracy:   {oracle_q_acc:.4f}  [% questions with perfect chain possible]")
    print(f"    Chained micro subtask acc: {subtask_acc:.4f}  ({correct_subtasks}/{total_subtasks})  [per-slot diagnostic]")
    print(f"    Best single model:         {best_model_acc:.4f}  ({best_model_name})")
    print(f"    Average single model:      {avg_model_acc:.4f}")
    print(f"\n  Chained routing distribution (# subtask picks):")
    for m, cnt in sorted(routing_dist.items(), key=lambda x: -x[1]):
        print(f"    {m:<40s}  {cnt:6d}  ({routing_dist_pct[m]:.2%})")

    print(f"\n  Per-model subtask accuracy:")
    for m, acc in sorted(per_model_acc.items(), key=lambda x: -x[1]):
        print(f"    {m:<40s}  {acc:.4f}")

    pred_path = os.path.join(out_dir, f"{dataset_name}_subtask_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\n  Predictions saved to: {pred_path}")

    return {
        "dataset":                  dataset_name,
        "n_questions":              total_questions,
        "n_subtasks":               total_subtasks,
        # Primary chained metrics (one independent model chosen per subtask)
        "chained_accuracy":               round(question_acc, 6),
        "chained_correct":                correct_questions,
        "oracle_chained_accuracy":        round(oracle_q_acc, 6),
        # Diagnostic micro-subtask metrics
        "chained_micro_subtask_accuracy": round(subtask_acc, 6),
        "chained_micro_subtask_correct":  correct_subtasks,
        "oracle_micro_subtask_feasibility": round(oracle_st_acc, 6),
        # Per-model breakdown
        "best_single_model":        best_model_name,
        "best_single_subtask_acc":  round(best_model_acc, 6),
        "avg_single_subtask_acc":   round(avg_model_acc, 6),
        "per_model_subtask_accuracy": {m: round(a, 6) for m, a in per_model_acc.items()},
        "routing_distribution":     routing_dist,
        "routing_distribution_pct": routing_dist_pct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--qid-filter", default=None,
                        help="Path to JSON {dataset: [qid, ...]} restricting evaluated questions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model2id: dict[str, int] = ckpt["model2id"]
    num_models = len(model2id)
    embed_dim = ckpt.get("embed_dim", 768)  # backward-compatible default
    print(f"Trained models ({num_models}): {sorted(model2id)}")
    print(f"embed_dim={embed_dim}")

    dummy_emb = torch.zeros(1, 768)
    clf = EmbedClassifier(num_models, dummy_emb, embed_dim=embed_dim).to(device)
    state = {k: v for k, v in ckpt["model_state_dict"].items()
             if not k.startswith("prompt_emb")}
    clf.load_state_dict(state, strict=False)
    clf.eval()
    print(f"Loaded checkpoint from {args.ckpt}")

    mpnet_path = MPNET_PATH if os.path.isdir(MPNET_PATH) else "all-mpnet-base-v2"
    encoder = SentenceTransformer(mpnet_path, device=str(device))

    loaders = {
        "morehopqa": load_morehopqa_subtasks,
        "musique":   load_musique_subtasks,
        "stepcot":   load_stepcot_subtasks,
    }

    qid_filter = None
    if args.qid_filter:
        with open(args.qid_filter) as _f:
            qid_filter = json.load(_f)
        print(f"  qid_filter loaded from {args.qid_filter}")

    all_summaries = []
    for ds in args.datasets:
        subtask_texts = loaders[ds]()
        print(f"\nLoaded subtasks for {len(subtask_texts)} questions from {ds}")
        if qid_filter and ds in qid_filter:
            allowed = set(qid_filter[ds])
            subtask_texts = {k: v for k, v in subtask_texts.items() if k in allowed}
            print(f"  qid_filter: keeping {len(subtask_texts)} questions")
        summary = evaluate_subtasks(ds, subtask_texts, device, encoder, clf, model2id, args.out_dir)
        if summary:
            all_summaries.append(summary)

    summary_path = os.path.join(args.out_dir, "subtask_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSubtask summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
