#!/usr/bin/env python3
"""
Evaluate a trained ModelSAT router on the held-out test sets
(morehopqa, musique, stepcot).

For each question the router scores all M candidate models with
Pr("Yes | capability_string, question") and picks the argmax.

Outputs (in --out-dir):
  summary.json          – per-dataset routing accuracy vs oracle
  predictions.json      – per-question model choice + correct flag
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR  = "/n/fs/scratch/dl3533/models"
E5_PATH     = os.path.join(MODELS_DIR, "e5-large-v2")
PHI3_PATH   = os.path.join(MODELS_DIR, "Phi-3-mini-128k-instruct")

MATRICES_DIR = os.path.join(PROJECT_DIR, "outputs", "updated-matrices")
CLEANED_DIR  = os.path.join(PROJECT_DIR, "data_preprocessing", "cleaned_trimmed_data")
RUNS_DIR     = os.path.join(PROJECT_DIR, "outputs", "runs_4k")
CAP_FILE     = os.path.join(SCRIPT_DIR, "modelsat_data", "capability_representations.json")
CKPT_DEFAULT = os.path.join(SCRIPT_DIR, "router_checkpoint_v2", "model_sat_stage2.pt")

DATASETS = ["morehopqa", "musique", "stepcot"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_capability_strings(cap_file: str, models: list[str]) -> dict[str, str]:
    with open(cap_file) as f:
        raw = json.load(f)
    return {m: raw[m]["capability_string"] for m in models}


def load_oracle_matrix(dataset: str, models: list[str]) -> tuple[list[str], torch.Tensor]:
    """Returns (question_ids, binary_matrix) where binary_matrix is (N, M) int8."""
    path = os.path.join(MATRICES_DIR, f"{dataset}_full_binary.json")
    with open(path) as f:
        d = json.load(f)
    all_cols = d["cols"]
    rows     = d["rows"]
    values   = d["values"]   # list of lists (N x len(all_cols))

    # Restrict to our model subset, in the order of `models`
    col_idx = [all_cols.index(m) for m in models]
    matrix  = torch.tensor(
        [[row[i] for i in col_idx] for row in values],
        dtype=torch.int8,
    )  # (N, M)
    return rows, matrix


def load_subtask_oracle(dataset: str, models: list[str]) -> dict:
    """Return {qid: {cols, binary, rows}} from the subtasks matrix, filtered to `models`."""
    path = os.path.join(MATRICES_DIR, f"{dataset}_subtasks.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Subtask oracle not found: {path}")
    raw = json.load(open(path))
    return raw  # dict keyed by question_id


def load_subtask_texts(dataset: str) -> dict[str, list[str]]:
    """Return {qid: [subtask_text, ...]} from cleaned data."""
    if dataset == "morehopqa":
        path = os.path.join(CLEANED_DIR, "morehopqa_cleaned.json")
        items = json.load(open(path))
        return {item["id"]: [s["question"] for s in item["question_decomposition"]]
                for item in items}
    elif dataset == "musique":
        result: dict[str, list[str]] = {}
        path = os.path.join(CLEANED_DIR, "musique_cleaned.jsonl")
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                result[item["id"]] = [s["question"] for s in item["question_decomposition"]]
        return result
    elif dataset == "stepcot":
        path = os.path.join(CLEANED_DIR, "stepcot_cleaned.json")
        items = json.load(open(path))
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


def load_questions(dataset: str, question_ids: list[str]) -> dict[str, str]:
    """
    Load question text for each question_id from the first available
    model's full.jsonl inside runs_4k/{dataset}/.
    """
    ds_dir = os.path.join(RUNS_DIR, dataset)
    # Pick first model directory that has full.jsonl
    jsonl_path = None
    for entry in sorted(os.listdir(ds_dir)):
        candidate = os.path.join(ds_dir, entry, "full.jsonl")
        if os.path.isfile(candidate):
            jsonl_path = candidate
            break
    if jsonl_path is None:
        raise FileNotFoundError(f"No full.jsonl found under {ds_dir}")

    qmap: dict[str, str] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qmap[rec["question_id"]] = rec["question"]

    # Keep only the question_ids we need (in order)
    out = {}
    for qid in question_ids:
        if qid in qmap:
            out[qid] = qmap[qid]
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def route_dataset(
    model,
    cap_strings: list[str],   # M capability strings, one per model
    questions: list[str],      # N question texts
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Returns (N, M) float tensor of Pr("Yes") for each (question, model) pair.
    Processes `batch_size` questions at a time, scoring all M models per batch.
    """
    M = len(cap_strings)
    N = len(questions)
    scores = torch.zeros(N, M)

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="  Scoring", unit="batch"):
            batch_qs = questions[start : start + batch_size]
            k = len(batch_qs)

            # Repeat each question M times so that:
            #   input_texts[i*M + j] = batch_qs[i] scored against cap_strings[j]
            # ModelSAT forward: cap_strings has length M, input_texts has length k*M
            # where cap_strings[j] is broadcast to batch_qs[i] for all i.
            # But ModelSAT's broadcast is over contiguous k-blocks per model,
            # so we need to tile differently:
            # capability_strings = [cap_m0]*k + [cap_m1]*k + ... — but the API
            # says: M unique caps, B = k*M texts where cap[j] maps to texts[j*k:(j+1)*k].
            # So we pass: cap_strings (M,), questions tiled as M blocks of k.
            tiled_qs = []
            for cap_idx in range(M):
                tiled_qs.extend(batch_qs)   # k questions for this model

            # Forward: returns (k*M,) raw Yes logits
            yes_logits = model(cap_strings, tiled_qs)  # (k*M,)

            # Convert to Pr("Yes") via softmax([yes, no])
            # We only have yes_logit; need no_logit too.
            # model.forward returns yes_logit only, but model_sat.py explains
            # inference should use softmax([yes, no]). We need both — call the
            # internal path that returns both via model(caps, texts):
            # Actually, model.forward returns yes_logit (the raw yes logit, not softmaxed).
            # For routing we just need argmax over models — softmax is monotone so
            # argmax(yes_logit) == argmax(Pr("Yes")), no need to convert.
            yes_logits = yes_logits.float().cpu()  # (k*M,)

            # Reshape: (M, k) → transpose → (k, M)
            block = yes_logits.view(M, k).T  # (k, M)
            scores[start : start + k] = block

    return scores  # (N, M)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset: str,
    model,
    models: list[str],
    cap_strings: list[str],
    batch_size: int,
    device: torch.device,
    run_chained: bool = False,
) -> tuple[dict, dict | None, list[dict]]:
    print(f"\n[{dataset}]")

    # Load oracle
    question_ids, oracle = load_oracle_matrix(dataset, models)
    print(f"  {len(question_ids)} questions, {len(models)} models")

    # Load question text
    qtext_map = load_questions(dataset, question_ids)
    # Keep only questions that have text (in case of any mismatch)
    valid_ids = [qid for qid in question_ids if qid in qtext_map]
    if len(valid_ids) < len(question_ids):
        print(f"  [warn] {len(question_ids) - len(valid_ids)} questions missing text; skipping them")
    id_to_idx = {qid: i for i, qid in enumerate(question_ids)}
    valid_idx = [id_to_idx[qid] for qid in valid_ids]
    oracle_valid = oracle[valid_idx]          # (N', M)
    questions   = [qtext_map[qid] for qid in valid_ids]
    N = len(valid_ids)
    M = len(models)

    # Score all (question, model) pairs
    scores = route_dataset(model, cap_strings, questions, batch_size, device)  # (N, M)

    # Pick best model per question
    routed = scores.argmax(dim=1)  # (N,)

    # Oracle: best possible model per question
    oracle_float = oracle_valid.float()
    any_correct = oracle_float.sum(dim=1) > 0      # (N,) questions with any correct model
    oracle_model = oracle_float.argmax(dim=1)       # (N,) best model (ties → first)

    # Routing accuracy: was routed model correct?
    routed_correct = oracle_valid[torch.arange(N), routed].float()  # (N,)
    routing_acc = routed_correct.mean().item()

    # Oracle accuracy: if we always pick the best model
    oracle_acc = oracle_float[torch.arange(N), oracle_model].mean().item()

    # Best single-model accuracy
    per_model_acc = oracle_float.mean(dim=0)  # (M,)
    best_single_acc = per_model_acc.max().item()
    best_single_model = models[per_model_acc.argmax().item()]
    avg_single_acc = per_model_acc.mean().item()

    # Model choice distribution
    choice_counts: dict[str, int] = {m: 0 for m in models}
    for idx in routed.tolist():
        choice_counts[models[idx]] += 1
    choice_pct = {m: round(choice_counts[m] / N * 100, 2) for m in models}

    summary = {
        "dataset": dataset,
        "n_questions": N,
        "oracle_accuracy": round(oracle_acc, 4),
        "router_accuracy": round(routing_acc, 4),
        "best_single_model": best_single_model,
        "best_single_acc": round(best_single_acc, 4),
        "avg_single_acc": round(avg_single_acc, 4),
        "per_model_accuracy": {m: round(per_model_acc[i].item(), 4) for i, m in enumerate(models)},
        "routing_distribution": choice_counts,
        "routing_distribution_pct": choice_pct,
    }

    predictions = [
        {
            "question_id": valid_ids[i],
            "routed_model": models[routed[i].item()],
            "routed_correct": bool(routed_correct[i].item()),
            "oracle_model": models[oracle_model[i].item()],
            "scores": {m: round(scores[i, j].item(), 5) for j, m in enumerate(models)},
        }
        for i in range(N)
    ]

    print(f"  routing_acc={routing_acc:.4f}  oracle={oracle_acc:.4f}  best_single={best_single_acc:.4f}")

    # -------------------------------------------------------------------------
    # Subtask evaluation
    # -------------------------------------------------------------------------
    subtask_result = None
    try:
        subtasks_data = load_subtask_oracle(dataset, models)
    except FileNotFoundError as e:
        print(f"  [subtask] skipped: {e}")
        return summary, None, predictions

    # Build a prediction lookup: qid -> chosen model (from full-task routing above)
    pred_map = {p["question_id"]: p["routed_model"] for p in predictions}

    # --- Overall subtask (same route as full-task, scored at subtask level) ---
    total_subs         = 0
    subtask_router_correct  = 0
    subtask_oracle_correct  = 0
    question_router_correct = 0
    question_oracle_correct = 0
    per_model_sub_correct: dict[str, int] = {m: 0 for m in models}

    for qid, entry in subtasks_data.items():
        if qid not in pred_map:
            continue
        sub_cols    = entry["cols"]
        binary      = entry["binary"]   # list of lists (n_subs × n_cols)
        n_subs      = len(binary)
        if n_subs == 0:
            continue
        sub_col2idx = {m: j for j, m in enumerate(sub_cols)}

        total_subs += n_subs
        chosen = pred_map[qid]
        j = sub_col2idx.get(chosen)
        if j is not None:
            chosen_correct = sum(binary[s][j] for s in range(n_subs))
            subtask_router_correct  += chosen_correct
            question_router_correct += int(chosen_correct == n_subs)

        best_subs = 0
        for m in models:
            jm = sub_col2idx.get(m)
            if jm is None:
                continue
            m_correct = sum(binary[s][jm] for s in range(n_subs))
            best_subs = max(best_subs, m_correct)
            per_model_sub_correct[m] += m_correct
        subtask_oracle_correct  += best_subs
        question_oracle_correct += int(best_subs == n_subs)

    n_with_subs = sum(
        1 for qid, entry in subtasks_data.items()
        if qid in pred_map and len(entry["binary"]) > 0
    )

    if total_subs == 0:
        print("  [subtask] no subtask data matched predictions — skipping")
        return summary, None, predictions

    per_model_sub_acc = {m: per_model_sub_correct[m] / total_subs for m in models}
    best_sub_model    = max(per_model_sub_acc, key=per_model_sub_acc.get)

    print(f"  [subtask overall]  router={subtask_router_correct/total_subs:.4f}"
          f"  oracle={subtask_oracle_correct/total_subs:.4f}"
          f"  q_router={question_router_correct/n_with_subs:.4f}"
          f"  q_oracle={question_oracle_correct/n_with_subs:.4f}")

    subtask_result = {
        "dataset":                  dataset,
        "n_questions":              n_with_subs,
        "n_subtasks":               total_subs,
        "oracle_subtask_accuracy":  round(subtask_oracle_correct  / total_subs,   6),
        "subtask_router_accuracy":  round(subtask_router_correct  / total_subs,   6),
        "subtask_router_correct":   subtask_router_correct,
        "oracle_question_accuracy": round(question_oracle_correct / n_with_subs,  6),
        "question_router_accuracy": round(question_router_correct / n_with_subs,  6),
        "question_router_correct":  question_router_correct,
        "best_single_model":        best_sub_model,
        "best_single_subtask_acc":  round(per_model_sub_acc[best_sub_model], 6),
        "avg_single_subtask_acc":   round(sum(per_model_sub_acc.values()) / len(per_model_sub_acc), 6),
        "per_model_subtask_accuracy": {m: round(v, 6) for m, v in per_model_sub_acc.items()},
    }

    # --- Chained subtask (independent model per subtask) ---------------------
    if run_chained:
        try:
            sub_texts = load_subtask_texts(dataset)
        except FileNotFoundError as e:
            print(f"  [chained] skipped: {e}")
            return summary, subtask_result, predictions

        # Flatten to (qid, sidx, text) for questions present in both oracle and texts
        flat_items: list[tuple[str, int, str]] = []
        for qid, entry in subtasks_data.items():
            if qid not in sub_texts or qid not in pred_map:
                continue
            n_rows  = len(entry["binary"])
            texts_q = sub_texts[qid]
            for sidx in range(min(len(texts_q), n_rows)):
                flat_items.append((qid, sidx, texts_q[sidx]))

        if not flat_items:
            print("  [chained] skipped: no flat_items (oracle/text overlap is empty)")
            return summary, subtask_result, predictions

        print(f"  [chained] Scoring {len(flat_items)} subtask texts × {len(models)} models "
              f"(batch_size={batch_size}) — this may take a while...")
        flat_questions = [t for _, _, t in flat_items]
        ch_scores = route_dataset(model, cap_strings, flat_questions, batch_size, device)
        ch_chosen = ch_scores.argmax(dim=1).tolist()

        chained_router_correct    = 0
        chained_micro_oracle_correct = 0
        q_sub_correct:        dict[str, int] = {}
        q_sub_total:          dict[str, int] = {}
        q_oracle_sub_correct: dict[str, int] = {}

        for idx, (qid, sidx, _) in enumerate(flat_items):
            entry       = subtasks_data[qid]
            sub_cols    = entry["cols"]
            binary      = entry["binary"]
            sub_col2idx = {m: j for j, m in enumerate(sub_cols)}

            chosen_model = models[ch_chosen[idx]]
            j       = sub_col2idx.get(chosen_model)
            correct = int(binary[sidx][j]) if j is not None else 0
            chained_router_correct += correct

            slot_feasible = int(any(
                binary[sidx][sub_col2idx[m]] for m in models if m in sub_col2idx
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

        ch_sub_acc           = chained_router_correct        / total_ch
        ch_micro_oracle_acc  = chained_micro_oracle_correct  / total_ch
        ch_q_acc             = ch_q_all             / n_ch_qs if n_ch_qs else 0.0
        oracle_chained_q_acc = oracle_chained_q_all / n_ch_qs if n_ch_qs else 0.0

        print(f"  [chained]  accuracy (questions)={ch_q_acc:.4f}"
              f"  oracle_chained={oracle_chained_q_acc:.4f}"
              f"  micro={ch_sub_acc:.4f}  micro_oracle={ch_micro_oracle_acc:.4f}")

        subtask_result.update({
            "chained_n_questions":              n_ch_qs,
            "chained_n_subtasks":               total_ch,
            "chained_accuracy":                 round(ch_q_acc, 6),
            "chained_correct":                  ch_q_all,
            "oracle_chained_accuracy":          round(oracle_chained_q_acc, 6),
            "chained_micro_subtask_accuracy":   round(ch_sub_acc, 6),
            "chained_micro_subtask_correct":    chained_router_correct,
            "oracle_micro_subtask_feasibility": round(ch_micro_oracle_acc, 6),
        })

    return summary, subtask_result, predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        type=str, default=CKPT_DEFAULT)
    p.add_argument("--datasets",    nargs="+", default=DATASETS)
    p.add_argument("--batch-size",  type=int, default=4,
                   help="Questions per forward pass (M models are scored per question)")
    p.add_argument("--out-dir",     type=str,
                   default=os.path.join(SCRIPT_DIR, "results"))
    p.add_argument("--exclude-models", nargs="*", default=[],
                   help="Model names to exclude from routing")
    p.add_argument("--subtask-out", type=str, default=None,
                   help="Path for subtask summary JSON (default: <out-dir>/subtask_summary.json)")
    p.add_argument("--chained", action="store_true", default=False,
                   help="Also run chained subtask evaluation (slow: re-runs LLM inference per subtask)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load capability strings
    with open(CAP_FILE) as f:
        raw_caps = json.load(f)

    exclude = set(args.exclude_models)
    models = [m for m in sorted(raw_caps.keys()) if m not in exclude]
    cap_strings = [raw_caps[m]["capability_string"] for m in models]
    M = len(models)
    print(f"Models ({M}): {models}")

    # Load ModelSAT
    sys.path.insert(0, SCRIPT_DIR)
    from model_sat import ModelSAT
    print(f"\nLoading ModelSAT from {args.ckpt} ...")
    model = ModelSAT(E5_PATH, PHI3_PATH, device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    # Checkpoint stores each component separately (as saved by finetune_router.py)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.connector.load_state_dict(ckpt["connector_state_dict"])
    model.llm.load_state_dict(ckpt["llm_state_dict"])
    model.eval()
    print("  Loaded.")

    if args.chained:
        print("\nNOTE: --chained enabled; chained subtask eval will re-run LLM inference on all subtask texts (slow).")

    # Evaluate
    all_summaries    = []
    all_subtasks     = []
    all_predictions: dict[str, list[dict]] = {}

    for ds in args.datasets:
        summary, sub_r, preds = evaluate_dataset(
            ds, model, models, cap_strings, args.batch_size, device,
            run_chained=args.chained,
        )
        all_summaries.append(summary)
        if sub_r is not None:
            all_subtasks.append(sub_r)
        all_predictions[ds] = preds

    # Save
    summary_path = os.path.join(args.out_dir, "summary.json")
    pred_path    = os.path.join(args.out_dir, "predictions.json")

    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f)

    print(f"\nSaved summary  → {summary_path}")
    print(f"Saved preds    → {pred_path}")

    if all_subtasks:
        subtask_path = args.subtask_out or os.path.join(args.out_dir, "subtask_summary.json")
        with open(subtask_path, "w") as f:
            json.dump(all_subtasks, f, indent=2)
        print(f"Saved subtask  → {subtask_path}")

    # Print final table
    print("\n--- Summary ---")
    for s in all_summaries:
        print(f"  {s['dataset']:12s}  router={s['router_accuracy']:.4f}  oracle={s['oracle_accuracy']:.4f}  best_single={s['best_single_acc']:.4f}")


if __name__ == "__main__":
    main()
