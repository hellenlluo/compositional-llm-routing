#!/usr/bin/env python3
"""
Re-derive ModelSAT summary.json for a subset of models without re-running
inference. Loads the already-computed per-question scores from predictions.json
and the binary oracle matrices to determine per-question correctness.

Usage:
  python eval_subset.py --models m1 m2 ... --out-dir results/no_qwen3
  python eval_subset.py --models mistral-7b-instruct-v0.3 qwen1.5-0.5b-chat \
      phi-4-mini-instruct llama-3.1-nemotron-nano-8b --out-dir results/small4
"""

import argparse
import json
import os
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
MATRICES_DIR = os.path.join(PROJECT_DIR, "outputs", "updated-matrices")
CLEANED_DIR  = os.path.join(PROJECT_DIR, "data_preprocessing", "cleaned_trimmed_data")
PREDS_FILE   = os.path.join(SCRIPT_DIR, "results", "predictions.json")
MODELS_DIR   = "/n/fs/scratch/dl3533/models"
E5_PATH      = os.path.join(MODELS_DIR, "e5-large-v2")
PHI3_PATH    = os.path.join(MODELS_DIR, "Phi-3-mini-128k-instruct")
CAP_FILE     = os.path.join(SCRIPT_DIR, "modelsat_data", "capability_representations.json")
CKPT_DEFAULT = os.path.join(SCRIPT_DIR, "router_checkpoint_v2", "model_sat_stage2.pt")
DATASETS     = ["morehopqa", "musique", "stepcot"]


def load_subtask_oracle(dataset: str) -> dict:
    path = os.path.join(MATRICES_DIR, f"{dataset}_subtasks.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Subtask oracle not found: {path}")
    with open(path) as f:
        return json.load(f)


def evaluate_subtask_subset(
    dataset: str,
    records: list[dict],
    models: list[str],
) -> dict | None:
    try:
        subtasks_data = load_subtask_oracle(dataset)
    except FileNotFoundError as e:
        print(f"  [subtask] skipped: {e}")
        return None

    pred_map = {rec["question_id"]: max(
        {m: rec["scores"][m] for m in models if m in rec["scores"]},
        key=lambda m: rec["scores"][m],
    ) for rec in records}

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
        binary      = entry["binary"]
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

    if total_subs == 0:
        return None

    n_with_subs = sum(
        1 for qid, entry in subtasks_data.items()
        if qid in pred_map and len(entry["binary"]) > 0
    )
    per_model_sub_acc = {m: per_model_sub_correct[m] / total_subs for m in models}
    best_sub_model    = max(per_model_sub_acc, key=per_model_sub_acc.get)

    print(f"  [subtask]  router={subtask_router_correct/total_subs:.4f}"
          f"  oracle={subtask_oracle_correct/total_subs:.4f}"
          f"  q_router={question_router_correct/n_with_subs:.4f}"
          f"  q_oracle={question_oracle_correct/n_with_subs:.4f}")

    return {
        "dataset":                  dataset,
        "n_questions":              n_with_subs,
        "n_subtasks":               total_subs,
        "oracle_subtask_accuracy":  round(subtask_oracle_correct  / total_subs,  6),
        "subtask_router_accuracy":  round(subtask_router_correct  / total_subs,  6),
        "subtask_router_correct":   subtask_router_correct,
        "oracle_question_accuracy": round(question_oracle_correct / n_with_subs, 6),
        "question_router_accuracy": round(question_router_correct / n_with_subs, 6),
        "question_router_correct":  question_router_correct,
        "best_single_model":        best_sub_model,
        "best_single_subtask_acc":  round(per_model_sub_acc[best_sub_model], 6),
        "avg_single_subtask_acc":   round(sum(per_model_sub_acc.values()) / len(per_model_sub_acc), 6),
        "per_model_subtask_accuracy": {m: round(v, 6) for m, v in per_model_sub_acc.items()},
    }


def load_oracle(dataset: str, models: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    """
    Returns (question_ids, {model: [0/1, ...]} binary correctness per question).
    Only loads columns for the requested models.
    """
    path = os.path.join(MATRICES_DIR, f"{dataset}_full_binary.json")
    print(f"  Loading oracle: {path}", flush=True)
    with open(path) as f:
        d = json.load(f)
    all_cols = d["cols"]
    rows     = d["rows"]
    values   = d["values"]

    col_idx = {m: all_cols.index(m) for m in models}
    correctness = {m: [row[col_idx[m]] for row in values] for m in models}
    return rows, correctness


def evaluate_subset(
    dataset: str,
    records: list[dict],
    models: list[str],
    qid_to_idx: dict[str, int],
    correctness: dict[str, list[int]],
) -> dict:
    n = len(records)
    router_correct  = 0
    oracle_correct  = 0
    choice_counts   = {m: 0 for m in models}

    for rec in records:
        qid = rec["question_id"]
        if qid not in qid_to_idx:
            continue
        idx = qid_to_idx[qid]

        # Re-route: argmax of ModelSAT scores restricted to subset
        scores = {m: rec["scores"][m] for m in models}
        routed = max(scores, key=scores.get)
        choice_counts[routed] += 1

        if correctness[routed][idx]:
            router_correct += 1

        # Oracle: does any model in subset get it right?
        if any(correctness[m][idx] for m in models):
            oracle_correct += 1

    per_model_acc = {
        m: sum(correctness[m]) / n for m in models
    }
    best_single_model = max(per_model_acc, key=per_model_acc.get)

    choice_pct = {m: round(choice_counts[m] / n * 100, 2) for m in models}

    return {
        "dataset":                  dataset,
        "n_questions":              n,
        "oracle_accuracy":          round(oracle_correct / n, 6),
        "router_accuracy":          round(router_correct / n, 6),
        "router_correct":           router_correct,
        "best_single_model":        best_single_model,
        "best_single_acc":          round(per_model_acc[best_single_model], 6),
        "avg_single_acc":           round(sum(per_model_acc.values()) / len(models), 6),
        "per_model_accuracy":       {m: round(v, 6) for m, v in per_model_acc.items()},
        "routing_distribution":     choice_counts,
        "routing_distribution_pct": choice_pct,
    }


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


def evaluate_chained_subset(
    dataset: str,
    subtasks_data: dict,
    models: list[str],
    cap_strings: list[str],
    sat_model,
    batch_size: int,
    device,
) -> dict | None:
    """Run ModelSAT inference independently on each subtask text and compute chained metrics."""
    import torch
    from tqdm import tqdm

    try:
        sub_texts = load_subtask_texts(dataset)
    except FileNotFoundError as e:
        print(f"  [chained] skipped: {e}")
        return None

    flat_items: list[tuple[str, int, str]] = []
    for qid, entry in subtasks_data.items():
        if qid not in sub_texts:
            continue
        n_rows  = len(entry["binary"])
        texts_q = sub_texts[qid]
        for sidx in range(min(len(texts_q), n_rows)):
            flat_items.append((qid, sidx, texts_q[sidx]))

    if not flat_items:
        print("  [chained] skipped: no flat_items (oracle/text overlap is empty)")
        return None

    print(f"  [chained] Scoring {len(flat_items)} subtask texts × {len(models)} models "
          f"(batch_size={batch_size}) — this may take a while...")

    # Reuse route_dataset from eval.py (already on sys.path)
    from eval import route_dataset
    flat_questions = [t for _, _, t in flat_items]
    ch_scores = route_dataset(sat_model, cap_strings, flat_questions, batch_size, device)
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

    print(f"  [chained]  accuracy={ch_q_acc:.4f}  oracle_chained={oracle_chained_q_acc:.4f}"
          f"  micro={ch_sub_acc:.4f}  micro_oracle={ch_micro_oracle_acc:.4f}")

    return {
        "chained_n_questions":              n_ch_qs,
        "chained_n_subtasks":               total_ch,
        "chained_accuracy":                 round(ch_q_acc, 6),
        "chained_correct":                  ch_q_all,
        "oracle_chained_accuracy":          round(oracle_chained_q_acc, 6),
        "chained_micro_subtask_accuracy":   round(ch_sub_acc, 6),
        "chained_micro_subtask_correct":    chained_router_correct,
        "oracle_micro_subtask_feasibility": round(ch_micro_oracle_acc, 6),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",   nargs="+", required=True,
                   help="Models to include in routing")
    p.add_argument("--out-dir",  type=str, required=True,
                   help="Directory to save summary.json and subtask_summary.json")
    p.add_argument("--preds",    type=str, default=PREDS_FILE,
                   help="Path to predictions.json from full eval")
    p.add_argument("--subtask-out", type=str, default=None,
                   help="Path for subtask JSON (default: <out-dir>/subtask_summary.json)")
    p.add_argument("--chained",  action="store_true", default=False,
                   help="Also run chained subtask evaluation (requires GPU + ModelSAT checkpoint)")
    p.add_argument("--ckpt",     type=str, default=CKPT_DEFAULT,
                   help="ModelSAT checkpoint (only needed with --chained)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Questions per forward pass for chained eval (only needed with --chained)")
    p.add_argument("--qid-filter", type=str, default=None,
                   help="Path to JSON {dataset: [qid, ...]} restricting evaluated questions")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    models = args.models
    print(f"Models ({len(models)}): {models}")

    # Load ModelSAT up-front if chained eval is requested
    sat_model = None
    cap_strings: list[str] = []
    device = None
    if args.chained:
        import torch
        sys.path.insert(0, SCRIPT_DIR)
        from model_sat import ModelSAT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print(f"Loading ModelSAT from {args.ckpt} ...")
        sat_model = ModelSAT(E5_PATH, PHI3_PATH, device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        sat_model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        sat_model.connector.load_state_dict(ckpt["connector_state_dict"])
        sat_model.llm.load_state_dict(ckpt["llm_state_dict"])
        sat_model.eval()
        with open(CAP_FILE) as f:
            raw_caps = json.load(f)
        cap_strings = [raw_caps[m]["capability_string"] for m in models]
        print("  Loaded.")

    qid_filter = None
    if args.qid_filter:
        with open(args.qid_filter) as _f:
            qid_filter = json.load(_f)
        print(f"  qid_filter loaded from {args.qid_filter}")

    print(f"\nLoading predictions from {args.preds} ...")
    with open(args.preds) as f:
        all_preds = json.load(f)

    summary          = []
    subtask_summary  = []
    for dataset in DATASETS:
        if dataset not in all_preds:
            print(f"  [{dataset}] not found in predictions, skipping")
            continue
        records = all_preds[dataset]
        allowed_qids = set(qid_filter[dataset]) if (qid_filter and dataset in qid_filter) else None
        if allowed_qids is not None:
            records = [r for r in records if r["question_id"] in allowed_qids]
        print(f"\n[{dataset}]  {len(records)} questions")

        question_ids, correctness = load_oracle(dataset, models)
        qid_to_idx = {qid: i for i, qid in enumerate(question_ids)}

        result = evaluate_subset(dataset, records, models, qid_to_idx, correctness)
        summary.append(result)
        print(f"  router={result['router_accuracy']:.4f}  "
              f"oracle={result['oracle_accuracy']:.4f}  "
              f"best_single={result['best_single_acc']:.4f}")
        print(f"  routing_distribution_pct: "
              + "  ".join(f"{m.split('-')[0]}={pct}%" for m, pct in result["routing_distribution_pct"].items()))

        sub_result = evaluate_subtask_subset(dataset, records, models)
        if sub_result is not None:
            if args.chained:
                subtasks_data = load_subtask_oracle(dataset)
                if allowed_qids is not None:
                    subtasks_data = {qid: v for qid, v in subtasks_data.items()
                                     if qid in allowed_qids}
                chained_fields = evaluate_chained_subset(
                    dataset, subtasks_data, models, cap_strings,
                    sat_model, args.batch_size, device,
                )
                if chained_fields:
                    sub_result.update(chained_fields)
            subtask_summary.append(sub_result)

    out_path = os.path.join(args.out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")

    if subtask_summary:
        sub_path = args.subtask_out or os.path.join(args.out_dir, "subtask_summary.json")
        with open(sub_path, "w") as f:
            json.dump(subtask_summary, f, indent=2)
        print(f"Saved → {sub_path}")


if __name__ == "__main__":
    main()
