"""Zero-shot embedding-cosine router.

Embeds each model's capability description and each input question with the
same sentence encoder, then routes every question to the model whose
description has the highest cosine similarity. Compares router accuracy
against per-model accuracy, the oracle, and a uniform-random baseline using
the binary correctness matrix in `outputs/updated-matrices/`.

Usage:
    python -m eval.routers.embedding_router \
        --dataset morehopqa \
        --level full
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRICES_DIR = REPO_ROOT / "outputs" / "updated-matrices"
DATA_DIR = REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data"
DESCRIPTIONS_PATH = REPO_ROOT / "eval" / "configs" / "model_descriptions.yaml"


def load_questions(dataset: str) -> dict[str, dict]:
    """Return {qid: {'question': str, 'subtasks': [str, ...]}}."""
    if dataset == "morehopqa":
        path = DATA_DIR / "morehopqa_cleaned.json"
        data = json.loads(path.read_text())
        out = {}
        for d in data:
            subs = [s["question"] for s in d.get("question_decomposition", [])]
            out[d["id"]] = {"question": d["question"], "subtasks": subs}
        return out
    if dataset == "musique":
        path = DATA_DIR / "musique_cleaned.jsonl"
        out = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            qid = d.get("id") or d.get("qid")
            subs = [s["question"] for s in d.get("question_decomposition", [])]
            out[qid] = {"question": d["question"], "subtasks": subs}
        return out
    if dataset == "stepcot":
        path = DATA_DIR / "stepcot_cleaned.json"
        data = json.loads(path.read_text())
        out = {}
        for d in data:
            qid = d.get("id") or d.get("case_id")
            chain = d.get("vqa_chain", [])
            # final-question text varies; fall back to step 7 question
            full_q = d.get("question") or (chain[-1]["question"] if chain else "")
            subs = [s["question"] for s in chain]
            out[qid] = {"question": full_q, "subtasks": subs}
        return out
    raise ValueError(dataset)


def load_full_matrix(dataset: str):
    m = json.loads((MATRICES_DIR / f"{dataset}_full_binary.json").read_text())
    return m["rows"], m["cols"], np.asarray(m["values"], dtype=np.int8)


def load_subtask_matrices(dataset: str):
    return json.loads((MATRICES_DIR / f"{dataset}_subtasks.json").read_text())


def cosine_route(q_emb: np.ndarray, m_emb: np.ndarray) -> np.ndarray:
    # both already L2-normalized by sentence-transformers when normalize=True
    return q_emb @ m_emb.T  # [n_q, n_m]


def evaluate_routing(
    routed_idx: np.ndarray, binary: np.ndarray
) -> tuple[float, np.ndarray]:
    n_q = binary.shape[0]
    correct = binary[np.arange(n_q), routed_idx]
    return correct.mean(), correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="morehopqa", choices=["morehopqa", "musique", "stepcot"])
    ap.add_argument("--level", default="full", choices=["full", "subtask"])
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude-models", nargs="*", default=None,
                    help="Model IDs to exclude from routing (filters matrix columns).")
    ap.add_argument("--model-set", default=None,
                    help="Name for this model subset (used as output sub-folder, e.g. no_qwen3).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    exclude_set = set(args.exclude_models) if args.exclude_models else set()

    descriptions = yaml.safe_load(DESCRIPTIONS_PATH.read_text())["descriptions"]

    if args.level == "full":
        rows, cols, binary = load_full_matrix(args.dataset)
        questions_by_id = load_questions(args.dataset)
        # restrict descriptions to models present in this matrix, minus exclusions
        all_model_ids = list(cols)
        keep_idx  = [i for i, m in enumerate(all_model_ids) if m not in exclude_set]
        model_ids = [all_model_ids[i] for i in keep_idx]
        binary    = binary[:, keep_idx]
        descs = [descriptions[m] for m in model_ids]
        q_texts = [questions_by_id[qid]["question"] for qid in rows]
    else:
        # Flatten subtasks across all questions; align with subtask matrices.
        sub_mats = load_subtask_matrices(args.dataset)
        questions_by_id = load_questions(args.dataset)
        # Use the first matrix to read column order (same across all questions).
        first = next(iter(sub_mats.values()))
        all_model_ids = list(first["cols"])
        keep_idx  = [i for i, m in enumerate(all_model_ids) if m not in exclude_set]
        model_ids = [all_model_ids[i] for i in keep_idx]
        descs = [descriptions[m] for m in model_ids]
        q_texts: list[str] = []
        binary_rows: list[list[int]] = []
        rows: list[str] = []  # tag is "qid::step{i}"
        for qid, mat in sub_mats.items():
            subs = questions_by_id[qid]["subtasks"]
            for step_idx, step_binary in enumerate(mat["binary"]):
                if step_idx >= len(subs):
                    break
                q_texts.append(subs[step_idx])
                binary_rows.append([step_binary[i] for i in keep_idx])
                rows.append(f"{qid}::step{step_idx + 1}")
        binary = np.asarray(binary_rows, dtype=np.int8)

    if exclude_set:
        print(f"[setup] excluded models: {sorted(exclude_set)}")
    print(f"[setup] dataset={args.dataset} level={args.level}")
    print(f"[setup] {len(q_texts)} queries, {len(model_ids)} models: {model_ids}")
    print(f"[setup] encoder={args.encoder}")

    encoder = SentenceTransformer(args.encoder)
    print("[encode] descriptions ...")
    m_emb = encoder.encode(descs, normalize_embeddings=True, show_progress_bar=False)
    print("[encode] questions ...")
    q_emb = encoder.encode(q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    sims = cosine_route(np.asarray(q_emb), np.asarray(m_emb))
    routed_idx = sims.argmax(axis=1)

    _, router_correct = evaluate_routing(routed_idx, binary)
    per_model_acc = binary.mean(axis=0)
    rng_per_hop = np.array([
        binary[i, rng.randrange(len(model_ids))] for i in range(binary.shape[0])
    ], dtype=np.int8)

    if args.level == "subtask":
        # Aggregate per question. A question is "covered" by the router iff
        # every hop in its chain is answered correctly by its routed model.
        # Same definition for the oracle (every hop has >=1 correct model)
        # and per-model accuracy (the model gets every hop right). This
        # matches the strict question-level oracle used elsewhere
        # (router_analysis/heterogeneous_accuracy.py).
        from collections import defaultdict
        q_router_correct: dict[str, list[int]] = defaultdict(list)
        q_oracle_covered: dict[str, list[int]] = defaultdict(list)
        q_per_model: dict[str, list[np.ndarray]] = defaultdict(list)
        q_random:    dict[str, list[int]] = defaultdict(list)
        for i, tag in enumerate(rows):
            qid = tag.split("::", 1)[0]
            q_router_correct[qid].append(int(router_correct[i]))
            q_oracle_covered[qid].append(int(binary[i].max()))
            q_per_model[qid].append(binary[i])
            q_random[qid].append(int(rng_per_hop[i]))

        q_router_score = np.array([int(all(v)) for v in q_router_correct.values()])
        q_oracle_score = np.array([int(all(v)) for v in q_oracle_covered.values()])
        # per-model: question correct iff this model answers every hop
        per_model_acc = np.zeros(len(model_ids), dtype=float)
        for hops in q_per_model.values():
            stacked = np.stack(hops, axis=0)
            per_model_acc += stacked.all(axis=0)
        per_model_acc /= max(len(q_per_model), 1)
        q_random_score = np.array([int(all(v)) for v in q_random.values()])

        router_acc = float(q_router_score.mean())
        oracle_acc = float(q_oracle_score.mean())
        rng_acc    = float(q_random_score.mean())
        n_questions_eval = int(len(q_router_score))
        n_hops_eval = int(len(rows))
    else:
        router_acc = float(router_correct.mean())
        oracle_acc = float((binary.max(axis=1)).mean())
        rng_acc    = float(rng_per_hop.mean())
        n_questions_eval = int(binary.shape[0])
        n_hops_eval = n_questions_eval

    # Routed-to histogram
    routed_counts = {model_ids[i]: int((routed_idx == i).sum()) for i in range(len(model_ids))}

    print("\n=== Results ===")
    if args.level == "subtask":
        print(f"[metric] question-level: every hop must be covered (strict oracle)")
        print(f"[metric] {n_questions_eval} questions, {n_hops_eval} hops total")
    print(f"router (cos-sim, argmax):  {router_acc:.4f}")
    print(f"oracle (best-of-models):   {oracle_acc:.4f}")
    print(f"random (uniform):          {rng_acc:.4f}")
    print("per-model accuracy:")
    best_single = (None, -1.0)
    for m, a in sorted(zip(model_ids, per_model_acc), key=lambda x: -x[1]):
        print(f"  {m:35s} {a:.4f}")
        if a > best_single[1]:
            best_single = (m, a)
    print(f"best single model:         {best_single[0]} ({best_single[1]:.4f})")
    print("\nrouter routed-to histogram:")
    for m, c in sorted(routed_counts.items(), key=lambda x: -x[1]):
        print(f"  {m:35s} {c:5d}  ({c / len(q_texts):.1%})")

    if args.out:
        out_path = Path(args.out)
    elif args.model_set:
        out_path = (
            REPO_ROOT / "outputs" / "routing" / args.model_set
            / f"embedding_{args.dataset}_{args.level}.json"
        )
    else:
        out_path = REPO_ROOT / "outputs" / "routing" / f"embedding_{args.dataset}_{args.level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "level": args.level,
        "encoder": args.encoder,
        "n_queries": len(q_texts),
        "model_ids": model_ids,
        "router_accuracy": float(router_acc),
        "oracle_accuracy": float(oracle_acc),
        "random_accuracy": float(rng_acc),
        "best_single_model": {"id": best_single[0], "accuracy": float(best_single[1])},
        "per_model_accuracy": {m: float(a) for m, a in zip(model_ids, per_model_acc)},
        "routed_counts": routed_counts,
        "routed_idx": routed_idx.tolist(),
        "row_ids": rows,
    }, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
