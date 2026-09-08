"""Embedding router with example-based descriptions (SIMROUTE Variant B).

For each candidate model we construct a "description" by concatenating the K
training-set questions where that model has the highest specialisation margin:
    spec(q, mk) = r(q, mk) - mean_{m' != mk} r(q, m')
The top-K examples where mk is correct are concatenated as its description.

Supports full-question and subtask-level routing with --model-set filtering,
mirroring the interface of embedding_router.py.

Usage:
    python -m eval.routers.example_router \
        --dataset morehopqa --level full \
        --model-set no_qwen3 \
        --exclude-models qwen3-30b-a3b qwen3-4b-thinking-2507
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRICES_DIR = REPO_ROOT / "outputs" / "updated-matrices"
DATA_DIR = REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data"
CAPABILITY_PATH = REPO_ROOT / "eval" / "configs" / "model_descriptions.yaml"


def load_questions(dataset: str) -> dict[str, dict]:
    """Return {qid: {'question': str, 'subtasks': [str, ...]}}."""
    if dataset == "morehopqa":
        data = json.loads((DATA_DIR / "morehopqa_cleaned.json").read_text())
        out = {}
        for d in data:
            subs = [s["question"] for s in d.get("question_decomposition", [])]
            out[d["id"]] = {"question": d["question"], "subtasks": subs}
        return out
    if dataset == "musique":
        out = {}
        for line in (DATA_DIR / "musique_cleaned.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                qid = d.get("id") or d.get("qid")
                subs = [s["question"] for s in d.get("question_decomposition", [])]
                out[qid] = {"question": d["question"], "subtasks": subs}
        return out
    raise ValueError(dataset)


def load_full_matrix(dataset: str):
    m = json.loads((MATRICES_DIR / f"{dataset}_full_binary.json").read_text())
    return m["rows"], m["cols"], np.asarray(m["values"], dtype=np.int8)


def load_subtask_matrices(dataset: str):
    return json.loads((MATRICES_DIR / f"{dataset}_subtasks.json").read_text())


def build_example_descriptions(
    rows: list[str],
    cols: list[str],
    binary: np.ndarray,
    questions: dict[str, dict],
    train_idx: np.ndarray,
    k: int,
) -> dict[str, str]:
    """For each model, pick top-K train questions by specialisation margin."""
    descs: dict[str, str] = {}
    train_bin = binary[train_idx]  # [n_train, n_models]
    n_models = train_bin.shape[1]
    for j, m in enumerate(cols):
        peer_mean = (train_bin.sum(axis=1) - train_bin[:, j]) / max(1, n_models - 1)
        score = train_bin[:, j].astype(np.float32) - peer_mean
        order = np.argsort(-score)
        picks: list[int] = []
        for o in order:
            if train_bin[o, j] == 1:
                picks.append(int(o))
            if len(picks) == k:
                break
        sel_qids = [rows[train_idx[p]] for p in picks]
        descs[m] = " \n".join(questions[qid]["question"] for qid in sel_qids)
    return descs


def evaluate(routed_idx: np.ndarray, binary: np.ndarray) -> tuple[float, np.ndarray]:
    correct = binary[np.arange(binary.shape[0]), routed_idx]
    return float(correct.mean()), correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="morehopqa", choices=["morehopqa", "musique"])
    ap.add_argument("--level", default="full", choices=["full", "subtask"])
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-models", nargs="*", default=None)
    ap.add_argument("--model-set", default=None,
                    help="Name for this model subset (used as output sub-folder).")
    args = ap.parse_args()

    rng_np = np.random.default_rng(args.seed)
    exclude_set = set(args.exclude_models) if args.exclude_models else set()
    questions_by_id = load_questions(args.dataset)

    # ── Load full-question matrix (always needed to build train descriptions) ──
    rows_full, cols_full, binary_full = load_full_matrix(args.dataset)
    all_model_ids = list(cols_full)
    keep_idx = [i for i, m in enumerate(all_model_ids) if m not in exclude_set]
    model_ids = [all_model_ids[i] for i in keep_idx]
    binary_full = binary_full[:, keep_idx]

    # Train/test split on full-question matrix
    n_q = len(rows_full)
    perm = rng_np.permutation(n_q)
    n_train = int(round(n_q * args.train_frac))
    train_idx = perm[:n_train]
    test_idx_full = perm[n_train:]

    print(f"[setup] dataset={args.dataset} level={args.level} encoder={args.encoder}")
    print(f"[setup] {n_train} train / {len(test_idx_full)} test, {len(model_ids)} models: {model_ids}")
    if exclude_set:
        print(f"[setup] excluded: {sorted(exclude_set)}")

    # Build example descriptions from train split
    print("[build] example descriptions from train split ...")
    ex_descs_map = build_example_descriptions(
        rows_full, model_ids, binary_full, questions_by_id, train_idx, args.k
    )
    ex_descs = [ex_descs_map[m] for m in model_ids]

    encoder = SentenceTransformer(args.encoder)
    print("[encode] example descriptions ...")
    desc_emb = np.asarray(encoder.encode(ex_descs, normalize_embeddings=True, show_progress_bar=False))

    if args.level == "full":
        # ── Full-question evaluation on test split ─────────────────────────────
        test_rows = [rows_full[i] for i in test_idx_full]
        test_binary = binary_full[test_idx_full]
        test_q_texts = [questions_by_id[qid]["question"] for qid in test_rows]

        print("[encode] test questions ...")
        q_emb = np.asarray(encoder.encode(
            test_q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
        ))

        sims = q_emb @ desc_emb.T
        routed_idx = sims.argmax(axis=1)

        router_acc, correct = evaluate(routed_idx, test_binary)
        oracle_acc = float(test_binary.max(axis=1).mean())
        rng_pick = np.random.default_rng(args.seed + 1).integers(0, len(model_ids), size=len(test_rows))
        rand_acc = float(test_binary[np.arange(len(test_rows)), rng_pick].mean())
        per_model_acc = test_binary.mean(axis=0)
        best_idx = int(np.argmax(per_model_acc))
        routed_counts = {model_ids[i]: int((routed_idx == i).sum()) for i in range(len(model_ids))}

        print(f"\noracle:         {oracle_acc:.4f}")
        print(f"random:         {rand_acc:.4f}")
        print(f"best single:    {model_ids[best_idx]} {float(per_model_acc[best_idx]):.4f}")
        print(f"example router: {router_acc:.4f}")

        out = {
            "dataset": args.dataset,
            "level": "full",
            "variant": "B_example",
            "encoder": args.encoder,
            "k": args.k,
            "n_train": n_train,
            "n_questions": len(test_rows),
            "model_ids": model_ids,
            "router_accuracy": router_acc,
            "oracle_accuracy": oracle_acc,
            "random_accuracy": rand_acc,
            "best_single_model": {"id": model_ids[best_idx], "accuracy": float(per_model_acc[best_idx])},
            "per_model_accuracy": {m: float(a) for m, a in zip(model_ids, per_model_acc)},
            "routed_counts": routed_counts,
            "row_ids": test_rows,
            "routed_idx": routed_idx.tolist(),
        }

    else:
        # ── Subtask-level evaluation ───────────────────────────────────────────
        # Use the same example descriptions built from full-question train split,
        # but embed each subtask text independently at test time.
        sub_mats = load_subtask_matrices(args.dataset)

        # Identify which base question IDs belong to test split
        test_qid_set = set(rows_full[i] for i in test_idx_full)

        q_texts: list[str] = []
        binary_rows: list[list[int]] = []
        slot_ids: list[str] = []

        # Align subtask matrix columns to model_ids
        first = next(iter(sub_mats.values()))
        sub_cols = list(first["cols"])
        sub_keep = [sub_cols.index(m) for m in model_ids if m in sub_cols]
        # Only use models present in both matrices
        model_ids_sub = [sub_cols[i] for i in sub_keep]

        for qid, mat in sub_mats.items():
            if qid not in test_qid_set:
                continue
            subs = questions_by_id[qid]["subtasks"]
            for step_idx, step_binary in enumerate(mat["binary"]):
                if step_idx >= len(subs):
                    break
                q_texts.append(subs[step_idx])
                binary_rows.append([step_binary[i] for i in sub_keep])
                slot_ids.append(f"{qid}::step{step_idx + 1}")

        binary_sub = np.asarray(binary_rows, dtype=np.int8)
        # Re-encode descriptions filtered to sub model set
        desc_emb_sub = np.asarray(encoder.encode(
            [ex_descs_map[m] for m in model_ids_sub],
            normalize_embeddings=True, show_progress_bar=False
        ))

        print(f"[setup] {len(q_texts)} subtask slots across {len(test_qid_set)} test questions")
        print("[encode] subtask texts ...")
        q_emb = np.asarray(encoder.encode(
            q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
        ))

        sims = q_emb @ desc_emb_sub.T
        routed_idx = sims.argmax(axis=1)

        router_acc, _ = evaluate(routed_idx, binary_sub)
        oracle_acc = float(binary_sub.max(axis=1).mean())
        rng_pick = np.random.default_rng(args.seed + 1).integers(0, len(model_ids_sub), size=len(q_texts))
        rand_acc = float(binary_sub[np.arange(len(q_texts)), rng_pick].mean())
        per_model_acc = binary_sub.mean(axis=0)
        best_idx = int(np.argmax(per_model_acc))
        routed_counts = {model_ids_sub[i]: int((routed_idx == i).sum()) for i in range(len(model_ids_sub))}

        print(f"\noracle (subtask): {oracle_acc:.4f}")
        print(f"random (subtask): {rand_acc:.4f}")
        print(f"example router:   {router_acc:.4f}")

        # Question-level chained accuracy
        slots_by_qid: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for slot_id, ridx in zip(slot_ids, routed_idx.tolist()):
            base = slot_id.split("::")[0]
            step = int(slot_id.split("::step")[1]) - 1
            slots_by_qid[base].append((step, ridx))

        n_chained = chained_correct = chained_oracle = 0
        for qid, slots in slots_by_qid.items():
            if qid not in sub_mats:
                continue
            mat = sub_mats[qid]
            b = mat["binary"]
            n_subs = len(b)
            if not n_subs or len(slots) != n_subs:
                continue
            n_chained += 1
            all_ok = all(b[s][sub_keep[ridx]] for s, ridx in slots)
            if all_ok:
                chained_correct += 1
            if all(any(b[s][i] for i in sub_keep) for s in range(n_subs)):
                chained_oracle += 1

        chained_acc = chained_correct / max(n_chained, 1)
        print(f"chained accuracy: {chained_acc:.4f}  (n={n_chained})")

        out = {
            "dataset": args.dataset,
            "level": "subtask",
            "variant": "B_example",
            "encoder": args.encoder,
            "k": args.k,
            "n_train": n_train,
            "n_subtask_slots": len(q_texts),
            "model_ids": model_ids_sub,
            "router_accuracy": router_acc,
            "oracle_accuracy": oracle_acc,
            "random_accuracy": rand_acc,
            "best_single_model": {"id": model_ids_sub[best_idx], "accuracy": float(per_model_acc[best_idx])},
            "per_model_accuracy": {m: float(a) for m, a in zip(model_ids_sub, per_model_acc)},
            "routed_counts": routed_counts,
            "chained_n_questions": n_chained,
            "chained_accuracy": chained_acc,
            "chained_oracle_accuracy": chained_oracle / max(n_chained, 1),
            "row_ids": slot_ids,
            "routed_idx": routed_idx.tolist(),
        }

    # Save
    if args.model_set:
        out_dir = REPO_ROOT / "outputs" / "routing" / args.model_set
    else:
        out_dir = REPO_ROOT / "outputs" / "routing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"example_{args.dataset}_{args.level}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
