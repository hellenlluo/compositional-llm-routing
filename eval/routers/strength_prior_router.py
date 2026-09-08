"""Cosine description router with an empirical-strength prior (SIMROUTE Variant C).

Score for routing question q to model m:
    score(q, m) = cosine(q_emb, desc_emb_m)  +  lambda * global_acc[m]

where global_acc[m] is model m's accuracy on the TRAIN split. Lambda is
selected on a held-out DEV split, then accuracy is reported on TEST.

This isolates whether description-cosine carries any signal beyond the
"always pick the strongest model" floor.

Usage:
    python -m eval.routers.strength_prior_router \
        --dataset morehopqa \
        --model-set no_qwen3 \
        --exclude-models qwen3-30b-a3b qwen3-4b-thinking-2507
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRICES_DIR = REPO_ROOT / "outputs" / "updated-matrices"
DATA_DIR = REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data"
CAPABILITY_PATH = REPO_ROOT / "eval" / "configs" / "model_descriptions.yaml"


def load_questions(dataset: str) -> dict[str, str]:
    if dataset == "morehopqa":
        data = json.loads((DATA_DIR / "morehopqa_cleaned.json").read_text())
        return {d["id"]: d["question"] for d in data}
    if dataset == "musique":
        out = {}
        for line in (DATA_DIR / "musique_cleaned.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d.get("id") or d.get("qid")] = d["question"]
        return out
    raise ValueError(dataset)


def load_full_matrix(dataset: str):
    m = json.loads((MATRICES_DIR / f"{dataset}_full_binary.json").read_text())
    return m["rows"], m["cols"], np.asarray(m["values"], dtype=np.int8)


def evaluate(routed_idx: np.ndarray, binary: np.ndarray) -> float:
    return float(binary[np.arange(binary.shape[0]), routed_idx].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="morehopqa", choices=["morehopqa", "musique"])
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument(
        "--lambdas", nargs="+", type=float,
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-models", nargs="*", default=None)
    ap.add_argument("--model-set", default=None,
                    help="Name for this model subset (used as output sub-folder).")
    args = ap.parse_args()

    exclude_set = set(args.exclude_models) if args.exclude_models else set()
    rows, cols, binary = load_full_matrix(args.dataset)

    # Filter to model set
    all_model_ids = list(cols)
    keep_idx = [i for i, m in enumerate(all_model_ids) if m not in exclude_set]
    model_ids = [all_model_ids[i] for i in keep_idx]
    binary = binary[:, keep_idx]

    questions = load_questions(args.dataset)

    n_q = len(rows)
    perm = np.random.default_rng(args.seed).permutation(n_q)
    n_train = int(round(n_q * args.train_frac))
    n_dev = int(round(n_q * args.dev_frac))
    train_idx = perm[:n_train]
    dev_idx = perm[n_train: n_train + n_dev]
    test_idx = perm[n_train + n_dev:]

    train_binary = binary[train_idx]
    dev_binary   = binary[dev_idx]
    test_binary  = binary[test_idx]

    test_rows     = [rows[i] for i in test_idx]
    dev_q_texts   = [questions[rows[i]] for i in dev_idx]
    test_q_texts  = [questions[rows[i]] for i in test_idx]

    descriptions = yaml.safe_load(CAPABILITY_PATH.read_text())["descriptions"]
    descs = [descriptions[m] for m in model_ids]

    print(f"[setup] dataset={args.dataset} encoder={args.encoder}")
    print(f"[setup] {n_train} train / {n_dev} dev / {len(test_idx)} test")
    print(f"[setup] {len(model_ids)} models: {model_ids}")
    if exclude_set:
        print(f"[setup] excluded: {sorted(exclude_set)}")

    encoder = SentenceTransformer(args.encoder)
    desc_emb = np.asarray(encoder.encode(descs, normalize_embeddings=True, show_progress_bar=False))

    print("[encode] dev questions ...")
    dev_q_emb = np.asarray(encoder.encode(
        dev_q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ))
    print("[encode] test questions ...")
    test_q_emb = np.asarray(encoder.encode(
        test_q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    ))

    # Global accuracy from train only
    global_acc = train_binary.mean(axis=0)

    # Baselines on test
    test_per_model = test_binary.mean(axis=0)
    oracle_acc = float(test_binary.max(axis=1).mean())
    rng_pick = np.random.default_rng(args.seed + 1).integers(0, len(model_ids), size=len(test_idx))
    rand_acc = float(test_binary[np.arange(len(test_idx)), rng_pick].mean())
    best_test_idx = int(np.argmax(test_per_model))
    always_best_train_idx = int(np.argmax(global_acc))

    # Cosine scores on dev and test
    dev_cos  = dev_q_emb @ desc_emb.T
    test_cos = test_q_emb @ desc_emb.T

    # Lambda sweep on dev
    print("\n=== Lambda sweep (DEV accuracy) ===")
    dev_results = []
    for lam in args.lambdas:
        scores = dev_cos + lam * global_acc[None, :]
        routed = scores.argmax(axis=1)
        acc = evaluate(routed, dev_binary)
        dev_results.append((lam, acc))
        print(f"  λ={lam:>6.3f}   dev acc = {acc:.4f}")

    best_lam, best_dev_acc = max(dev_results, key=lambda x: x[1])
    print(f"\n[selected] λ* = {best_lam}  (dev acc = {best_dev_acc:.4f})")

    # Final eval on test
    pure_cos_routed = test_cos.argmax(axis=1)
    pure_cos_acc = evaluate(pure_cos_routed, test_binary)

    sel_scores = test_cos + best_lam * global_acc[None, :]
    sel_routed = sel_scores.argmax(axis=1)
    sel_acc = evaluate(sel_routed, test_binary)

    print("\n=== TEST results ===")
    print(f"oracle:                     {oracle_acc:.4f}")
    print(f"random:                     {rand_acc:.4f}")
    print(f"always-best (train pick):   {model_ids[always_best_train_idx]} {float(test_per_model[always_best_train_idx]):.4f}")
    print(f"pure cosine (λ=0):          {pure_cos_acc:.4f}")
    print(f"cosine + strength (λ*={best_lam}): {sel_acc:.4f}")

    routed_counts_sel = {model_ids[j]: int((sel_routed == j).sum()) for j in range(len(model_ids))}
    routed_counts_pure = {model_ids[j]: int((pure_cos_routed == j).sum()) for j in range(len(model_ids))}

    out = {
        "dataset": args.dataset,
        "level": "full",
        "variant": "C_strength_prior",
        "encoder": args.encoder,
        "n_train": int(n_train),
        "n_dev": int(n_dev),
        "n_questions": int(len(test_idx)),
        "model_ids": model_ids,
        "selected_lambda": best_lam,
        "dev_results": dev_results,
        "router_accuracy": sel_acc,
        "oracle_accuracy": oracle_acc,
        "random_accuracy": rand_acc,
        "pure_cosine_accuracy": pure_cos_acc,
        "best_single_model": {
            "id": model_ids[best_test_idx],
            "accuracy": float(test_per_model[best_test_idx]),
        },
        "always_best_train": {
            "id": model_ids[always_best_train_idx],
            "accuracy": float(test_per_model[always_best_train_idx]),
        },
        "global_acc_train": {m: float(a) for m, a in zip(model_ids, global_acc)},
        "per_model_test_accuracy": {m: float(a) for m, a in zip(model_ids, test_per_model)},
        "routed_counts": routed_counts_sel,
        "routed_counts_pure": routed_counts_pure,
        "row_ids": test_rows,
        "routed_idx": sel_routed.tolist(),
    }

    if args.model_set:
        out_dir = REPO_ROOT / "outputs" / "routing" / args.model_set
    else:
        out_dir = REPO_ROOT / "outputs" / "routing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"strength_prior_{args.dataset}_full.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
