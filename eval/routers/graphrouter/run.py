"""GraphRouter adapter for our morehopqa correctness matrix.

Reuses ``EncoderDecoderNet``, ``form_data``, and ``GNN_prediction`` from
``repos/GraphRouter/model/graph_nn.py`` but builds the input tensors
directly from ``outputs/updated-matrices/`` instead of round-tripping
through their CSV pipeline.

Loss is BCE on per-edge correctness (their default), mirrored from their
training loop.

Usage:
    python -m eval.routers.graphrouter.run --dataset morehopqa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPHROUTER_REPO = REPO_ROOT / "repos" / "GraphRouter"
MATRICES_DIR = REPO_ROOT / "outputs" / "updated-matrices"
DATA_DIR = REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data"
DESCRIPTIONS_PATH = REPO_ROOT / "eval" / "configs" / "model_descriptions.yaml"


# Stub object that mimics wandb's .log() — the GraphRouter trainer expects it.
class StdoutWandb:
    def __init__(self):
        self.history: list[dict] = []

    def log(self, payload: dict):
        clean = {k: float(v) if hasattr(v, "item") else v for k, v in payload.items()}
        self.history.append(clean)
        if len(self.history) % 10 == 0:
            tail = self.history[-1]
            print(
                f"  step {len(self.history):4d}  "
                f"train_loss={tail.get('train_loss', 0):.4f}  "
                f"val_loss={tail.get('validate_loss', 0):.4f}  "
                f"val_acc={tail.get('validate_accuracy', 0):.4f}  "
                f"val_f1={tail.get('validate_f1', 0):.4f}"
            )


TASK_DESCRIPTIONS = {
    "morehopqa": (
        "Multi-hop question answering over Wikipedia-style passages where the "
        "final hop typically requires a string or arithmetic transformation "
        "such as counting letters, syllables, reversing a number, or "
        "concatenating digits. Each question chains together two or more "
        "factual lookups before applying the final transformation."
    ),
    "musique": (
        "Multi-hop question answering over Wikipedia paragraphs that requires "
        "combining information from multiple passages. Questions are mostly "
        "factual lookups about named entities, places, films, biographies, "
        "and historical events with no manipulation step."
    ),
    "stepcot": (
        "Diagnostic radiology report interpretation. Each question presents "
        "a clinical report and asks for the best diagnostic option from a "
        "fixed list, requiring medical reasoning, anatomy knowledge, and "
        "differential diagnosis."
    ),
}


def load_questions(dataset: str) -> tuple[list[str], list[str]]:
    """Return (question_ids, question_texts) ordered to match the binary matrix rows."""
    if dataset == "morehopqa":
        data = json.loads((DATA_DIR / "morehopqa_cleaned.json").read_text())
        by_id = {d["id"]: d["question"] for d in data}
    elif dataset == "musique":
        by_id = {}
        for line in (DATA_DIR / "musique_cleaned.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                by_id[d.get("id") or d.get("qid")] = d["question"]
    elif dataset == "stepcot":
        data = json.loads((DATA_DIR / "stepcot_cleaned.json").read_text())
        by_id = {}
        for d in data:
            qid = d.get("id") or d.get("case_id")
            chain = d.get("vqa_chain", [])
            full_q = d.get("question") or (chain[-1]["question"] if chain else "")
            by_id[qid] = full_q
    else:
        raise ValueError(dataset)
    matrix = json.loads((MATRICES_DIR / f"{dataset}_full_binary.json").read_text())
    return matrix["rows"], [by_id[qid] for qid in matrix["rows"]], matrix["cols"], np.asarray(matrix["values"], dtype=np.int8)


def build_split_masks(n_query: int, n_llm: int, train_frac: float, dev_frac: float, seed: int):
    """Return (train_mask, dev_mask, test_mask) as float tensors of length n_query*n_llm.
    All edges of a given query share a split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_query)
    n_train = int(round(n_query * train_frac))
    n_dev = int(round(n_query * dev_frac))
    train_q = perm[:n_train]
    dev_q = perm[n_train : n_train + n_dev]
    test_q = perm[n_train + n_dev :]

    edge_query_idx = np.repeat(np.arange(n_query), n_llm)  # which query each edge belongs to
    train_mask = np.isin(edge_query_idx, train_q).astype(np.float32)
    dev_mask = np.isin(edge_query_idx, dev_q).astype(np.float32)
    test_mask = np.isin(edge_query_idx, test_q).astype(np.float32)
    return train_mask, dev_mask, test_mask, train_q, dev_q, test_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="morehopqa", choices=["morehopqa", "musique", "stepcot"])
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--embedding-dim", type=int, default=8)
    ap.add_argument("--edge-dim", type=int, default=3)
    ap.add_argument("--train-mask-rate", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    # Add GraphRouter repo to import path.
    sys.path.insert(0, str(GRAPHROUTER_REPO))
    sys.path.insert(0, str(GRAPHROUTER_REPO / "model"))
    from graph_nn import form_data, EncoderDecoderNet  # noqa: E402

    rows, q_texts, cols, binary = load_questions(args.dataset)
    n_query = len(rows)
    n_llm = len(cols)
    print(f"[setup] dataset={args.dataset}  {n_query} queries × {n_llm} models")

    descriptions = yaml.safe_load(DESCRIPTIONS_PATH.read_text())["descriptions"]
    llm_texts = [descriptions[m] for m in cols]
    task_text = TASK_DESCRIPTIONS[args.dataset]

    print(f"[encode] {args.encoder}")
    encoder = SentenceTransformer(args.encoder)
    q_emb = encoder.encode(q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    llm_emb = encoder.encode(llm_texts, normalize_embeddings=True, show_progress_bar=False)
    task_emb_single = encoder.encode([task_text], normalize_embeddings=True, show_progress_bar=False)[0]
    print(f"[encode] q={q_emb.shape} llm={llm_emb.shape} task={task_emb_single.shape}")

    # Per the form_data interface: task_id is one task embedding per query.
    # llm_features projection in their FeatureAlign is Linear(llm_dim, common*2),
    # but task projection is Linear(llm_dim, common). So task_id and llm_features
    # must share dim — both 768 here, fine.
    task_id_per_query = np.tile(task_emb_single[None, :], (n_query, 1))

    # Build edge cost (per_active_b) per LLM, broadcast to per-edge.
    params_records = json.loads((MATRICES_DIR / "model_params.json").read_text())
    params_by_id = {r["model_id"]: float(r["params_active_b"]) for r in params_records}
    cost_per_llm = np.asarray([params_by_id[m] for m in cols], dtype=np.float32)
    # Normalize costs to [0, 1] over our pool so the magnitude matches effect.
    cost_per_llm = cost_per_llm / max(cost_per_llm.max(), 1e-9)
    # Per-edge cost (effect_list ordering: query-major, llm-minor)
    cost_list = np.tile(cost_per_llm, n_query).astype(np.float32)

    # Per-edge effect = correctness ∈ {0, 1}. Order = query-major, llm-minor.
    effect_list = binary.reshape(-1).astype(np.float32)

    # Edge endpoints (their convention: org = query idx, des = LLM idx as 0..n_llm-1).
    edge_org_id = np.repeat(np.arange(n_query), n_llm).tolist()
    edge_des_id = (list(range(n_llm)) * n_query)

    # Splits.
    mask_train, mask_dev, mask_test, train_q, dev_q, test_q = build_split_masks(
        n_query, n_llm, args.train_frac, args.dev_frac, args.seed
    )
    print(f"[split] train_q={len(train_q)} dev_q={len(dev_q)} test_q={len(test_q)}")

    # Combined edge feature: their pipeline puts (cost, effect) here and the model
    # concatenates it with effect → 3-d. Match that exactly.
    combined_edge = np.stack([cost_list, effect_list], axis=1)  # (E, 2)

    # Label = one-hot of best LLM per query (their convention).
    # With binary correctness, ties are broken by argmax of `effect_list` which
    # picks the first-correct-LLM. We'll use effect (=correctness) directly so
    # the label aligns with the "is this the best LLM" interpretation;
    # the BCELoss in their trainer compares the masked predicted edges against
    # `data.label[mask]` element-wise.
    effect_re = effect_list.reshape(n_query, n_llm)
    label_arg = effect_re.argmax(axis=1)
    label_onehot = np.zeros_like(effect_re)
    label_onehot[np.arange(n_query), label_arg] = 1.0
    # Replace label with a per-edge correctness signal (more informative than one-hot).
    # GraphRouter's trainer only uses label[mask] for BCE, so the meaning of label
    # is interchangeable as long as it's a [0, 1] target. Using raw correctness
    # is a strictly better target for our binary setup.
    label_perEdge = effect_list.copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    fd = form_data(device)

    # Their trainer calls `.clone()` on data.edge_mask, so it must be a tensor.
    mask_train_t = torch.tensor(mask_train, dtype=torch.float32, device=device)
    mask_dev_t = torch.tensor(mask_dev, dtype=torch.float32, device=device)
    mask_test_t = torch.tensor(mask_test, dtype=torch.float32, device=device)

    def make_data(edge_mask_t):
        return fd.formulation(
            task_id=task_id_per_query,
            query_feature=q_emb,
            llm_feature=llm_emb,
            org_node=edge_org_id,
            des_node=edge_des_id,
            edge_feature=effect_list,
            label=label_perEdge.reshape(-1, 1),
            edge_mask=edge_mask_t,
            combined_edge=combined_edge,
            train_mask=mask_train_t,
            valide_mask=mask_dev_t,
            test_mask=mask_test_t,
        )

    data_train = make_data(mask_train_t)
    data_dev = make_data(mask_dev_t)
    data_test = make_data(mask_test_t)

    config = {
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "train_epoch": args.epochs,
        "train_mask_rate": args.train_mask_rate,
        "batch_size": args.batch_size,
        "llm_num": n_llm,
        "model_path": str(REPO_ROOT / "outputs" / "routing" / f"graphrouter_{args.dataset}.pt"),
    }
    Path(config["model_path"]).parent.mkdir(parents=True, exist_ok=True)

    print(f"[train] {args.epochs} epochs, hidden={args.embedding_dim}, edge_dim={args.edge_dim}")
    t0 = time.time()

    # Custom training loop: track actual dev routing accuracy (not their broken val F1).
    model = EncoderDecoderNet(
        query_feature_dim=q_emb.shape[1],
        llm_feature_dim=llm_emb.shape[1],
        hidden_features=args.embedding_dim,
        in_edges=args.edge_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCELoss()

    train_mask_bool = mask_train_t.bool()
    dev_mask_bool = mask_dev_t.bool()
    test_mask_bool = mask_test_t.bool()

    dev_binary = binary[dev_q]  # (n_dev, n_llm)

    def dev_routing_accuracy() -> float:
        model.eval()
        edge_can_see = train_mask_bool
        with torch.no_grad():
            edge_pred = model(
                task_id=data_train.task_id,
                query_features=data_train.query_features,
                llm_features=data_train.llm_features,
                edge_index=data_train.edge_index,
                edge_mask=dev_mask_bool,
                edge_can_see=edge_can_see,
                edge_weight=data_train.combined_edge,
            )
        edge_pred_qm = edge_pred.reshape(-1, n_llm).cpu().numpy()
        routed = edge_pred_qm.argmax(axis=1)
        return float(dev_binary[np.arange(len(dev_q)), routed].mean())

    best_dev_acc = -1.0
    best_state = None
    history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        loss_accum = 0.0
        for _ in range(args.batch_size):
            mask = train_mask_bool.clone()
            random_mask = torch.rand(mask.size(), device=device) < args.train_mask_rate
            mask = torch.where(mask & random_mask, torch.tensor(False, device=device), mask)
            edge_can_see = torch.logical_and(~mask, train_mask_bool)
            optimizer.zero_grad()
            preds = model(
                task_id=data_train.task_id,
                query_features=data_train.query_features,
                llm_features=data_train.llm_features,
                edge_index=data_train.edge_index,
                edge_mask=mask,
                edge_can_see=edge_can_see,
                edge_weight=data_train.combined_edge,
            )
            loss = bce(preds.reshape(-1), data_train.label[mask].reshape(-1))
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
        loss_mean = loss_accum / args.batch_size

        dev_acc = dev_routing_accuracy()
        history.append({"epoch": epoch + 1, "train_loss": loss_mean, "dev_routing_acc": dev_acc})
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:4d}  train_loss={loss_mean:.4f}  dev_routing_acc={dev_acc:.4f}  (best {best_dev_acc:.4f})")

    train_time = time.time() - t0
    print(f"[train] done in {train_time:.1f}s — best dev routing acc = {best_dev_acc:.4f}")

    # Restore best checkpoint.
    model.load_state_dict(best_state)
    torch.save(best_state, config["model_path"])
    model.eval()
    gnn = type("Stub", (), {"model": model})()  # for the eval block below

    # Build test mask + edges visible during inference (train ∪ dev — paper convention).
    test_mask_t = torch.tensor(mask_test, dtype=torch.bool)
    edge_can_see_test = torch.logical_or(
        torch.tensor(mask_train, dtype=torch.bool),
        torch.tensor(mask_dev, dtype=torch.bool),
    )
    with torch.no_grad():
        edge_pred = gnn.model(
            task_id=data_test.task_id,
            query_features=data_test.query_features,
            llm_features=data_test.llm_features,
            edge_index=data_test.edge_index,
            edge_mask=test_mask_t,
            edge_can_see=edge_can_see_test,
            edge_weight=data_test.combined_edge,
        )
    edge_pred_qm = edge_pred.reshape(-1, n_llm).cpu().numpy()
    routed_idx = edge_pred_qm.argmax(axis=1)

    # Look up actual correctness from the matrix on test queries.
    test_binary = binary[test_q]
    routed_correct = test_binary[np.arange(len(test_q)), routed_idx]
    router_acc = float(routed_correct.mean())

    # Baselines on the SAME test split (seed 0) for apples-to-apples.
    per_model_test = test_binary.mean(axis=0)
    oracle_acc = float(test_binary.max(axis=1).mean())
    rng_pick = np.random.default_rng(args.seed + 1).integers(0, n_llm, size=len(test_q))
    rand_acc = float(test_binary[np.arange(len(test_q)), rng_pick].mean())

    # "Always-best" picked using TRAIN accuracy.
    train_acc_per_model = binary[train_q].mean(axis=0)
    always_best_idx = int(train_acc_per_model.argmax())
    always_best_id = cols[always_best_idx]
    always_best_acc = float(per_model_test[always_best_idx])

    # Routed-to histogram.
    hist = {cols[j]: int((routed_idx == j).sum()) for j in range(n_llm)}

    print("\n=== TEST (n={}) ===".format(len(test_q)))
    print(f"oracle:                          {oracle_acc:.4f}")
    print(f"always-best (train-pick):        {always_best_id:35s} {always_best_acc:.4f}")
    print(f"random uniform:                  {rand_acc:.4f}")
    print(f"GraphRouter:                     {router_acc:.4f}")
    print("\nrouted-to (GraphRouter):")
    for m, c in sorted(hist.items(), key=lambda x: -x[1]):
        if c == 0:
            continue
        print(f"  {m:35s} {c:5d}  ({c / len(test_q):.1%})")

    out = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "n_train_query": int(len(train_q)),
        "n_dev_query": int(len(dev_q)),
        "n_test_query": int(len(test_q)),
        "n_models": int(n_llm),
        "model_ids": cols,
        "config": config,
        "best_dev_routing_acc": float(best_dev_acc),
        "training_history": history,
        "embedding_dim": args.embedding_dim,
        "edge_dim": args.edge_dim,
        "epochs": args.epochs,
        "train_time_sec": train_time,
        "test_oracle": oracle_acc,
        "test_random": rand_acc,
        "test_best_single_train_pick": {"id": always_best_id, "accuracy": always_best_acc},
        "test_router_accuracy": router_acc,
        "routed_counts": hist,
        "per_model_test_accuracy": {m: float(a) for m, a in zip(cols, per_model_test)},
    }
    out_path = REPO_ROOT / "outputs" / "routing" / f"graphrouter_{args.dataset}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
