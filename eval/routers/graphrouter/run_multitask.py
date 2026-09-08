"""GraphRouter — multi-task version (morehopqa + musique + stepcot jointly).

Same model and training as ``run.py`` but the graph contains queries from
all three datasets, each query node carrying its own task embedding.
This is the setting where GraphRouter should outperform always-best,
because different tasks have genuinely different best models.

Usage:
    python -m eval.routers.graphrouter.run_multitask
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

DATASETS = ["morehopqa", "musique", "stepcot"]

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


def load_dataset_questions(dataset: str) -> dict[str, str]:
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
    if dataset == "stepcot":
        data = json.loads((DATA_DIR / "stepcot_cleaned.json").read_text())
        out = {}
        for d in data:
            qid = d.get("id") or d.get("case_id")
            chain = d.get("vqa_chain", [])
            full_q = d.get("question") or (chain[-1]["question"] if chain else "")
            out[qid] = full_q
        return out
    raise ValueError(dataset)


def stratified_split(task_of_query: np.ndarray, train_frac: float, dev_frac: float, seed: int):
    """Stratified split by task: each task contributes proportional rows to each split."""
    rng = np.random.default_rng(seed)
    n_query = len(task_of_query)
    train_q, dev_q, test_q = [], [], []
    for t in np.unique(task_of_query):
        idx = np.where(task_of_query == t)[0]
        idx = rng.permutation(idx)
        n_t = len(idx)
        n_train_t = int(round(n_t * train_frac))
        n_dev_t = int(round(n_t * dev_frac))
        train_q.extend(idx[:n_train_t].tolist())
        dev_q.extend(idx[n_train_t : n_train_t + n_dev_t].tolist())
        test_q.extend(idx[n_train_t + n_dev_t :].tolist())
    return np.array(train_q), np.array(dev_q), np.array(test_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--embedding-dim", type=int, default=16)
    ap.add_argument("--edge-dim", type=int, default=3)
    ap.add_argument("--train-mask-rate", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--subsample-musique", type=int, default=None,
                    help="Optionally cap musique to N queries to balance with smaller tasks.")
    args = ap.parse_args()

    sys.path.insert(0, str(GRAPHROUTER_REPO))
    sys.path.insert(0, str(GRAPHROUTER_REPO / "model"))
    from graph_nn import form_data, EncoderDecoderNet  # noqa: E402

    # ------------------------------------------------------------
    # Load all three datasets, concatenate.
    # ------------------------------------------------------------
    all_qids: list[str] = []
    all_q_texts: list[str] = []
    task_of_query: list[int] = []
    binary_rows: list[np.ndarray] = []
    cols_canonical = None
    n_per_task: dict[str, int] = {}

    for t_idx, ds in enumerate(DATASETS):
        m = json.loads((MATRICES_DIR / f"{ds}_full_binary.json").read_text())
        if cols_canonical is None:
            cols_canonical = m["cols"]
        else:
            assert cols_canonical == m["cols"], f"column order differs in {ds}"
        ds_questions = load_dataset_questions(ds)
        rng = np.random.default_rng(args.seed)
        rows = m["rows"]
        if ds == "musique" and args.subsample_musique is not None:
            keep_idx = rng.choice(len(rows), size=args.subsample_musique, replace=False)
            keep_idx.sort()
            rows = [rows[i] for i in keep_idx]
            values = [m["values"][i] for i in keep_idx]
        else:
            values = m["values"]
        n_per_task[ds] = len(rows)
        for qid, val in zip(rows, values):
            all_qids.append(qid)
            all_q_texts.append(ds_questions[qid])
            task_of_query.append(t_idx)
            binary_rows.append(np.asarray(val, dtype=np.int8))

    n_query = len(all_qids)
    n_llm = len(cols_canonical)
    binary = np.stack(binary_rows, axis=0)  # (n_query, n_llm)
    task_of_query = np.array(task_of_query, dtype=np.int32)

    print(f"[setup] {n_query} queries total: {n_per_task}")
    print(f"[setup] {n_llm} models, encoder={args.encoder}")

    # ------------------------------------------------------------
    # Encode questions and node descriptions.
    # ------------------------------------------------------------
    descriptions = yaml.safe_load(DESCRIPTIONS_PATH.read_text())["descriptions"]
    llm_texts = [descriptions[m_id] for m_id in cols_canonical]
    task_texts = [TASK_DESCRIPTIONS[ds] for ds in DATASETS]

    print(f"[encode] queries (~{n_query}) ...")
    encoder = SentenceTransformer(args.encoder)
    q_emb = encoder.encode(all_q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    llm_emb = encoder.encode(llm_texts, normalize_embeddings=True, show_progress_bar=False)
    task_emb_per_task = encoder.encode(task_texts, normalize_embeddings=True, show_progress_bar=False)
    # Per-query task embedding (matches form_data interface).
    task_id_per_query = task_emb_per_task[task_of_query]
    print(f"[encode] q={q_emb.shape} llm={llm_emb.shape} task={task_emb_per_task.shape}")

    # ------------------------------------------------------------
    # Edges, costs, labels.
    # ------------------------------------------------------------
    params_records = json.loads((MATRICES_DIR / "model_params.json").read_text())
    params_by_id = {r["model_id"]: float(r["params_active_b"]) for r in params_records}
    cost_per_llm = np.asarray([params_by_id[m_id] for m_id in cols_canonical], dtype=np.float32)
    cost_per_llm = cost_per_llm / max(cost_per_llm.max(), 1e-9)
    cost_list = np.tile(cost_per_llm, n_query).astype(np.float32)
    effect_list = binary.reshape(-1).astype(np.float32)

    edge_org_id = np.repeat(np.arange(n_query), n_llm).tolist()
    edge_des_id = (list(range(n_llm)) * n_query)
    combined_edge = np.stack([cost_list, effect_list], axis=1)
    label_perEdge = effect_list.copy()

    # ------------------------------------------------------------
    # Stratified splits.
    # ------------------------------------------------------------
    train_q, dev_q, test_q = stratified_split(task_of_query, args.train_frac, args.dev_frac, args.seed)
    print(f"[split] train_q={len(train_q)}  dev_q={len(dev_q)}  test_q={len(test_q)}")

    edge_query_idx = np.repeat(np.arange(n_query), n_llm)
    mask_train = np.isin(edge_query_idx, train_q).astype(np.float32)
    mask_dev = np.isin(edge_query_idx, dev_q).astype(np.float32)
    mask_test = np.isin(edge_query_idx, test_q).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    fd = form_data(device)
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

    # ------------------------------------------------------------
    # Custom training loop with dev routing accuracy as the early-stop metric.
    # ------------------------------------------------------------
    print(f"[train] {args.epochs} epochs, hidden={args.embedding_dim}, edge_dim={args.edge_dim}")
    t0 = time.time()
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
    dev_binary = binary[dev_q]
    dev_task = task_of_query[dev_q]
    test_binary = binary[test_q]
    test_task = task_of_query[test_q]

    def routing_accuracy(mask_bool: torch.Tensor, q_idx: np.ndarray, target_binary: np.ndarray) -> tuple[float, np.ndarray]:
        model.eval()
        with torch.no_grad():
            edge_pred = model(
                task_id=data_train.task_id,
                query_features=data_train.query_features,
                llm_features=data_train.llm_features,
                edge_index=data_train.edge_index,
                edge_mask=mask_bool,
                edge_can_see=train_mask_bool,
                edge_weight=data_train.combined_edge,
            )
        edge_pred_qm = edge_pred.reshape(-1, n_llm).cpu().numpy()
        routed = edge_pred_qm.argmax(axis=1)
        return float(target_binary[np.arange(len(q_idx)), routed].mean()), routed

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

        dev_acc, _ = routing_accuracy(dev_mask_bool, dev_q, dev_binary)
        history.append({"epoch": epoch + 1, "train_loss": loss_mean, "dev_routing_acc": dev_acc})
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:4d}  loss={loss_mean:.4f}  dev_acc={dev_acc:.4f}  (best {best_dev_acc:.4f})")

    train_time = time.time() - t0
    print(f"[train] done in {train_time:.1f}s — best dev acc = {best_dev_acc:.4f}")

    # ------------------------------------------------------------
    # Final evaluation: overall + per-task on the test split.
    # ------------------------------------------------------------
    model.load_state_dict(best_state)
    test_acc, routed_idx = routing_accuracy(test_mask_bool, test_q, test_binary)

    # Baselines on the SAME test split.
    train_acc_per_model = binary[train_q].mean(axis=0)
    always_best_idx = int(train_acc_per_model.argmax())
    always_best_id = cols_canonical[always_best_idx]
    always_best_acc = float(test_binary[:, always_best_idx].mean())
    oracle_acc = float(test_binary.max(axis=1).mean())
    rng_pick = np.random.default_rng(args.seed + 1).integers(0, n_llm, size=len(test_q))
    rand_acc = float(test_binary[np.arange(len(test_q)), rng_pick].mean())

    # Per-task breakdown.
    print("\n=== TEST overall ===")
    print(f"  oracle:                  {oracle_acc:.4f}")
    print(f"  always-best ({always_best_id}): {always_best_acc:.4f}")
    print(f"  random uniform:          {rand_acc:.4f}")
    print(f"  GraphRouter:             {test_acc:.4f}")

    per_task_summary: dict[str, dict] = {}
    print("\n=== TEST per task ===")
    for t_idx, ds in enumerate(DATASETS):
        sel = test_task == t_idx
        if sel.sum() == 0:
            continue
        bt = test_binary[sel]
        rt = routed_idx[sel]
        gr = float(bt[np.arange(sel.sum()), rt].mean())
        ab_t = bt[:, always_best_idx].mean()
        # Per-task best (using train accuracy on that task)
        train_per_task = binary[train_q[task_of_query[train_q] == t_idx]].mean(axis=0)
        per_task_best_idx = int(train_per_task.argmax())
        per_task_best_id = cols_canonical[per_task_best_idx]
        per_task_best_acc = float(bt[:, per_task_best_idx].mean())
        oracle_t = float(bt.max(axis=1).mean())
        print(f"\n  [{ds}]  n_test={sel.sum()}")
        print(f"    oracle:                          {oracle_t:.4f}")
        print(f"    always-best (global, {always_best_id}): {ab_t:.4f}")
        print(f"    per-task best (train-pick {per_task_best_id}): {per_task_best_acc:.4f}")
        print(f"    GraphRouter:                     {gr:.4f}")
        per_task_summary[ds] = {
            "n_test": int(sel.sum()),
            "oracle": oracle_t,
            "always_best_global": float(ab_t),
            "per_task_best": {"id": per_task_best_id, "accuracy": per_task_best_acc},
            "graphrouter": gr,
        }

    # Routed-to histogram (overall).
    hist = {cols_canonical[j]: int((routed_idx == j).sum()) for j in range(n_llm)}
    print("\nrouted-to (GraphRouter):")
    for m_id, c in sorted(hist.items(), key=lambda x: -x[1]):
        if c == 0:
            continue
        print(f"  {m_id:35s} {c:5d}  ({c / len(test_q):.1%})")

    out = {
        "datasets": DATASETS,
        "encoder": args.encoder,
        "n_per_task": n_per_task,
        "n_train": int(len(train_q)),
        "n_dev": int(len(dev_q)),
        "n_test": int(len(test_q)),
        "n_models": int(n_llm),
        "model_ids": cols_canonical,
        "embedding_dim": args.embedding_dim,
        "edge_dim": args.edge_dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "best_dev_routing_acc": float(best_dev_acc),
        "train_time_sec": train_time,
        "test_oracle": oracle_acc,
        "test_random": rand_acc,
        "test_always_best_global": {"id": always_best_id, "accuracy": always_best_acc},
        "test_graphrouter_overall": test_acc,
        "per_task": per_task_summary,
        "routed_counts": hist,
        "training_history": history,
    }
    out_path = REPO_ROOT / "outputs" / "routing" / "graphrouter_multitask.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
