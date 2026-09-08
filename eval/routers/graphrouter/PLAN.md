# GraphRouter Implementation Plan — within-dataset MoReHopQA

Created: 2026-05-02 15:22 EDT

## Goal

Implement a faithful adaptation of GraphRouter (Feng et al., ICLR 2025) on
the morehopqa correctness matrix. Treat it as **within-dataset routing**
for now (single task node). Multi-task (morehopqa + musique + stepcot)
extension is planned but out of scope for this pass.

The headline question we want to answer:

> Can a 2-layer heterogeneous GNN that consumes (correctness, cost) edge
> features beat the "always pick qwen3-30b-a3b" baseline on morehopqa,
> where pure cosine over descriptions cannot?

## Inputs (already in repo)

| input | source | shape |
|---|---|---|
| binary correctness | `outputs/updated-matrices/morehopqa_full_binary.json` | 1118 × 10 |
| per-active-b cost-adjusted matrix | `outputs/updated-matrices/morehopqa_full_per_active_b.json` | 1118 × 10 |
| model params | `outputs/updated-matrices/model_params.json` | 10 |
| question text | `data_preprocessing/cleaned_trimmed_data/morehopqa_cleaned.json` | 1118 |
| model descriptions | `eval/configs/model_descriptions.yaml` | 10 |
| sentence encoder | `BAAI/bge-base-en-v1.5` (already cached) | 768-d |

No new data collection or labeling needed.

## Directory layout

```
eval/routers/graphrouter/
├── PLAN.md                  ← this file
├── __init__.py
├── build_graph.py           ← step 1: encode + materialize HeteroData
├── model.py                 ← step 2: heterogeneous GAT
├── train.py                 ← step 3: training loop with edge masking
└── evaluate.py              ← step 4: routing eval against baselines
outputs/routing/
└── graphrouter_morehopqa.json
```

## Dependencies to install

- `torch_geometric` (PyG) — for `HeteroData`, `HeteroConv`, custom message passing
- `numpy`, `torch`, `sentence_transformers`, `pyyaml` — already installed

PyG should pip install cleanly on macOS arm64 with PyTorch 2.8. No CUDA
required; CPU is enough for this size graph (<12k edges).

## Step 1 — build_graph.py

Goal: produce a `HeteroData` object plus train/dev/test masks.

1. Load binary matrix (1118 × 10) and questions (1118 strings) and model
   descriptions (10 strings).
2. Write a single task description for morehopqa (1 string).
3. Encode all texts with `bge-base-en-v1.5` (BERT-style mean pool). This
   is the "frozen encoder" stage. Result:
   - task_x:    (1, 768)
   - query_x:   (1118, 768)
   - llm_x:     (10, 768)
4. Build edges:
   - `(task, has_query, query)` — index pairs (0, q) for q in 0..1117
   - `(query, in_task, task)` — reverse
   - `(llm, answered, query)` — index pairs (m, q) for all 11_180 cells
   - `(query, answered_by, llm)` — reverse
5. Edge features for `(llm, answered, query)`:
   - feature_dim = 2: `[correct ∈ {0, 1}, cost = params_active_b]`
   - normalize cost to [0, 1] via division by max param count
6. Random 80/10/10 split BY QUERY (seed=0). Save as boolean masks of
   length 1118.
7. Save `morehopqa_graph.pt` with the HeteroData object plus the mask
   tensors and the original `cols` list (for reading back later).

Smoke test: print node/edge shapes, confirm `data.validate()` passes.

## Step 2 — model.py

Goal: a 2-layer heterogeneous GNN that produces per-(query, LLM) edge
predictions.

Architecture choices, mirroring the paper but compatible with PyG:

```
class HeteroGraphRouter(nn.Module):
    proj_q = Linear(768, 32)
    proj_t = Linear(768, 32)
    proj_m = Linear(768, 32)
    proj_edge = Linear(2, 32)   # project (correct, cost) → message-dim modulation

    layer1: HeteroConv({
        ('task', 'has_query', 'query'): GATConv(32, 32),
        ('llm',  'answered',  'query'): EdgeWeightedConv(32, 32, edge_dim=32),
        ('query','answered_by','llm'):  EdgeWeightedConv(32, 32, edge_dim=32),
    })
    layer2: same structure, 32 → 32

    # task-query combined embedding
    qt_mlp = MLP(32 + 32 → 32)

    # edge prediction head
    def edge_pred(h_qt, h_m): return (h_qt @ h_m.T) / sqrt(32)  # [n_q, n_m]
```

`EdgeWeightedConv` is a custom MessagePassing layer where the message is
`Linear(edge_attr) ⊙ ReLU(W · x_neighbor)` — element-wise modulation by
the projected edge feature, followed by mean aggregation. This realizes
the paper's `w_mq^T · W · h_q` term with stable broadcasting.

Forward pass:
```
h = {'task': proj_t(task_x), 'query': proj_q(query_x), 'llm': proj_m(llm_x)}
edge_attr = proj_edge(raw_edge_attr)                    # (E, 32)
for layer in [layer1, layer2]:
    h = layer(h, edge_index_dict, edge_attr_dict)
    h = {k: F.relu(v) for k, v in h.items()}

h_t_per_query = h['task'][task_idx_per_query]            # (1118, 32)
h_qt = qt_mlp(torch.cat([h_t_per_query, h['query']], 1)) # (1118, 32)
logits = h_qt @ h['llm'].T                              # (1118, 10)
return logits
```

## Step 3 — train.py

Loss: **per-edge BCE** on `correct ∈ {0, 1}` rather than the paper's
"best-LLM-is-1, others-0" cross-entropy. Reasons:
- The matrix has 4–8 correct LLMs per question on morehopqa — collapsing
  to one positive class throws away most of the signal.
- BCE keeps the routing task as "predict correctness" which is closer
  to what we want at inference (rank LLMs by P(correct)).
- The argmax routing decision is unchanged.

Training loop:

```
for epoch in range(NUM_EPOCHS):
    # 1. Mask all dev+test edges so the GNN cannot see those rows.
    masked_edge_index, masked_edge_attr = drop_edges(query_split == "train")
    # 2. Forward pass through the masked graph.
    logits = model(data, masked_edge_index, masked_edge_attr)  # (1118, 10)
    # 3. BCE loss only on training queries.
    loss = BCE(logits[train_mask], binary[train_mask])
    loss.backward()
    optimizer.step()
    # 4. Each epoch end: eval routing accuracy on dev split.
```

Hyperparameters (matching paper where applicable):
- Adam, lr 1e-3 with cosine decay over 200 epochs
- Hidden dim 32, 2 GNN layers
- Batch: full graph at once (it's small)
- Dropout 0.1 on hidden states
- Early stop on dev routing accuracy with patience 30

Save the best checkpoint (by dev acc) to
`outputs/routing/graphrouter_morehopqa_best.pt`.

## Step 4 — evaluate.py

On the test split:

```
logits = best_model(data, full_edge_index_with_TEST_EDGES_MASKED, ...)
routed_idx = logits[test_mask].argmax(dim=1)             # (n_test,)
routed_correct = binary[test_mask, routed_idx]           # (n_test,)
acc = routed_correct.mean()
```

Important: at inference we still mask the test query's own LLM-edges so
the model can't peek. (The paper does this implicitly by the data split.)

Report against the baselines we already have (use the same test split
seed=0 so numbers are comparable to `strength_prior_morehopqa.json`):

- oracle (best-of-models per Q)
- always-best-train (qwen3-30b-a3b based on train accuracy)
- random uniform
- pure cosine (λ=0)
- cosine + strength prior (λ\*=1.0)
- **GraphRouter**

Plus diagnostics:
- routing histogram (which LLMs get picked, %)
- per-LLM precision when picked
- accuracy by question difficulty bucket (easy = many models correct,
  hard = few models correct)

Save a single JSON to
`outputs/routing/graphrouter_morehopqa.json` with the same schema as
`strength_prior_morehopqa.json` for easy comparison.

## Decision points before I start coding

I'll use these defaults unless you say otherwise:

1. **Loss**: BCE per edge (vs paper's CE on best-LLM-as-1). I think BCE
   is the right call here given binary correctness; the paper's loss
   makes more sense when rewards are continuous.
2. **Encoder**: keep `bge-base-en-v1.5` (not BERT). Strictly stronger,
   compatible dim, already cached.
3. **Cost feature on edges**: `params_active_b` (not total). Active
   params is the right cost signal for MoE models like qwen3-30b-a3b.
4. **Single task node** (morehopqa only). If GraphRouter beats the floor
   here, we extend to all-three-datasets in a follow-up pass.
5. **Hidden dim 32, 2 layers** — paper's optimum, no reason to deviate
   on a smaller graph.

## Expected timeline

- Step 1 (build_graph): 30 min
- Step 2 (model): 45 min — the custom edge-weighted conv is the
  trickiest piece
- Step 3 (train): 30 min wiring + 5 min training (CPU)
- Step 4 (evaluate): 20 min
- Buffer for debugging: 30 min

Total: roughly 2.5 hours of focused work. Most of it goes to Step 2
because the message-passing layer needs to handle directed heterogeneous
edges with edge features correctly.

## Risks / things that could fail

1. **PyG install issues on Mac arm64**. Mitigation: PyG 2.4+ has
   wheels for arm64 and works on CPU without torch-scatter when not
   using sparse SSP message-passing. Should be fine.
2. **Loss doesn't move**. Most likely cause: edge masking is wrong and
   the model trivially memorizes the edge feature it's predicting.
   Mitigation: verify by training without edge masking and checking that
   train-set accuracy collapses to "predict correct edge attr from edge
   attr" (sanity check), then re-enable masking.
3. **Result still ties always-best**. Possible if morehopqa has weak
   per-query specialist signal. Would mean the dataset itself is
   dominated by one strong model. Diagnostic: check oracle gap (87% vs
   70%) — there's clearly headroom, so this shouldn't happen, but if it
   does, scaling out to multi-task with shared task nodes should help.
4. **Overfitting**. With only 894 train queries × 10 LLMs = 8940 edges
   and ~50k params, overfitting is plausible. Mitigation: dropout,
   early stop on dev, and the small hidden dim (32) the paper validated.

## Comparison to EmbedLLM (sanity baseline)

If GraphRouter clears always-best, also run EmbedLLM MF
(`repos/EmbedLLM/algorithm/mf.py`) on the same split to confirm both
methods of "learn LLM embeddings from data" beat the description-cosine
ceiling. This is a smaller code lift (~30 min) and provides a second
data point that the structural fix is "use empirical signal directly"
not "use a graph specifically."

## Out of scope for this pass

- Multi-task (3-task) version with shared LLM and task nodes
- New-LLM zero-shot setting (no held-out LLMs in our 10-model pool yet)
- Cost-aware reward sweep (Performance-First / Balance / Cost-First) —
  could be added by varying the BCE label or by using a continuous
  reward target instead. Cleanest as a follow-up.
- Stronger encoders (mxbai, bge-large) — only if results need it.

## Acceptance criteria

GraphRouter pass is **successful** if it beats always-best on the test
split. **Strongly successful** if it beats always-best by ≥ 3 points
and the routing histogram shows non-trivial diversification (i.e., it
sometimes picks specialists when warranted).
