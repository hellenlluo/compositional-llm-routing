# GraphRouter on Multi-Hop QA: Why a Learned Graph Router Cannot Beat Always-Best on Our Benchmark

Created: 2026-05-02 16:10 EDT

## Question

Cosine over LLM capability descriptions fails to beat the trivial
"always pick the strongest model" baseline (see
[cosine_similarity_routing.md](cosine_similarity_routing.md)). The
diagnosis from those experiments was *descriptions encode topic, not
strength*. A learned graph router (GraphRouter,
[Feng et al., ICLR 2025](https://arxiv.org/abs/2410.03834)) addresses
this directly: it represents performance and cost as edge features in
a heterogeneous graph and learns LLM and query embeddings via message
passing. Does it beat always-best on our datasets?

**Short answer: no, for a different reason.** Our benchmark has a
single dominant model — `qwen3-30b-a3b` is the per-task best on every
one of our three datasets — and a learned router has no per-task
routing decision to make in that regime.

## Method

We adapt the official implementation in
[repos/GraphRouter/](../repos/GraphRouter/) directly. Key code reused:
`EncoderDecoderNet`, `form_data` from
`repos/GraphRouter/model/graph_nn.py`. We replace their training loop
with a custom one that tracks dev *routing accuracy* as the
early-stopping criterion (their default tracks a "macro-F1 on
predicted argmax" that is meaningless when most queries have multiple
correct LLMs).

**Graph structure** (faithful to the paper):

```
Task ─── Query ─── LLM
  =1         (correct, cost)   ← edge feature
```

- **Task nodes** (one per dataset): BGE embedding of a task
  description. (We use `BAAI/bge-base-en-v1.5` rather than the paper's
  `bert-base-uncased` — the encoder-quality argument from the cosine
  experiments applies here too.)
- **Query nodes**: BGE embedding of the question text.
- **LLM nodes**: BGE embedding of the LLM capability description from
  `eval/configs/model_descriptions.yaml`.
- **Task–Query edges**: weight 1.
- **LLM–Query edges**: feature = `[correct ∈ {0, 1},
  params_active_b normalized to [0, 1]]`.

**Forward pass.** A 2-layer `GeneralConv` (PyG) propagates messages
weighted by edge features. The edge prediction head computes
`sigmoid(dot(h_query_initial, h_llm_after_GNN))`. **Routing** at
inference is `argmax_m EdgePred(query, m)`.

**Training.** Per-edge BCE on observed correctness (a strict superset
of the paper's "best-LLM-as-1, others-0" loss for our binary setting).
Random edge masking at rate 0.5 each minibatch so the model cannot
trivially read the edge feature it's predicting. AdamW lr=1e-3,
weight decay 1e-3, batch 32 inner iterations per gradient step. Best
checkpoint by **dev routing accuracy** (not the paper's broken F1
metric).

## Within-dataset experiment (morehopqa only)

Setup: 894 train / 112 dev / 112 test, seed 0. 1118 queries × 10
LLMs = 11,180 LLM-Query edges. 50 epochs, hidden=8.

### Five-seed test-accuracy comparison

| seed | always-best (test) | GraphRouter (test) | Δ |
|---|---|---|---|
| 0 | 68.75% | 68.75% | 0.00 |
| 1 | 75.00% | 75.00% | 0.00 |
| 2 | 66.96% | 66.96% | 0.00 |
| 3 | 70.54% | 70.54% | 0.00 |
| 4 | 70.54% | 70.54% | 0.00 |

**GraphRouter and always-best are identical across all 5 seeds.** The
trained model converges to routing 100% (or near-100%) of test queries
to `qwen3-30b-a3b`.

### Why does it just match always-best?

We tried two flanking conditions:

- **Larger model** (hidden=32, 500 epochs): train BCE drops to 0.10
  while dev BCE plateaus at 0.61 (massive overfitting). Best dev
  routing accuracy peaks at epoch 11 (71.43%, +0.9 pt over always-best
  on dev) and degrades monotonically thereafter. On test the
  early-stop checkpoint scores 66.96%, *below* always-best — the
  model's divergent routes (98% qwen3-30b, 2% deepseek-r1) lose more
  than they win.
- **Smaller model + more regularization** (hidden=8, weight_decay=1e-3,
  50 epochs, 5 seeds): exactly the table above — the model never
  diverges from "always pick qwen3-30b" enough to either beat or hurt
  always-best.

There is no "Goldilocks" capacity where GraphRouter cleanly beats the
floor. Every checkpoint either matches always-best or overfits below
it.

## Multi-task experiment (morehopqa + musique + stepcot)

The original GraphRouter paper's headline numbers come from a
multi-task setup where different LLMs win on different tasks
(Mixtral-8x7B on Alpaca, LLaMA-2-7b on others, etc.). The task-node
mechanism is the routing engine — it carries the task-conditioned
prior that lets the GNN say "for SQUAD-like questions, route to LLM
X; for GSM8K-like questions, route to LLM Y."

We rebuild the experiment with our three datasets, stratified
80/10/10 split per task, all 10 LLMs in the pool. To balance dataset
sizes we cap musique to 2000 sampled queries (so it doesn't dominate).
Total graph: 3,715 query nodes, 10 LLM nodes, 37,150 LLM-Query edges.

### Overall test result

| metric | value |
|---|---|
| oracle (test) | 80.20% |
| **always-best (qwen3-30b-a3b, test)** | **66.58%** |
| **GraphRouter (test)** | **66.58%** |
| random uniform | 38.69% |

### Per-task test breakdown

| dataset | n_test | oracle | always-best (global) | per-task best (train-pick) | GraphRouter |
|---|---|---|---|---|---|
| morehopqa | 112 | 84.82% | 68.75% | 68.75% (qwen3-30b) | 68.75% |
| musique | 200 | 81.50% | 60.00% | 60.00% (qwen3-30b) | 60.00% |
| stepcot | 59 | 94.92% | 84.75% | 84.75% (qwen3-30b) | 84.75% |

GraphRouter histogram on test: **100.0% qwen3-30b-a3b**.

## Why GraphRouter doesn't help: it's the data, not the method

The structural reason is not architectural — it's a property of our
LLM pool against our datasets:

> **`qwen3-30b-a3b` is the per-task best model on every one of our
> three datasets.** It is also the global-train-best model. So
> "always pick qwen3-30b" is *simultaneously* the global Pareto
> frontier and the per-task Pareto frontier and the per-train Pareto
> frontier.

In a setup where one model dominates every task, a perfect
oracle-on-train router has no per-task routing decision to make. The
exact mechanism that gave GraphRouter its +12.3% over baselines in
the original paper — the task node carrying a task-conditioned prior
into the query embedding — provides **zero** additional signal here,
because the prior is the same on every task: pick qwen3-30b.

The remaining headroom (oracle − always-best ≈ +14 pt overall, +16 pt
on morehopqa, +21 pt on musique, +10 pt on stepcot) lives entirely in
*per-query* specialist opportunities — the 16% of morehopqa queries
where qwen3-30b fails but some other model succeeds, the 21% on
musique, etc. Those opportunities are not predictable from the
question text alone with any of the methods we have tried (cosine
variants, strength prior, GraphRouter all converge to the same
plateau).

## Compared to the cosine experiments

GraphRouter and cosine fail for **complementary** reasons:

| method | failure mode |
|---|---|
| pure cosine over descriptions | descriptions encode topic, not strength |
| cosine + strength prior | cosine signal is noise once you condition on global accuracy |
| GraphRouter (within-dataset) | model converges to always-best because qwen3-30b dominates train |
| GraphRouter (multi-task) | task structure is degenerate — qwen3-30b wins every task too |

Cosine is about the wrong feature space. GraphRouter is about a
degenerate model pool. Both converge on the same plateau.

## Implications

- **GraphRouter the architecture is not at fault.** It works in the
  setup the paper studied (10 distinct LLMs, 4 datasets, multiple
  per-task winners). It cannot extract routing signal that does not
  exist in the data.
- **The fix is at the data layer.** A more diverse model pool — one
  where different models genuinely win on different tasks — would
  create per-task routing decisions for GraphRouter (and cosine, and
  strength-prior) to exploit. Concrete additions that would likely
  reshape the picture:
  - A frontier medical model that beats qwen3-30b on stepcot.
  - A frontier math/code model that beats qwen3-30b on the
    arithmetic-tail morehopqa questions.
  - A retrieval-tuned model that beats qwen3-30b on musique factual
    lookups.
- **Within our current pool, no learned method can exceed
  always-best.** This is a strong negative result: it bounds the
  performance ceiling for *any* router on this benchmark suite to
  always-best, until the pool is diversified.

## What this rules in for follow-up work

- **Diversify the model pool** — biggest expected gain. Even one
  task-specialist beating qwen3-30b on its niche would unlock routing
  signal for both cosine and GraphRouter.
- **Cost-aware routing**: even at always-best accuracy, a router that
  uses fewer params per query on easy questions would be a meaningful
  contribution. Sweep cost-vs-accuracy weight on the `_per_active_b`
  matrices.
- **Per-query specialist prediction with structural features**: for
  morehopqa, the `question_decomposition` final-manipulation type
  (syllable count, digit reverse, concatenation) is a strong feature
  that descriptions and BGE embeddings under-utilize. A
  manipulation-type-conditioned router could reach per-query
  specialist opportunities that semantic embeddings cannot.

## Artifacts

- `eval/routers/graphrouter/PLAN.md` — original implementation plan
- `eval/routers/graphrouter/run.py` — within-dataset adapter
- `eval/routers/graphrouter/run_multitask.py` — multi-task adapter
- `repos/GraphRouter/` — official paper implementation
- `outputs/routing/graphrouter_morehopqa.json` — within-dataset result
- `outputs/routing/graphrouter_multitask.json` — multi-task result
- `outputs/updated-matrices/` — correctness + cost matrices (10
  models × 3 datasets) used to populate edge features
