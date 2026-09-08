# Cosine-Similarity Routing on Multi-Hop QA: A Negative Result

Created: 2026-05-02 15:50 EDT

## Question

Can a zero-shot router that compares an embedded test question to embedded
LLM capability descriptions, via cosine similarity, beat the trivial
"always pick the strongest model" baseline?

## Setup

- **Datasets**: morehopqa (1,118 Q), musique (10,354 Q). The matrices in
  `outputs/updated-matrices/` give per-question binary correctness
  (`{0, 1}`) for 10 candidate LLMs that span 0.5B–30B parameters,
  including math specialists (`mathstral-7b`), medical specialists
  (`medgemma-4b-it`), reasoning specialists (`qwen3-4b-thinking-2507`,
  `deepseek-r1-distill-llama-8b`), and a strong generalist
  (`qwen3-30b-a3b`).
- **Encoder**: `BAAI/bge-base-en-v1.5` (768-dim, top-tier MTEB).
- **Capability descriptions**: `eval/configs/model_descriptions.yaml` —
  one keyword-dense paragraph per model, hand-written to mirror the
  vocabulary of the question types each model is known to handle well.
- **Splits**: random by query, seed 0. 80/10/10 (train/dev/test) for
  variants requiring a held-out tuning set; 80/20 (train/test) for
  variants without.
- **Reference baselines** (all on the same test split):
  - Oracle: `max_m correct(q, m)` averaged over test queries.
  - Always-best: model with highest accuracy on the train rows.
  - Random uniform.

All routing scripts live in `eval/routers/`. All result JSONs live in
`outputs/routing/`.

## Methods

### Variant A — Pure cosine (capability descriptions)

For each test query and each model:

```
score(q, m) = cosine(BGE(q), BGE(description[m]))
```

Routed model = `argmax_m score(q, m)`. Fully zero-shot — no labelled
data is consulted, no parameters are trained.

### Variant B — Example-based descriptions

For each model `m`, build a description by concatenating the K=10
training questions on which `m` is *uniquely correct*, ranked by a
specialization score:

```
score(q, m) = correct(q, m) − mean_{m'≠m} correct(q, m')
```

(Take the top-K rows where `correct(q, m) == 1`.) The intuition: rather
than describing what the model *should* be good at, describe it by the
empirical distribution of questions it actually wins. This implicitly
folds performance information into the embedding without inventing new
prose.

### Variant C — Cosine + empirical-strength prior

```
score(q, m) = cosine(q, desc_m) + λ · global_acc(m)
```

`global_acc(m)` is `m`'s accuracy on the train rows. λ is selected on
the dev split. As λ → 0 the router collapses to pure cosine; as λ → ∞
it collapses to "always pick `argmax_m global_acc(m)`". Any λ in
between is the regime where cosine matters.

## Results

### Variant A (pure cosine)

Subtask-level numbers below use the **strict question-level oracle**: a
question counts as correct iff every hop in its chain is answered correctly
(or, for the oracle, has at least one model that answers it correctly). This
matches the metric used in `router_analysis/heterogeneous_accuracy.py`. An
earlier draft of this table reported a per-hop coverage average for the
subtask row, which is not comparable to the full-question metric and
overstated the subtask figures by 30+ points; the numbers here are
corrected.

| dataset | level | router | oracle | always-best | random | gap to always-best |
|---|---|---|---|---|---|---|
| morehopqa | full Q | 49.19% | 86.67% | 70.39% (qwen3-30b) | 37.84% | **−21.2 pt** |
| morehopqa | subtask | 39.98% | 78.00% | 62.79% (qwen3-30b) | 27.46% | **−22.8 pt** |
| musique | full Q | 46.60% | 79.82% | 58.96% (qwen3-30b) | 36.50% | **−12.4 pt** |

Routing histograms reveal the problem: the strongest model
(`qwen3-30b-a3b`) is selected ≤ 1% of the time on full-question routing.
The cosine machinery routes by topical match, but topical match does not
align with empirical strength. At subtask level the same failure mode
holds — only 0.6% of hops route to qwen3-30b — and now compounding
across the chain makes the gap to always-best slightly *wider* rather
than narrower.

### Variant B (example-based descriptions)

| dataset | example router | capability router | always-best | oracle |
|---|---|---|---|---|
| morehopqa | 42.86% | 44.64% | 69.64% | 87.05% |
| musique | 40.46% | 46.60% | 58.96% | 79.82% |

**Example descriptions are *worse* than capability prose.** The
specialization score selects each model's idiosyncratic wins, including
for weak models. So `qwen1.5-0.5b-chat` (4% accuracy) gets characterized
by questions where it specifically beat its peers, and traffic flows to
it. Routing distributes evenly across all 10 models, which is
catastrophic when half are weak.

### Variant C (cosine + strength prior, morehopqa)

Lambda sweep on a held-out dev split:

| λ | dev acc |
|---|---|
| 0.00 | 41.07% (pure cosine — fails) |
| 0.30 | 59.82% |
| 0.70 | 68.75% |
| **1.00** | **70.54%** (selected) |
| 1.50 | 70.54% (plateau) |
| 10.0 | 70.54% (plateau) |

Test:

| method | accuracy |
|---|---|
| oracle | 84.82% |
| always-best (qwen3-30b) | 68.75% |
| **cosine + strength prior (λ\*=1.0)** | **67.86%** |
| pure cosine (λ=0) | 48.21% |
| random | 31.25% |

The dev curve is **monotonically increasing then flat**: there is no λ
where cosine improves over the prior alone. Above λ ≥ 1.5 the router
behaves identically to "always pick qwen3-30b". On test, the chosen
λ\*=1.0 router lands fractionally below always-best (-0.9 pt) because
the 4 of 112 test queries it routes away from qwen3-30b are a net loss.

## Why cosine over descriptions cannot win

Three structural reasons, each independently sufficient:

1. **Descriptions encode topic, not strength.** The decision boundary
   the router needs is "which model is strongest on this question",
   not "what topic is this question about". A capability description
   ("good at math reasoning") cannot express "this model is just
   bigger and smarter than the alternatives." When one model is
   strictly better — e.g., `qwen3-30b-a3b` is +16 pt over the next
   model on morehopqa — no cosine in description-space recovers that.

2. **Question texts cluster weakly.** All MoReHopQA questions share a
   "what is X of Y of Z?" surface form. Cosine in this regime is
   dominated by syntactic similarity, not topical signal. There simply
   isn't enough variation in the embedded questions to separate
   "questions a math specialist would win" from the rest, given the
   homogeneous query distribution.

3. **The strength prior plateau experiment is a hypothesis test.**
   If cosine carried any usable signal independent of global accuracy,
   the dev curve would peak somewhere in the middle (a sweet spot
   where cosine refines the prior). It does not — the curve rises
   monotonically and plateaus at the always-best floor. The optimal
   policy under this objective is to set λ → ∞, which means *ignore
   cosine entirely*.

## Diagnostic: where is the headroom, and why can't cosine reach it?

On morehopqa, oracle (86.67%) − always-best (70.39%) = **+16.3 pt of
specialist headroom**. 16.3% of questions are "specialist
opportunities": qwen3-30b fails but at least one other model
succeeds. Among these:

| specialist | wins / 182 specialist Q's |
|---|---|
| qwen3-4b-thinking-2507 | 47.8% |
| deepseek-r1-distill-llama-8b | 39.0% |
| llama-3.1-8b-instruct | 36.3% |
| phi-4-mini-instruct | 23.6% |
| mathstral-7b | 22.5% |

The headroom is real, but it is not predictable from the question
text alone with description-cosine. We confirmed this with three
independent probes (pure cosine, example-based descriptions, strength
prior) — all converge to the same plateau at "always pick qwen3-30b".

## Implications

- **Stronger encoders will not help.** The bottleneck is the
  formulation, not the embedding quality. Going from `bge-base` to
  `bge-large` or `mxbai-embed-large` would not change the qualitative
  picture; the dev plateau argument is encoder-independent.
- **Better-written descriptions will not help.** Variant B confirms
  that even descriptions built from the model's empirical wins — which
  fold actual performance into the text — fail the same way.
- **For routing on this kind of homogeneous within-dataset benchmark,
  description-cosine is structurally the wrong tool.** The signal it
  needs is not in the descriptions.

## What this rules in for follow-up work

The negative result rules in any method that uses **empirical
performance directly** rather than re-deriving it through a description.
A learned graph-based router (GraphRouter) is evaluated separately in
[graphrouter_routing.md](graphrouter_routing.md).

## Artifacts

- `eval/routers/embedding_router.py` — Variant A
- `eval/routers/example_router.py` — Variant B
- `eval/routers/strength_prior_router.py` — Variant C
- `eval/configs/model_descriptions.yaml` — capability descriptions used
- `outputs/routing/embedding_morehopqa_full.json`
- `outputs/routing/embedding_morehopqa_subtask.json`
- `outputs/routing/example_morehopqa.json`
- `outputs/routing/example_musique.json`
- `outputs/routing/strength_prior_morehopqa.json`
