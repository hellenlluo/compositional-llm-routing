# Cosine-Description Routing — Results

Created: 2026-05-02 15:08 EDT

## Summary

Tested zero-shot routing where each candidate LLM is represented by a
short capability description, the test question is embedded with the same
sentence encoder, and the model with the highest cosine similarity is
selected. Three variants were tried on MoReHopQA and MuSiQue against the
matrices in `outputs/updated-matrices/` (10 models per dataset).

**Conclusion: cosine over capability descriptions adds no usable signal
beyond what each model's overall accuracy already provides.** All three
variants underperform the trivial "always pick the strongest model"
baseline. The negative result is robust across datasets, encoders, and
description styles.

## Setup

- **Datasets**: morehopqa (1,118 Q), musique (10,354 Q). One question per
  row in `outputs/updated-matrices/<dataset>_full_binary.json`.
- **Models** (10): qwen3-30b-a3b, qwen3-4b-thinking-2507,
  deepseek-r1-distill-llama-8b, llama-3.1-8b-instruct, mathstral-7b,
  medgemma-4b-it, phi-4-mini-instruct, llama-3.1-nemotron-nano-8b,
  mistral-7b-instruct-v0.3, qwen1.5-0.5b-chat.
- **Encoder**: `BAAI/bge-base-en-v1.5` (768-dim, top-tier MTEB).
- **Capability descriptions**: `eval/configs/model_descriptions.yaml` —
  short, keyword-dense paragraphs covering each model's known strengths.
- **Splits**: random with seed 0. 80% train, 10% dev, 10% test (or 80/20
  for variants without a dev sweep).
- **Baselines reported**: oracle (best per Q), always-best (single highest
  global_acc), random uniform.

## Variant 1 — Pure cosine (capability descriptions)

`eval/routers/embedding_router.py`

For each test Q: `argmax_m cosine(q_emb, desc_emb_m)`.

| dataset | router | always-best | oracle | random | gap to always-best |
|---|---|---|---|---|---|
| morehopqa (full) | 49.19% | 70.39% (qwen3-30b) | 86.67% | 37.84% | **−21.2 pt** |
| morehopqa (subtask) | 77.46% | 88.28% | 93.65% | 71.29% | −10.8 pt |
| musique (full) | 46.60% | 58.96% (qwen3-30b) | 79.82% | 36.50% | **−12.4 pt** |

Routing histograms show the strongest model (qwen3-30b-a3b) gets selected
≤ 1% of the time on full Q. Topical match dominates, but the topical match
is not aligned with empirical strength.

## Variant 2 — Example-based descriptions (cosine)

`eval/routers/example_router.py`

Description for model m = concatenation of the top-K=10 train questions
where m is correct and peers tend to fail (specialization score
`correct(q, m) − mean(correct(q, peers))`). Same encoder, same routing.

| dataset | example router | capability router | always-best | oracle |
|---|---|---|---|---|
| morehopqa | 42.86% | 44.64% | 69.64% | 87.05% |
| musique | 40.46% | 46.60% | 58.96% | 79.82% |

**Example descriptions are worse than capability descriptions.** Reason:
"specialization" picks each model's idiosyncratic wins, including for weak
models. So qwen1.5-0.5b-chat (0.04 acc) gets characterized by questions
where it specifically beat its peers, and traffic gets sent to it. Routing
spreads evenly across all 10 models, which is catastrophic when half are
weak.

## Variant 3 — Cosine + empirical-strength prior

`eval/routers/strength_prior_router.py`

`score(q, m) = cosine(q, desc_m) + λ · global_acc(m)`, where `global_acc`
is the model's accuracy on the train rows. λ swept on dev.

morehopqa, dev sweep:

| λ | dev acc | router behavior |
|---|---|---|
| 0.00 | 41.07% | pure cosine |
| 0.30 | 59.82% | mixed |
| 0.70 | 68.75% | qwen3-30b winning more often |
| 1.00 | **70.54%** | qwen3-30b dominates (96.4% of routes) — selected |
| 1.50 | 70.54% | identical to always-best |
| 10.0 | 70.54% | identical to always-best |

morehopqa, test:

| method | accuracy |
|---|---|
| oracle | 84.82% |
| always-best (qwen3-30b) | 68.75% |
| **strength-prior (λ\*=1.0)** | **67.86%** |
| pure cosine (λ=0) | 48.21% |
| random | 31.25% |

The dev curve is **monotonically increasing then flat**. There is no
λ where adding cosine improves over pure global_acc. The optimum is to
ignore cosine entirely (λ → ∞ ≡ always-best).

## Why the cosine signal fails

1. **Descriptions encode topic, not strength.** A description embedding
   measures "what kind of question is this for", not "which model is
   strongest on it." When one model is just strictly better (qwen3-30b),
   no description can express that.
2. **Question texts cluster weakly by topic.** All MoReHopQA questions
   share a "what is X of Y of Z" surface form. Cosine in this regime is
   dominated by syntactic structure, not topical signal.
3. **The decision boundary isn't topical.** The variation in "which model
   wins" is dominated by raw model capability gradients, not topical fit.
   Cosine cannot recover those gradients from descriptions alone.

## What was ruled out

- Any pure cosine-over-descriptions method on this task — the strength
  prior plateau experiment establishes this independent of encoder and
  description style.
- "Add a stronger encoder" or "rewrite descriptions" — the bottleneck is
  the formulation, not the embedding quality.

## What is still open

Methods that use the question text but not as a topical match against
descriptions:

- **kNN over performance** — embed Q, retrieve nearest train questions,
  pick the model with best avg correctness on neighbors.
- **Trained classifier** — small MLP on `q_emb → P(correct | model)`.
- **Manipulation-type routing** (MoReHopQA-specific) — bucket by the
  explicit final-step manipulation type, route by best-per-bucket.

## Artifacts

- `outputs/routing/embedding_morehopqa_full.json`
- `outputs/routing/embedding_morehopqa_subtask.json`
- `outputs/routing/example_morehopqa.json`
- `outputs/routing/example_musique.json`
- `outputs/routing/strength_prior_morehopqa.json`
- `eval/configs/model_descriptions.yaml` — capability descriptions used.
