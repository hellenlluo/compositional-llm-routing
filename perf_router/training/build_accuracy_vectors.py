#!/usr/bin/env python3
"""Build per-model accuracy vectors from all available benchmark results.

Sources (in order of precedence):
  1. embedllm-eval/results/{model}/summary.json  -- per-category accuracy across
     MMLU (57 subjects), GPQA (diamond/extended/main × 5 formats), GSM8K, PIQA,
     social_iqa, medmcqa, logiqa, mathqa, asdiv, and TruthfulQA variants.
     Handles two summary.json formats:
       - Old (per-model file):  {"accuracy_by_category": {cat: {"correct": N, "total": N, "accuracy": F}}}
       - New (combined file):   {model_id: {"by_category": {cat: {"n_correct": N, "n_total": N, "accuracy": F}}}}
  2. outputs/full_accuracy_table.json            -- morehopqa/musique/stepcot
     overall accuracy from the efficiency-router evaluation.
  3. perf_router/results/logicbench/{model}/{task}/{logic}/{axiom}/
       results_judged.json  (preferred, written by judge_logicbench.py)
       results.json         (fallback, regex judge)
     -- per-axiom accuracy for each BQA/MCQA × logic_type × axiom combination.

Output (perf_router/accuracy_vectors.json):
  {
    "feature_names": [str, ...],           -- ordered feature names
    "vectors": {model_id: [float, ...]},   -- one entry per model, missing → 0.0
    "meta": {feature: {"n_models": int, "mean_samples": float}}
  }
"""

import json
import os
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR     = os.path.dirname(SCRIPT_DIR)           # perf_router/
PROJECT_DIR  = os.path.dirname(EVAL_DIR)             # project root

EMBEDLLM_RESULTS  = os.path.join(PROJECT_DIR, "embedllm-eval", "results")
EFF_ACC_TABLE     = os.path.join(PROJECT_DIR, "outputs", "full_accuracy_table.json")
LB_RESULTS_DIR    = os.path.join(EVAL_DIR, "results", "logicbench")
LB_JUDGED_PATH    = os.path.join(EVAL_DIR, "results", "qwen3", "lb_judged_accuracies.json")
OUT_PATH          = os.path.join(EVAL_DIR, "results", "qwen3", "accuracy_vectors.json")

# Skip embedllm categories with fewer than MIN_SAMPLES across all models.
MIN_SAMPLES = 5


# ---------------------------------------------------------------------------
# Summary.json parser — handles both old and new formats
# ---------------------------------------------------------------------------

def _parse_summary(summary: dict, model_id: str) -> dict[str, tuple[float, int]]:
    """Return {category: (accuracy, n_total)} for one model from a summary dict.

    Handles:
      Old format  — top-level keys include "accuracy_by_category"
                    values: {"correct": N, "total": N, "accuracy": F}
      New format  — top-level keys are model names; entry has "by_category"
                    values: {"n_correct": N, "n_total": N, "accuracy": F}
    """
    # New combined format: model names are top-level keys
    if model_id in summary:
        entry  = summary[model_id]
        by_cat = entry.get("by_category", {})
        return {
            cat: (info["accuracy"], info.get("n_total", info.get("total", 0)))
            for cat, info in by_cat.items()
        }

    # Old per-model format
    by_cat = summary.get("accuracy_by_category", {})
    return {
        cat: (info["accuracy"], info.get("total", info.get("n_total", 0)))
        for cat, info in by_cat.items()
    }


# ---------------------------------------------------------------------------
# Source 1: embedllm-eval per-category accuracy
# ---------------------------------------------------------------------------

def load_embedllm_categories() -> dict[str, dict[str, float]]:
    """Return {model_id: {category: accuracy}} from summary.json files."""
    result: dict[str, dict[str, float]] = {}
    if not os.path.isdir(EMBEDLLM_RESULTS):
        print(f"  [warn] embedllm results dir not found: {EMBEDLLM_RESULTS}")
        return result
    for model in sorted(os.listdir(EMBEDLLM_RESULTS)):
        path = os.path.join(EMBEDLLM_RESULTS, model, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            summary = json.load(f)
        parsed = _parse_summary(summary, model)
        result[model] = {cat: acc for cat, (acc, _) in parsed.items()}
    return result


def filter_embedllm_categories(
    embedllm: dict[str, dict[str, float]]
) -> tuple[list[str], dict[str, dict]]:
    """Return (sorted list of valid feature names, meta dict).

    A category is valid if at least one model has >= MIN_SAMPLES observations.
    """
    cat_samples: dict[str, list[int]] = defaultdict(list)
    for model in sorted(os.listdir(EMBEDLLM_RESULTS)):
        path = os.path.join(EMBEDLLM_RESULTS, model, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            summary = json.load(f)
        parsed = _parse_summary(summary, model)
        for cat, (_, n_total) in parsed.items():
            cat_samples[cat].append(n_total)

    valid: list[str] = []
    meta: dict[str, dict] = {}
    for cat, samples in cat_samples.items():
        max_samples = max(samples) if samples else 0
        if max_samples < MIN_SAMPLES:
            continue
        feat = f"cat_{cat}"
        valid.append(feat)
        meta[feat] = {
            "n_models":    len(samples),
            "mean_samples": round(sum(samples) / len(samples), 1) if samples else 0,
        }
    return sorted(valid), meta


# ---------------------------------------------------------------------------
# Source 2: efficiency-router dataset accuracy
# ---------------------------------------------------------------------------

def load_efficiency_accuracies() -> dict[str, dict[str, float]]:
    """Return {model_id: {feature_name: accuracy}} from full_accuracy_table.json."""
    result: dict[str, dict[str, float]] = {}
    if not os.path.isfile(EFF_ACC_TABLE):
        print(f"  [warn] efficiency accuracy table not found: {EFF_ACC_TABLE}")
        return result
    with open(EFF_ACC_TABLE) as f:
        records = json.load(f)
    for rec in records:
        model = rec["model_id"]
        feat  = f"eff_{rec['dataset']}"
        result.setdefault(model, {})[feat] = rec["accuracy"]
    return result


def get_efficiency_features(eff: dict[str, dict[str, float]]) -> list[str]:
    feats: set[str] = set()
    for model_feats in eff.values():
        feats.update(model_feats.keys())
    return sorted(feats)


# ---------------------------------------------------------------------------
# Source 3: LogicBench per-axiom accuracy
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _judge_bqa(answer: str, response: str) -> bool:
    """BQA: check for yes/no answer.

    1. Strip <think> tags, then check for direct prefix match.
    2. For verbose models: scan the last 300 chars for the answer as a
       standalone word, ensuring no opposite answer follows it.
    """
    ans  = answer.strip().lower()
    resp = _strip_think_tags(response).lower()

    if resp.startswith(ans):
        return True

    tail = resp[-300:]
    matches = list(re.finditer(rf"\b{re.escape(ans)}\b", tail))
    if matches:
        opposite   = "no" if ans == "yes" else "yes"
        last_ans   = matches[-1].start()
        opp_match  = list(re.finditer(rf"\b{re.escape(opposite)}\b", tail))
        last_opp   = opp_match[-1].start() if opp_match else -1
        return last_ans > last_opp

    return False


def _judge_mcqa(answer: str, response: str) -> bool:
    """MCQA: check for correct choice number.

    LogicBench MCQA uses "choice_N" as the answer (e.g. "choice_3") and the
    model response is the choice number optionally in parentheses (e.g. "(3)"
    or " 3").  Extract the number from both and compare.
    """
    # Extract expected choice number from "choice_N"
    ans_match = re.search(r"(\d+)", answer)
    if not ans_match:
        return False
    expected = ans_match.group(1)

    resp = _strip_think_tags(response).strip()
    # Find the first digit (or digit in parens) in the response
    resp_match = re.search(r"\(?(\d+)\)?", resp)
    if not resp_match:
        return False
    return resp_match.group(1) == expected


def logicbench_accuracy(results_path: str, task_type: str) -> float | None:
    """Compute accuracy from a LogicBench results.json using the regex judge.

    Handles two sample structures:
      BQA  -- sample contains "qa_pairs": [{question, answer, response}, ...]
      MCQA -- sample contains flat fields: {question, choices, answer, response}

    For verbose reasoning models, prefer lb_judged_accuracies.json produced by
    judge_logicbench_bedrock.py (handled in load_logicbench_accuracies).
    """
    try:
        with open(results_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    judge = _judge_bqa if task_type == "BQA" else _judge_mcqa
    correct = total = 0
    for sample in data.get("samples", []):
        if task_type == "BQA":
            # BQA: multiple qa_pairs per sample
            for qa in sample.get("qa_pairs", []):
                answer   = qa.get("answer", "")
                response = qa.get("response", "")
                if not (answer and response):
                    continue
                correct += int(judge(answer, response))
                total   += 1
        else:
            # MCQA: one answer/response per sample (flat structure)
            answer   = sample.get("answer", "")
            response = sample.get("response", "")
            if not (answer and response):
                continue
            correct += int(judge(answer, response))
            total   += 1
    return correct / total if total > 0 else None


def load_logicbench_accuracies() -> dict[str, dict[str, float]]:
    """Return {model_id: {feature_name: accuracy}} from LogicBench result files.

    Features are aggregated at the (task_type, logic_type) level by averaging
    accuracy across all axioms within each group.  This gives coarser but more
    robust features: lb_BQA_first_order_logic, lb_BQA_nm_logic, etc.

    LLM-judged accuracies from lb_judged_accuracies.json take precedence over
    the regex judge for individual axiom scores before aggregation.
    """
    from collections import defaultdict

    # Load pre-judged per-axiom accuracies
    judged: dict[str, dict[str, float]] = {}
    if os.path.isfile(LB_JUDGED_PATH):
        with open(LB_JUDGED_PATH) as f:
            judged = json.load(f)
        print(f"  Using LLM-judged accuracies for: {sorted(judged.keys())}")

    if not os.path.isdir(LB_RESULTS_DIR):
        print(f"  [warn] LogicBench results dir not found: {LB_RESULTS_DIR}")
        return {}

    # Accumulate per-axiom scores, then average within each (task, logic) group
    # raw[model][feat_coarse] = list of per-axiom accuracy values
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for model in sorted(os.listdir(LB_RESULTS_DIR)):
        model_dir = os.path.join(LB_RESULTS_DIR, model)
        if not os.path.isdir(model_dir):
            continue
        for task_type in sorted(os.listdir(model_dir)):
            task_dir = os.path.join(model_dir, task_type)
            if not os.path.isdir(task_dir):
                continue
            for logic_type in sorted(os.listdir(task_dir)):
                logic_dir = os.path.join(task_dir, logic_type)
                if not os.path.isdir(logic_dir):
                    continue
                feat_coarse = f"lb_{task_type}_{logic_type}"
                for axiom in sorted(os.listdir(logic_dir)):
                    res_path = os.path.join(logic_dir, axiom, "results.json")
                    if not os.path.isfile(res_path):
                        continue
                    feat_fine   = f"lb_{task_type}_{logic_type}_{axiom}"
                    feat_coarse = f"lb_{task_type}_{logic_type}"
                    # Prefer LLM-judged value (keyed at fine or coarse level)
                    if model in judged and feat_fine in judged[model]:
                        acc = judged[model][feat_fine]
                    else:
                        acc = logicbench_accuracy(res_path, task_type)
                    if acc is not None:
                        raw[model][feat_coarse].append(acc)

    # Average within each group
    result: dict[str, dict[str, float]] = {}
    for model, groups in raw.items():
        result[model] = {
            feat: sum(vals) / len(vals)
            for feat, vals in groups.items()
            if vals
        }
    return result


def get_logicbench_features(lb: dict[str, dict[str, float]]) -> list[str]:
    feats: set[str] = set()
    for model_feats in lb.values():
        feats.update(model_feats.keys())
    return sorted(feats)


# ---------------------------------------------------------------------------
# Assemble final vectors
# ---------------------------------------------------------------------------

def build_vectors(
    feature_names: list[str],
    all_sources: list[dict[str, dict[str, float]]],
) -> dict[str, list[float]]:
    """Merge sources and emit one float vector per model.

    Missing values are filled with 0.0 (conservative: assume the model
    performs at chance / below average if we have no data).
    """
    all_models: set[str] = set()
    for src in all_sources:
        all_models.update(src.keys())

    merged: dict[str, dict[str, float]] = {m: {} for m in all_models}
    for src in all_sources:
        for model, feats in src.items():
            merged[model].update(feats)

    vectors: dict[str, list[float]] = {}
    for model in sorted(all_models):
        vectors[model] = [merged[model].get(f, 0.0) for f in feature_names]
    return vectors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Building accuracy vectors ===\n")

    # --- Source 1: EmbedLLM categories ---
    print("Loading embedllm-eval per-category summaries...")
    embedllm = load_embedllm_categories()
    print(f"  Found {len(embedllm)} models")
    valid_cat_names, cat_meta = filter_embedllm_categories(embedllm)
    print(f"  {len(valid_cat_names)} categories with >= {MIN_SAMPLES} samples")
    embedllm_mapped: dict[str, dict[str, float]] = {}
    for model, cats in embedllm.items():
        embedllm_mapped[model] = {
            f"cat_{cat}": acc for cat, acc in cats.items()
            if f"cat_{cat}" in valid_cat_names
        }

    # NOTE: eff_* features (morehopqa/musique/stepcot accuracies) are intentionally
    # excluded — those datasets are the held-out test set for router evaluation.
    # Including them would leak test-set information into the model embeddings.

    # --- Source 2: LogicBench ---
    print("\nLoading LogicBench per-axiom results...")
    lb = load_logicbench_accuracies()
    lb_feats = get_logicbench_features(lb)
    n_judged = sum(
        1 for m in lb
        for feat_path in [
            os.path.join(LB_RESULTS_DIR, m)
        ]
        if any(
            os.path.isfile(os.path.join(feat_path, tt, lt, ax, "results_judged.json"))
            for tt in (os.listdir(feat_path) if os.path.isdir(feat_path) else [])
            for lt in (os.listdir(os.path.join(feat_path, tt)) if os.path.isdir(os.path.join(feat_path, tt)) else [])
            for ax in (os.listdir(os.path.join(feat_path, tt, lt)) if os.path.isdir(os.path.join(feat_path, tt, lt)) else [])
        )
    )
    print(f"  {len(lb)} models with LogicBench data, {len(lb_feats)} features "
          f"({n_judged} model(s) using LLM-judged results)")

    # --- Assemble feature list (cat + lb only; eff excluded as test-set leakage) ---
    feature_names = valid_cat_names + lb_feats
    print(f"\nTotal features: {len(feature_names)} "
          f"(cat={len(valid_cat_names)}, lb={len(lb_feats)})")

    all_meta = dict(cat_meta)
    for f in lb_feats:
        n = sum(1 for m in lb if f in lb[m])
        all_meta[f] = {"n_models": n, "mean_samples": -1}

    # --- Build vectors ---
    vectors = build_vectors(feature_names, [embedllm_mapped, lb])
    print(f"\nAccuracy vectors built for {len(vectors)} models:")
    for model, vec in vectors.items():
        n_nonzero = sum(1 for v in vec if v > 0)
        print(f"  {model}: {n_nonzero}/{len(vec)} non-zero features")

    # --- Save ---
    output = {
        "feature_names": feature_names,
        "vectors":       vectors,
        "meta":          all_meta,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
