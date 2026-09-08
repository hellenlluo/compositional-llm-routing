"""Sample N prompts from EmbedLLM's benchmark pool, stratified proportionally
to the original per-category distribution.

EmbedLLM (RZ412/EmbedLLM on HuggingFace) pools ~35.7k prompts across 80
fine-grained category tags (57 MMLU subjects, 15 GPQA variants, plus ASDiv,
MathQA, MedMCQA, SocialIQA, PIQA, GSM8K, TruthfulQA_MC1, LogiQA).

To build EmbedLLM-style embeddings for our 8 models we need correctness
labels on these prompts. This script samples a subset we'll actually run
inference on, preserving the original per-category proportions via
largest-remainder apportionment (so the global total is exactly N).

Outputs:
  outputs/embedllm/raw/question_order.csv   # cached prompt text (~20MB)
  outputs/embedllm/categories.csv           # prompt_id -> category (cached)
  outputs/embedllm/prompts_subset.jsonl     # the actual sampled subset
  outputs/embedllm/sampling_report.json     # per-category + per-family counts

Run:
  python -m eval.embedllm_baseline.sample_prompts --n 3000
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "embedllm"
RAW_DIR = OUT_DIR / "raw"

HF_BASE = "https://huggingface.co/datasets/RZ412/EmbedLLM/resolve/main"
SPLIT_FILES = ["train.csv", "val.csv", "test.csv"]


def _download(filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    url = f"{HF_BASE}/{filename}"
    print(f"downloading {url} -> {dest}", flush=True)
    subprocess.run(["curl", "-sSL", "-o", str(dest), url], check=True)


def _stream_categories(filename: str, out: dict[str, str]) -> None:
    """Add (prompt_id -> category) pairs for any prompt_ids not yet seen."""
    url = f"{HF_BASE}/{filename}"
    print(f"streaming {filename} for (prompt_id, category) pairs...", flush=True)
    p = subprocess.Popen(["curl", "-sSL", url], stdout=subprocess.PIPE)
    assert p.stdout is not None
    reader = csv.reader(io.TextIOWrapper(p.stdout, encoding="utf-8", newline=""))
    header = next(reader)
    pi, ci = header.index("prompt_id"), header.index("category")
    added = 0
    for row in reader:
        pid = row[pi]
        if pid in out:
            continue
        out[pid] = row[ci]
        added += 1
    p.wait()
    print(f"  + {added} new prompts (total so far: {len(out)})", flush=True)


def _build_category_map(cache: Path) -> dict[str, str]:
    """Return {prompt_id: category}, caching to CSV so re-runs are instant."""
    if cache.exists():
        print(f"loading cached {cache}")
        with cache.open() as f:
            return {row["prompt_id"]: row["category"] for row in csv.DictReader(f)}
    cat_map: dict[str, str] = {}
    for f in SPLIT_FILES:
        _stream_categories(f, cat_map)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "category"])
        for pid, cat in sorted(cat_map.items(), key=lambda x: int(x[0])):
            w.writerow([pid, cat])
    return cat_map


def _load_questions(question_csv: Path) -> dict[str, str]:
    with question_csv.open() as f:
        return {row["prompt_id"]: row["prompt"] for row in csv.DictReader(f)}


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Hamilton / largest-remainder apportionment summing exactly to `total`."""
    grand = sum(counts.values())
    raw = {c: total * n / grand for c, n in counts.items()}
    alloc = {c: int(v) for c, v in raw.items()}
    remainder = total - sum(alloc.values())
    if remainder:
        order = sorted(raw.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))
        for c, _ in order[:remainder]:
            alloc[c] += 1
    return alloc


def _family(category: str) -> str:
    if category.startswith("mmlu_"):
        return "MMLU"
    if category.startswith("gpqa_"):
        return "GPQA"
    return {
        "medmcqa": "MedMCQA",
        "mathqa": "MathQA",
        "asdiv": "ASDiv",
        "social_iqa": "SocialIQA",
        "piqa": "PIQA",
        "gsm8k": "GSM8K",
        "truthfulqa_mc1": "TruthfulQA",
        "logiqa": "LogiQA",
    }.get(category, category)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000,
                    help="Total prompts to sample across all categories.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_DIR / "prompts_subset.jsonl"))
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    question_csv = RAW_DIR / "question_order.csv"
    categories_csv = OUT_DIR / "categories.csv"

    _download("question_order.csv", question_csv)
    cat_map = _build_category_map(categories_csv)
    prompts = _load_questions(question_csv)

    missing = [pid for pid in cat_map if pid not in prompts]
    if missing:
        print(f"warning: {len(missing)} prompt_ids have category but no prompt text; dropping")
        for pid in missing:
            cat_map.pop(pid, None)

    category_counts = Counter(cat_map.values())
    print(f"\n{len(cat_map)} prompts across {len(category_counts)} categories; sampling N={args.n}")

    alloc = largest_remainder(dict(category_counts), args.n)
    assert sum(alloc.values()) == args.n, (sum(alloc.values()), args.n)

    by_cat: dict[str, list[str]] = defaultdict(list)
    for pid, cat in cat_map.items():
        by_cat[cat].append(pid)

    rng = random.Random(args.seed)
    sampled: list[dict] = []
    for cat, k in alloc.items():
        pool = sorted(by_cat[cat], key=lambda x: int(x))
        k = min(k, len(pool))
        chosen = rng.sample(pool, k)
        sampled.extend(
            {"prompt_id": int(pid), "category": cat, "family": _family(cat), "prompt": prompts[pid]}
            for pid in chosen
        )

    sampled.sort(key=lambda r: r["prompt_id"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    family_alloc: Counter = Counter()
    for cat, k in alloc.items():
        family_alloc[_family(cat)] += k

    report = {
        "n_total_source_prompts": len(cat_map),
        "n_categories": len(category_counts),
        "n_sampled": len(sampled),
        "seed": args.seed,
        "per_family": dict(family_alloc),
        "per_category": {c: {"source": category_counts[c], "sampled": alloc[c]} for c in sorted(alloc)},
    }
    (OUT_DIR / "sampling_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nwrote {len(sampled)} prompts to {out_path}")
    print("\nPer-family allocation:")
    for fam, k in sorted(family_alloc.items(), key=lambda x: -x[1]):
        pct = 100 * k / args.n
        print(f"  {fam:12s} {k:5d}  ({pct:4.1f}%)")


if __name__ == "__main__":
    main()
