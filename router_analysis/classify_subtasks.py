"""
Classify every subtask in morehopqa + musique using Claude Haiku via AWS
Bedrock, then write the labels back into each subtask entry in the cleaned
dataset files.

Each call returns a JSON object with three fields:

  category          — RETRIEVAL | COMPUTATION | REASONING
                      (top-level kind, kept for backward compatibility with
                       earlier `label` field consumers)
  retrieval_subtype — only when category == RETRIEVAL:
                      FACT_LOOKUP | GENEALOGY | TEMPORAL_FACT | NUMERIC_FACT
                      (NONE for non-RETRIEVAL subtasks)
  domain            — MEDICAL | GEOGRAPHIC | HISTORICAL | ENTERTAINMENT |
                      SPORTS | SCIENCE | BIOGRAPHICAL | OTHER
                      (subject-matter axis; the only label that can plausibly
                       activate a domain specialist like medgemma or
                       mathstral when routing)

Definitions:
  RETRIEVAL   — answer is a fact looked up from world knowledge or context.
                Sub-divided into:
                  FACT_LOOKUP    — generic single-hop entity property
                                   (director, author, capital, country)
                  GENEALOGY      — family relations (mother, father, spouse,
                                   grandparent, sibling, child)
                  TEMPORAL_FACT  — birth / death / founding / release / end /
                                   establishment dates
                  NUMERIC_FACT   — answer is a specific number from memory or
                                   the passage (population, area, count)
  COMPUTATION — answer is the result of a deterministic procedure (letter
                counting, arithmetic, string reversal, ASCII/binary encoding,
                date arithmetic, timezone conversion). Note: COMPUTATION
                sub-types (PARSE, STRING_OP, ARITHMETIC, DATE_OP, ENCODING,
                STRING_OP_PHONETIC) are populated DETERMINISTICALLY by
                router_analysis/add_deterministic_axes.py from the question
                text — Haiku is not asked for them.
  REASONING   — logical inference, comparison, or multi-step deduction that
                is neither a direct fact lookup nor a fixed algorithm.

After the script completes, each subtask entry will carry:

  {
    "question": "...", "answer": "...",
    "label": "RETRIEVAL",            # back-compat alias of `category`
    "category": "RETRIEVAL",
    "retrieval_subtype": "GENEALOGY",
    "domain": "BIOGRAPHICAL"
  }

Re-running the script overwrites all four fields from scratch.

Usage:
  python3 classify_subtasks.py                       # all datasets
  python3 classify_subtasks.py --dataset morehopqa --workers 20
  python3 classify_subtasks.py --dry-run --dataset morehopqa

Credentials:
  Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION in your
  environment (or use a configured AWS profile). Claude Haiku must be enabled
  for your Bedrock account in the target region.

Dependencies:
  pip install boto3 tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data"

CLEANED_PATHS = {
    "morehopqa": DATA_DIR / "morehopqa_cleaned.json",
    "musique":   DATA_DIR / "musique_cleaned.jsonl",
}

BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

CLASSIFY_PROMPT = """\
You will classify a single multi-hop QA subtask along three axes and return a
JSON object with exactly these three fields: category, retrieval_subtype, domain.

AXIS 1 — category (one of: RETRIEVAL, COMPUTATION, REASONING)

  RETRIEVAL   = The answer is a fact looked up from world knowledge or
                provided context (name, place, date, title, country, count
                from memory, etc.).
  COMPUTATION = The answer is produced by executing a fixed deterministic
                procedure on a known input (counting characters or
                syllables, arithmetic, string reversal, ASCII/binary
                encoding, date arithmetic, timezone conversion, etc.).
  REASONING   = The answer requires logical inference, comparison, or
                multi-step deduction that is neither a direct fact lookup
                nor a fixed algorithm.

AXIS 2 — retrieval_subtype (only when category == RETRIEVAL; otherwise NONE)

  GENEALOGY    = Family/kin relation: mother, father, parent, son, daughter,
                 brother, sister, sibling, spouse, husband, wife,
                 grandparent, grandchild, uncle, aunt, cousin, child.
  FACT_LOOKUP  = Any other RETRIEVAL — generic entity property (director,
                 author, capital, country, performer, located-in, dates of
                 birth/death/founding, populations/counts, etc.). Use this
                 as the default for non-genealogy retrieval.

AXIS 3 — domain (always one of)

  GEOGRAPHIC    = Places, cities, countries, regions, administrative
                  entities, bodies of water, time zones, borders.
  ENTERTAINMENT = Films, TV, music, games, books, performers, directors,
                  actors, fiction, comics.
  SCIENCE       = Math, physics, chemistry, biology, astronomy, engineering,
                  technology subject matter.
  OTHER         = Anything that does not fit cleanly into the three above
                  (people's biographies, history, sports, business, etc.).

Examples:

Question: Kam Heskin plays Paige Morgan in which 2004 film?
Answer: The Prince and Me
Output: {{"category":"RETRIEVAL","retrieval_subtype":"FACT_LOOKUP","domain":"ENTERTAINMENT"}}

Question: Who is the mother of Princess Stéphanie of Monaco?
Answer: Grace Kelly
Output: {{"category":"RETRIEVAL","retrieval_subtype":"GENEALOGY","domain":"OTHER"}}

Question: Who is the father of Chiang Hsiao-Wu?
Answer: Chiang Ching-kuo
Output: {{"category":"RETRIEVAL","retrieval_subtype":"GENEALOGY","domain":"OTHER"}}

Question: When was Hunter Davies born?
Answer: 7 January 1936
Output: {{"category":"RETRIEVAL","retrieval_subtype":"FACT_LOOKUP","domain":"OTHER"}}

Question: How many non-Hispanic whites lived in New York City in 2012?
Answer: 2.7 million
Output: {{"category":"RETRIEVAL","retrieval_subtype":"FACT_LOOKUP","domain":"GEOGRAPHIC"}}

Question: How many letters are there between the first and last letters of Martha?
Answer: 4
Output: {{"category":"COMPUTATION","retrieval_subtype":"NONE","domain":"OTHER"}}

Question: What is the ASCII code of the first letter of the film title?
Answer: 84
Output: {{"category":"COMPUTATION","retrieval_subtype":"NONE","domain":"OTHER"}}

Question: Manchester Township >> located in the administrative territorial entity
Answer: Dearborn County
Output: {{"category":"RETRIEVAL","retrieval_subtype":"FACT_LOOKUP","domain":"GEOGRAPHIC"}}

Question: Which of the two film directors was born earlier?
Answer: Martha Coolidge
Output: {{"category":"REASONING","retrieval_subtype":"NONE","domain":"ENTERTAINMENT"}}

Question: What are the prime factors of 2003?
Answer: [2003]
Output: {{"category":"COMPUTATION","retrieval_subtype":"NONE","domain":"SCIENCE"}}

Question: who invaded #1 and tried to take over their country
Answer: North Korea
Output: {{"category":"RETRIEVAL","retrieval_subtype":"FACT_LOOKUP","domain":"OTHER"}}

Now classify this subtask. Respond with EXACTLY one JSON object (no prose, no
backticks, no commentary) with the three fields shown above.

Question: {question}
Answer: {answer}
Output:"""

VALID_CATEGORIES = {"RETRIEVAL", "COMPUTATION", "REASONING"}
VALID_RETRIEVAL_SUBTYPES = {"FACT_LOOKUP", "GENEALOGY", "NONE"}
VALID_DOMAINS = {"GEOGRAPHIC", "ENTERTAINMENT", "SCIENCE", "OTHER"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset: str) -> list[dict]:
    """Return the full list of question entries for the given dataset."""
    path = CLEANED_PATHS[dataset]
    if path.suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f]
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Bedrock call
# ---------------------------------------------------------------------------

_thread_local = threading.local()

def _get_client():
    """Return a per-thread Bedrock runtime client."""
    if not hasattr(_thread_local, "client"):
        import boto3
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        _thread_local.client = boto3.client("bedrock-runtime", region_name=region)
    return _thread_local.client


import re as _re
_JSON_OBJ_RE = _re.compile(r"\{[^{}]*\}", _re.DOTALL)


def _parse_classification(text: str) -> dict:
    """Extract category / retrieval_subtype / domain from a Haiku response.

    Robust to: code-fenced JSON, lowercase fields, missing fields, and
    plain non-JSON responses (falls back to substring match on the category
    name so older one-token replies still classify correctly).
    """
    m = _JSON_OBJ_RE.search(text)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None

    if obj is None:
        # Fallback: scan free text for known tokens.
        upper = text.upper()
        cat = next((c for c in VALID_CATEGORIES if c in upper), f"UNKNOWN:{text[:30]}")
        sub = next((s for s in VALID_RETRIEVAL_SUBTYPES if s != "NONE" and s in upper),
                   "FACT_LOOKUP" if cat == "RETRIEVAL" else "NONE")
        dom = next((d for d in VALID_DOMAINS if d in upper), "OTHER")
        if cat != "RETRIEVAL":
            sub = "NONE"
        return {"category": cat, "retrieval_subtype": sub, "domain": dom}

    cat = str(obj.get("category", "")).strip().upper()
    sub = str(obj.get("retrieval_subtype", "NONE")).strip().upper()
    dom = str(obj.get("domain", "OTHER")).strip().upper()

    if cat not in VALID_CATEGORIES:
        cat = next((c for c in VALID_CATEGORIES if c in cat), f"UNKNOWN:{cat[:30]}")
    if cat != "RETRIEVAL":
        sub = "NONE"
    elif sub not in VALID_RETRIEVAL_SUBTYPES or sub == "NONE":
        sub = "FACT_LOOKUP"
    if dom not in VALID_DOMAINS:
        dom = "OTHER"

    return {"category": cat, "retrieval_subtype": sub, "domain": dom}


def classify_one(question: str, answer: str, retries: int = 4) -> dict:
    """Call Bedrock and return {category, retrieval_subtype, domain}."""
    prompt = CLASSIFY_PROMPT.format(question=question, answer=answer)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 96,
        "messages": [{"role": "user", "content": prompt}],
    })
    backoff = 2.0
    for attempt in range(retries):
        try:
            resp = _get_client().invoke_model(
                modelId=BEDROCK_MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            text = json.loads(resp["body"].read())["content"][0]["text"].strip()
            return _parse_classification(text)
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                raise RuntimeError(f"Bedrock call failed after {retries} attempts: {exc}") from exc
    return {"category": "ERROR", "retrieval_subtype": "NONE", "domain": "OTHER"}


# ---------------------------------------------------------------------------
# Main classification loop
# ---------------------------------------------------------------------------

def classify_dataset(dataset: str, workers: int = 20, dry_run: bool = False) -> None:
    entries = load_dataset(dataset)

    # Flatten all subtasks into a work list
    work: list[tuple[int, int, dict]] = [
        (ei, si, s)
        for ei, entry in enumerate(entries)
        for si, s in enumerate(entry["question_decomposition"])
    ]

    total = len(work)
    print(f"[{dataset}] {total} subtasks to classify.")

    if dry_run:
        print(f"[{dataset}] DRY RUN — showing first 3 prompts (truncated):")
        for ei, si, s in work[:3]:
            rendered = CLASSIFY_PROMPT.format(question=s["question"], answer=s["answer"])
            print(f"\n  ── subtask {ei}.{si} ──")
            print(f"  Q: {s['question']!r}")
            print(f"  A: {s['answer']!r}")
            print(f"  Prompt tail (last 400 chars):\n  ...{rendered[-400:]}")
        return

    def _task(ei: int, si: int, s: dict) -> tuple[int, int, dict]:
        result = classify_one(s["question"], s["answer"])
        return ei, si, result

    bar = tqdm(total=total, desc=dataset, unit="subtask")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, ei, si, s): (ei, si) for ei, si, s in work}
        for fut in as_completed(futures):
            try:
                ei, si, result = fut.result()
                sub = entries[ei]["question_decomposition"][si]
                # `label` kept as a back-compat alias of `category` so that
                # any existing analysis script that reads "label" still works.
                sub["label"]             = result["category"]
                sub["category"]          = result["category"]
                sub["retrieval_subtype"] = result["retrieval_subtype"]
                sub["domain"]            = result["domain"]
            except Exception as exc:
                ei, si = futures[fut]
                tqdm.write(f"ERROR [entry {ei}, subtask {si}]: {exc}")
            bar.update(1)
    bar.close()

    # Write labels back into the cleaned file
    path = CLEANED_PATHS[dataset]
    if path.suffix == ".jsonl":
        with path.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    else:
        path.write_text(json.dumps(entries, indent=2))

    print(f"[{dataset}] Done — labels written to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Classify subtasks via Claude Haiku on Bedrock.")
    ap.add_argument("--dataset", choices=list(CLEANED_PATHS) + ["all"], default="all",
                    help="Which dataset to classify (default: all)")
    ap.add_argument("--workers", type=int, default=20,
                    help="Number of concurrent Bedrock threads (default: 20)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print prompts for the first 5 subtasks without calling Bedrock")
    args = ap.parse_args()

    datasets = list(CLEANED_PATHS) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        classify_dataset(dataset=ds, workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
