"""
trim_half.py

Takes a random half of a dataset with optional stratification.
Supports JSON arrays (.json) and line-delimited JSONL (.jsonl).
The sample is reproducible given the same seed.

Without --stratify-by: simple random half of all records.
With --stratify-by <field>: sample half from each stratum independently,
  preserving the original distribution of that field.

Special value for --stratify-by:
  "id_hop_prefix"      -- extracts the hop-count prefix from raw MuSiQue ids
                          (e.g. "2hop__482757_12019" -> "2hop")
  "hop_from_decomp_len" -- uses len(question_decomposition) as the hop count;
                           use this on cleaned MuSiQue files where the original
                           id has been replaced with a UUID.

--half-hops:
  Only applies when --stratify-by id_hop_prefix is set. Treats stratum
  "2hop" as the "easy" tier and keeps only 1/3 of it, while keeping ALL
  records from 3-hop and 4-hop strata.

Usage:
    python trim_half.py <input> <output> --seed <int> [--stratify-by FIELD]
                        [--half-hops]

Examples:
    # MuSiQue raw: 1/3 of 2-hop, all 3-hop and 4-hop
    python trim_half.py data/musique_ans_v1.0_train.jsonl data/musique_trim.jsonl \\
        --seed 42 --stratify-by id_hop_prefix --half-hops

    # MuSiQue cleaned (UUID ids): same trimming via decomp length
    python trim_half.py cleaned_data/musique_cleaned.jsonl cleaned_trimmed_data/musique_cleaned.jsonl \\
        --seed 42 --stratify-by hop_from_decomp_len --half-hops

    # MuSiQue: preserve hop-count distribution (plain half)
    python trim_half.py data/musique_ans_v1.0_train.jsonl data/musique_half.jsonl \\
        --seed 42 --stratify-by id_hop_prefix

    # MoReHopQA: preserve answer-type distribution
    python trim_half.py data/with_human_verification.json data/morehopqa_half.json \\
        --seed 42 --stratify-by answer_type

    # No stratification
    python trim_half.py data/with_human_verification.json data/morehopqa_half.json \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


def _detect_format(path: str, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    return "jsonl" if path.lower().endswith(".jsonl") else "json"


def _stratum_key(record: dict, field: str) -> str:
    if field == "id_hop_prefix":
        raw = record.get("id", "")
        return raw.split("__")[0] if "__" in raw else "unknown"
    if field == "hop_from_decomp_len":
        return str(len(record.get("question_decomposition", [])))
    val = record.get(field)
    return str(val) if val is not None else "unknown"


def _stratified_half(
    records: list, field: str, seed: int, half_hops: bool = False
) -> list:
    rng = random.Random(seed)
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        buckets[_stratum_key(r, field)].append(r)

    sampled = []
    for key in sorted(buckets):
        bucket = buckets[key]
        # With --half-hops: keep only 1/3 of the 2-hop stratum; keep all others in full.
        if half_hops and key in ("2hop", "2"):
            k = math.ceil(len(bucket) / 3)
            chosen = rng.sample(bucket, k=k)
            print(f"  stratum {key!r}: sampled {k} / {len(bucket)} (1/3)")
        elif half_hops:
            chosen = list(bucket)
            print(f"  stratum {key!r}: kept {len(chosen)} / {len(bucket)} (all)")
        else:
            k = math.ceil(len(bucket) / 2)
            chosen = rng.sample(bucket, k=k)
            print(f"  stratum {key!r}: sampled {k} / {len(bucket)}")
        sampled.extend(chosen)

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a random half of a JSON or JSONL dataset reproducibly."
    )
    parser.add_argument("input", help="Path to the input file (.json array or .jsonl)")
    parser.add_argument("output", help="Path for the sampled output file")
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="RNG seed for reproducible sampling.",
    )
    parser.add_argument(
        "--stratify-by",
        default=None,
        metavar="FIELD",
        help="Field to stratify on. Special values: 'id_hop_prefix' (raw MuSiQue ids), "
             "'hop_from_decomp_len' (cleaned MuSiQue with UUID ids).",
    )
    parser.add_argument(
        "--half-hops",
        action="store_true",
        help="When used with --stratify-by id_hop_prefix: keep only 1/3 of 2-hop "
             "records but keep ALL 3-hop and 4-hop records.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "json", "jsonl"),
        default="auto",
        help="auto: .jsonl -> JSONL; otherwise JSON array. Default: auto.",
    )
    args = parser.parse_args()

    fmt = _detect_format(args.input, args.input_format)

    # --- read ---
    print(f"Reading {args.input} ({fmt}) ...")
    if fmt == "jsonl":
        records = []
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(args.input, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            print("Error: expected a JSON array at the top level.", file=sys.stderr)
            sys.exit(1)

    total = len(records)

    if args.half_hops and args.stratify_by not in ("id_hop_prefix", "hop_from_decomp_len"):
        print("Error: --half-hops requires --stratify-by id_hop_prefix.", file=sys.stderr)
        sys.exit(1)

    # --- sample ---
    if args.stratify_by:
        print(
            f"Stratified sampling by '{args.stratify_by}' "
            f"(seed={args.seed}, half-hops={args.half_hops}) ..."
        )
        sampled = _stratified_half(
            records, args.stratify_by, args.seed, half_hops=args.half_hops
        )
    else:
        k = total // 2
        print(f"Sampling {k} / {total} records (seed={args.seed}) ...")
        random.seed(args.seed)
        sampled = random.sample(records, k=k)

    print(f"Total sampled: {len(sampled)} / {total}")

    # --- write ---
    out_fmt = _detect_format(args.output, args.input_format)
    print(f"Writing {args.output} ...")
    if out_fmt == "jsonl":
        with open(args.output, "w", encoding="utf-8") as f:
            for row in sampled:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(sampled, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
