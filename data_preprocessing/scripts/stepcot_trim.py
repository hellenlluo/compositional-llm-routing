"""
stepcot_trim.py

Loads StepCoT-style JSON (top-level array of objects with an "origin" field),
draws a random subset of up to N examples per distinct origin, and writes a new
JSON array.

Usage:
    python stepcot_trim.py <input.json> <output.json> [--per-origin N] [--seed S]

Example:
    python stepcot_trim.py data/stepcot_full.json data/stepcot_trim_3k.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random subsample of StepCoT data: N examples per origin value."
    )
    parser.add_argument("input", help="Path to input JSON (array of objects with 'origin')")
    parser.add_argument("output", help="Path for the trimmed JSON array")
    parser.add_argument(
        "--per-origin",
        type=int,
        default=200,
        help="Max examples to keep per origin (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible sampling (optional).",
    )
    args = parser.parse_args()

    if args.per_origin < 1:
        print("Error: --per-origin must be at least 1.", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Reading {args.input} ...")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: expected a JSON array at the top level.", file=sys.stderr)
        sys.exit(1)

    by_origin: dict[str, list] = defaultdict(list)
    for row in data:
        if not isinstance(row, dict):
            print("Error: expected each element to be an object.", file=sys.stderr)
            sys.exit(1)
        origin = row.get("origin")
        if origin is None:
            print("Error: record missing 'origin' field.", file=sys.stderr)
            sys.exit(1)
        by_origin[str(origin)].append(row)

    origins = sorted(by_origin.keys())
    print(f"Found {len(origins)} origin value(s): {', '.join(origins)}")

    trimmed: list = []
    for origin in origins:
        bucket = by_origin[origin]
        k = min(args.per_origin, len(bucket))
        if len(bucket) < args.per_origin:
            print(
                f"Warning: origin {origin!r} has only {len(bucket)} example(s); "
                f"taking all {k}.",
                file=sys.stderr,
            )
        chosen = random.sample(bucket, k=k)
        trimmed.extend(chosen)
        print(f"  {origin!r}: sampled {k} / {len(bucket)}")

    print(f"Writing {len(trimmed)} total rows to {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
