#!/usr/bin/env python3
"""
Trim training_pairs.jsonl by removing entries whose prompt_id appears in
circle_test_failures.json's 'exclude' list (circle-test failures + uninformative).

Input:
    modelsat_data/training_pairs.jsonl
    data/circle_test_failures.json

Output:
    modelsat_data/training_pairs_trimmed.jsonl

Usage:
    python modelsat_baseline/trim_training_pairs.py
    python modelsat_baseline/trim_training_pairs.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELSAT_DIR   = Path(__file__).resolve().parent
DATA_DIR       = MODELSAT_DIR / "data"
MODELSAT_DATA  = MODELSAT_DIR / "modelsat_data"

FAILURES_FILE  = DATA_DIR / "circle_test_failures.json"
INPUT_FILE     = MODELSAT_DATA / "training_pairs.jsonl"
OUTPUT_FILE    = MODELSAT_DATA / "training_pairs_trimmed.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats without writing output file")
    args = ap.parse_args()

    # Load the exclude set
    with FAILURES_FILE.open() as f:
        failures = json.load(f)
    exclude: set[int] = set(failures["exclude"])
    print(f"Exclude set: {len(exclude)} prompt_ids "
          f"({len(failures['circle_test_failures'])} circle-test failures + "
          f"{len(failures['uninformative'])} uninformative)")

    # Stream through training_pairs.jsonl
    kept = 0
    dropped = 0
    lines_out: list[str] = []

    with INPUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("prompt_id") in exclude:
                dropped += 1
            else:
                kept += 1
                lines_out.append(line)

    total = kept + dropped
    print(f"Input:   {total:,} entries")
    print(f"Dropped: {dropped:,} entries ({dropped/total*100:.1f}%)")
    print(f"Kept:    {kept:,} entries ({kept/total*100:.1f}%)")

    if args.dry_run:
        print("\nDry run — no file written.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as f:
        for line in lines_out:
            f.write(line + "\n")
    print(f"\nWritten → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
