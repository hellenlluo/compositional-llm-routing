#!/usr/bin/env python3
"""
Fix GPT-2 tokenizer artifacts in deepseek-r1-distill-llama-8b result JSON files.

Ġ (U+0120) -> ' '  (space before a word)
Ċ (U+010A) -> '\n' (newline)
"""

import json
import os
import sys

REPLACEMENTS = {
    "\u0120": " ",  # Ġ -> space
    "\u010a": "\n", # Ċ -> newline
}

STRING_FIELDS = {"model_answer", "raw_response"}


def fix_string(s: str) -> str:
    for bad, good in REPLACEMENTS.items():
        s = s.replace(bad, good)
    return s


def fix_record(record: dict) -> dict:
    return {
        k: fix_string(v) if k in STRING_FIELDS and isinstance(v, str) else v
        for k, v in record.items()
    }


def fix_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        fixed = [fix_record(r) for r in data]
    elif isinstance(data, dict):
        fixed = fix_record(data)
    else:
        print(f"  Skipping {path}: unexpected top-level type {type(data)}")
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"  Fixed: {os.path.basename(path)}")


def main(results_dir: str) -> None:
    json_files = sorted(
        os.path.join(results_dir, fn)
        for fn in os.listdir(results_dir)
        if fn.endswith(".json")
    )

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        sys.exit(1)

    print(f"Processing {len(json_files)} files in {results_dir} ...")
    for path in json_files:
        fix_file(path)
    print("Done.")


if __name__ == "__main__":
    default_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        "deepseek-r1-distill-llama-8b",
    )
    target = sys.argv[1] if len(sys.argv) > 1 else default_dir
    main(target)
