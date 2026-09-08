"""
pretty_jsonl.py

Reads JSON Lines (one JSON object per line) and writes a human-readable file
where each object is formatted with indentation (fields and nested structure
on separate lines). Records are separated by a blank line.

Streams line-by-line so large files do not need to fit in memory.

Usage:
    python pretty_jsonl.py <input.jsonl> [output_path]

If output_path is omitted, writes alongside the input as <name>.pretty.jsonl
(each logical record spans multiple lines; the file is for reading in an editor,
not strict one-object-per-line JSONL).

Example:
    python pretty_jsonl.py data/musique_ans_v1.0_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretty-print a JSONL file for easier scrolling in an editor."
    )
    parser.add_argument("input", help="Path to the input .jsonl file")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output path (default: <input>.pretty.jsonl next to input)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent width (default: 2)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input is not a file: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is not None:
        output_path = Path(args.output)
        if os.path.isdir(str(output_path)):
            output_path = output_path / (input_path.stem + ".pretty.jsonl")
    else:
        output_path = input_path.with_suffix(".pretty.jsonl")

    print(f"Reading {input_path} ...")
    print(f"Writing {output_path} ...")

    line_no = 0
    written = 0
    with open(input_path, "r", encoding="utf-8") as inf, open(
        output_path, "w", encoding="utf-8"
    ) as outf:
        first_record = True
        for line in inf:
            line_no += 1
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                print(
                    f"Error: invalid JSON on line {line_no}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not first_record:
                outf.write("\n")
            first_record = False
            outf.write(
                json.dumps(obj, indent=args.indent, ensure_ascii=False)
            )
            outf.write("\n")
            written += 1

    print(f"Done. Wrote {written} records.")


if __name__ == "__main__":
    main()
