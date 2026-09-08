"""
data_clean.py

MoReHopQA (JSON array): retains id, index, question, answer, question_decomposition,
context; flattens nested "details" in decomposition; replaces "_id" with a deterministic
UUID v5 from dataset_name and "_id".

MuSiQue (JSONL): keeps each record as-is except replaces top-level "id" with a
deterministic UUID v5 from dataset_name and the original "id"; adds 0-based "index".

StepCoT (JSON array, e.g. stepcot_trimmed.json): drops image_path; keeps patient_id
unchanged; adds new "id" (UUID v5 from dataset_name, origin, and patient_id).

Usage:
    python data_clean.py <input_path> <output_path> <dataset_name> [--input-format FMT]

Examples:
    python data_clean.py with_human_verification.json cleaned.json morehopqa
    python data_clean.py musique_ans_v1.0_train.jsonl cleaned.jsonl musique
    python data_clean.py stepcot_trimmed.json stepcot_cleaned.json stepcot
"""

import argparse
import json
import sys
import uuid
from pathlib import Path


def flatten_decomposition(question_decomposition: list) -> list:
    """
    Iterate over each sub-question. If it has a "details" list, replace it
    with the detail entries (and recurse in case details also have details).
    Otherwise keep the sub-question as-is, dropping unknown fields.
    """
    result = []
    for step in question_decomposition:
        if "details" in step and step["details"]:
            # Recurse to handle arbitrarily nested details
            result.extend(flatten_decomposition(step["details"]))
        else:
            result.append({
                "sub_id": step["sub_id"],
                "question": step.get("question", ""),
                "answer": step.get("answer", ""),
                "paragraph_support_title": step.get("paragraph_support_title", ""),
            })
    return result


def morehopqa_clean_entry(
    index: int,
    entry: dict,
    *,
    namespace: uuid.UUID,
    dataset_name: str,
) -> dict:
    return {
        "id": str(uuid.uuid5(namespace, f"{dataset_name}:{entry['_id']}")),
        "index": index,
        "question": entry["question"],
        "answer": entry["answer"],
        "question_decomposition": flatten_decomposition(
            entry.get("question_decomposition", [])
        ),
        "context": entry.get("context", []),
    }


def musique_clean_entry(
    index: int,
    entry: dict,
    *,
    namespace: uuid.UUID,
    dataset_name: str,
) -> dict:
    orig_id = entry["id"]
    out = dict(entry)
    out["id"] = str(uuid.uuid5(namespace, f"{dataset_name}:{orig_id}"))
    out["index"] = index
    return out


def stepcot_clean_entry(
    entry: dict,
    *,
    namespace: uuid.UUID,
    dataset_name: str,
) -> dict:
    patient_id = entry["patient_id"]
    origin = entry.get("origin", "")
    new_id = str(
        uuid.uuid5(namespace, f"{dataset_name}:{origin}:{patient_id}")
    )
    out: dict = {"id": new_id}
    for k, v in entry.items():
        if k == "image_path":
            continue
        out[k] = v
    return out


def _resolve_input_format(path: str, fmt: str) -> str:
    if fmt == "json":
        return "morehopqa"
    if fmt != "auto":
        return fmt
    low = path.lower()
    if low.endswith(".jsonl"):
        return "jsonl"
    if "stepcot" in Path(path).stem.lower():
        return "stepcot"
    return "morehopqa"


def main():
    namespace = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    parser = argparse.ArgumentParser(
        description="Clean MoReHopQA, MuSiQue JSONL, or StepCoT JSON; UUID v5 uses dataset_name."
    )
    parser.add_argument("input", help="Path to the input file (.json array or .jsonl)")
    parser.add_argument("output", help="Path for the cleaned output (.json or .jsonl)")
    parser.add_argument(
        "dataset_name",
        help="Label for UUID v5 namespacing (e.g. morehopqa, musique, stepcot).",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "json", "jsonl", "morehopqa", "stepcot"),
        default="auto",
        help="auto: .jsonl -> musique; filename contains 'stepcot' -> stepcot; else MoReHopQA. "
        "'json' is an alias for morehopqa.",
    )
    args = parser.parse_args()

    input_fmt = _resolve_input_format(args.input, args.input_format)

    if input_fmt == "jsonl":
        print(f"Reading {args.input} (JSONL) ...")
        cleaned = []
        with open(args.input, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cleaned.append(
                    musique_clean_entry(
                        len(cleaned),
                        entry,
                        namespace=namespace,
                        dataset_name=args.dataset_name,
                    )
                )
        print(f"Processed {len(cleaned)} entries ...")
        print(f"Writing {args.output} ...")
        if args.output.lower().endswith(".jsonl"):
            with open(args.output, "w", encoding="utf-8") as out:
                for row in cleaned:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            with open(args.output, "w", encoding="utf-8") as out:
                json.dump(cleaned, out, indent=2, ensure_ascii=False)
    else:
        print(f"Reading {args.input} ...")
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Error: expected a JSON array at the top level.", file=sys.stderr)
            sys.exit(1)

        print(f"Cleaning {len(data)} entries ({input_fmt}) ...")
        if input_fmt == "stepcot":
            cleaned = [
                stepcot_clean_entry(
                    entry, namespace=namespace, dataset_name=args.dataset_name
                )
                for entry in data
            ]
        else:
            cleaned = [
                morehopqa_clean_entry(
                    i, entry, namespace=namespace, dataset_name=args.dataset_name
                )
                for i, entry in enumerate(data)
            ]

        print(f"Writing {args.output} ...")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
