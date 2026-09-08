"""List all categories in the EmbedLLM prompt set with counts.

Usage:
  python embedllm_eval/list_categories.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = REPO_ROOT / "data_preprocessing" / "embedllm-sampled" / "prompts_subset.jsonl"


def main() -> None:
    counts: Counter[str] = Counter()
    families: dict[str, str] = {}
    with PROMPTS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            counts[entry["category"]] += 1
            families[entry["category"]] = entry.get("family", "unknown")

    print(f"{'Category':<50s} {'Family':<15s} {'Count':>6s}")
    print("-" * 75)
    for cat in sorted(counts):
        print(f"{cat:<50s} {families[cat]:<15s} {counts[cat]:>6d}")
    print("-" * 75)
    print(f"{'Total':<50s} {'':<15s} {sum(counts.values()):>6d}")
    print(f"Categories: {len(counts)}")


if __name__ == "__main__":
    main()
