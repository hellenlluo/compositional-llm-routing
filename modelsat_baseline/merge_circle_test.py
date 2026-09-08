#!/usr/bin/env python3
"""
Merge per-model circle test partial results into a single circle_test_failures.json.

Run this after all per-model jobs from submit_circle_test.sh have completed.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_FILE = DATA_DIR / "circle_test_failures.json"

MODELS = [
    "deepseek-r1-distill-llama-8b",
    "llama-3.1-8b-instruct",
    "llama-3.1-nemotron-nano-8b",
    "mathstral-7b",
    "medgemma-4b-it",
    "mistral-7b-instruct-v0.3",
    "phi-4-mini-instruct",
    "qwen1.5-0.5b-chat",
    "qwen3-30b-a3b",
    "qwen3-4b-thinking-2507",
]

all_failures: set[int]     = set()
all_uninformative: set[int] = set()
missing = []

for model in MODELS:
    partial = DATA_DIR / f"circle_test_{model}.json"
    if not partial.exists():
        missing.append(model)
        print(f"  MISSING: {partial.name}")
        continue
    data = json.loads(partial.read_text())
    failures     = data.get("failures", [])
    uninformative = data.get("uninformative", [])
    all_failures.update(failures)
    all_uninformative.update(uninformative)
    print(f"  {model}: {len(failures)} failures, {len(uninformative)} uninformative")

if missing:
    print(f"\nWARNING: {len(missing)} partial result(s) missing — merge is incomplete.")

output = {
    "circle_test_failures": sorted(all_failures),
    "uninformative":        sorted(all_uninformative),
    "exclude":              sorted(all_failures | all_uninformative),
}

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nMerged {len(MODELS) - len(missing)}/{len(MODELS)} models.")
print(f"  {len(all_failures)} circle test failures")
print(f"  {len(all_uninformative)} uninformative questions")
print(f"  {len(output['exclude'])} total to exclude → {OUT_FILE}")
