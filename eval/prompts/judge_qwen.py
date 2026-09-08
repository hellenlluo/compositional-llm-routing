"""Qwen-style judge prompts — use a clean JSON output format instead of
Prometheus's `[RESULT] N` pattern.

Qwen models follow "respond only in JSON" instructions much better than
Prometheus-7B's idiosyncratic template.
"""
from __future__ import annotations

import json
import re


_JUDGE_SYSTEM = (
    "You are a strict, accurate evaluator for question-answering tasks. "
    "Given a question, a model's answer, and a reference answer, decide if "
    "the model's answer is correct.\n\n"
    "Rules:\n"
    "- The model answer must match the reference answer semantically. Exact "
    "wording is not required — synonyms and rephrasing are fine if the "
    "meaning matches.\n"
    "- Any listed alias is equally acceptable as the reference answer.\n"
    "- Numeric answers must match in value (e.g. \"4\" == \"four\" == 4.0, but \"5\" is wrong).\n"
    "- If the model refuses to answer or says the information is not in the "
    "context, it is INCORRECT (mark as false).\n"
    "- If the model gives multiple candidates without committing, it is INCORRECT.\n"
    "- Ignore formatting differences (spaces, hyphens, case) when matching.\n\n"
    "Respond ONLY with a single JSON object on one line:\n"
    "{\"correct\": true} or {\"correct\": false}\n"
    "Do not include explanations, code fences, or any other text."
)


def build(dataset: str, question: str, response: str, gold: str,
          gold_aliases: list[str] | None = None,
          options: list[str] | None = None) -> str:
    ref = gold
    if dataset == "musique" and gold_aliases:
        ref = f"{gold} (also acceptable: {', '.join(gold_aliases)})"
    if dataset == "stepcot" and options:
        question = f"{question}\nOptions:\n" + "\n".join(options)

    return (
        f"{_JUDGE_SYSTEM}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"MODEL ANSWER:\n{response}\n\n"
        f"REFERENCE ANSWER:\n{ref}\n\n"
        f"Output the JSON object now:"
    )


_JSON_RE = re.compile(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', re.I)


def parse_correct(raw: str) -> tuple[bool | None, str]:
    """Return (correct, raw_text). None if we couldn't parse."""
    # try exact JSON first
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and "correct" in obj:
            return bool(obj["correct"]), raw.strip()
    except Exception:
        pass
    # regex fallback
    m = _JSON_RE.search(raw)
    if m:
        val = m.group(1).lower() == "true"
        return val, raw.strip()
    return None, raw.strip()
