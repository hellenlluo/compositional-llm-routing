"""Subtask prompt builders with teacher-forced gold answers for prior subtasks.

Translated verbatim from training_info/prompts.md. Sub-questions are 1-indexed
in the prompt text but the record stores them 0-indexed (subtask.idx).
"""
from __future__ import annotations

from eval.loaders import Record
from eval.prompts.full_question import _paragraphs_block, _context_block


def _prior_block(record: Record, upto_idx: int) -> str:
    if upto_idx == 0:
        return ""
    lines = ["Previously answered sub-questions:"]
    for j in range(upto_idx):
        s = record.subtasks[j]
        lines.append(f"Q{j + 1}: {s.question}")
        lines.append(f"A{j + 1}: {s.gold}")
    return "\n".join(lines) + "\n\n"


def musique_sub(record: Record, subtask_idx: int) -> str:
    ctx = _paragraphs_block(record.context_payload["paragraphs"])
    cur = record.subtasks[subtask_idx]
    prior = _prior_block(record, subtask_idx)
    return (
        "You are given a set of context paragraphs and a series of sub-questions "
        "that build on each other. Answer the current sub-question using the context "
        "and any previously answered sub-questions.\n\n"
        f"Context:\n{ctx}\n\n"
        f"{prior}"
        f"Current sub-question: {cur.question}\n\n"
        "Provide a brief explanation (1-2 sentences), then write your final answer concisely on its own line starting with \"ANSWER: \". "
        "Base your answer on the context and any previous answers above."
    )


def morehopqa_sub(record: Record, subtask_idx: int) -> str:
    ctx = _context_block(record.context_payload["context"])
    cur = record.subtasks[subtask_idx]
    prior = _prior_block(record, subtask_idx)
    if cur.paragraph_support_title:
        instruction = "Base your answer on the context and any previous answers above."
    else:
        instruction = (
            "This sub-question does not require looking up new information — "
            "derive the answer directly from the previously answered sub-questions "
            "using reasoning or computation."
        )
    return (
        "You are given context passages and a series of sub-questions that together "
        "solve a multi-hop reasoning problem. Answer the current sub-question.\n\n"
        f"Context:\n{ctx}\n\n"
        f"{prior}"
        f"Current sub-question: {cur.question}\n\n"
        "Provide a brief explanation (1-2 sentences), then write your final answer concisely on its own line starting with \"ANSWER: \". "
        f"{instruction}"
    )


def stepcot_sub(record: Record, subtask_idx: int) -> str:
    report = record.context_payload["report"]
    origin = record.origin or "unknown"
    cur = record.subtasks[subtask_idx]
    opts_block = "\n".join(cur.options or [])
    prior_lines = []
    if subtask_idx > 0:
        prior_lines.append("Previous diagnostic steps:")
        for j in range(subtask_idx):
            s = record.subtasks[j]
            prior_lines.append(f"Step {j + 1}: {s.question}")
            prior_lines.append(f"Answer: {s.gold}")
        prior_lines.append("")
    prior = "\n".join(prior_lines) + ("\n" if prior_lines else "")
    step_num = subtask_idx + 1
    return (
        f"You are a radiologist assistant analyzing a report from the {origin} dataset. "
        "Answer the current step of a structured diagnostic reasoning chain by selecting "
        "one of the provided options. If no finding is relevant, answer \"N/A\".\n\n"
        f"Report:\n{report}\n\n"
        f"{prior}"
        f"Step {step_num}: {cur.question}\n\n"
        f"Options:\n{opts_block}\n\n"
        "Select the single best option letter (or \"N/A\" if not applicable) and briefly explain why (1-2 sentences). "
        "Then write your final answer on its own line starting with \"ANSWER: \" followed by just the option letter or \"N/A\"."
    )


BUILDERS = {
    "musique": musique_sub,
    "morehopqa": morehopqa_sub,
    "stepcot": stepcot_sub,
}


def build(record: Record, subtask_idx: int) -> str:
    return BUILDERS[record.dataset](record, subtask_idx)
