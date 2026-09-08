"""Full-question prompt builders — direct translations of training_info/prompts.md."""
from __future__ import annotations

from eval.loaders import Record


def _paragraphs_block(paragraphs: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['title']}]: {p['paragraph_text']}" for p in paragraphs
    )


def _context_block(context: list) -> str:
    # context is list of [title, [sentences]]
    return "\n\n".join(
        f"[{title}]: {' '.join(sentences)}" for title, sentences in context
    )


def musique_full(record: Record) -> str:
    ctx = _paragraphs_block(record.context_payload["paragraphs"])
    return (
        "You are given a set of context paragraphs and a question that requires "
        "reasoning over multiple paragraphs to answer. Read the paragraphs carefully "
        "and answer the question.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {record.question}\n\n"
        "Provide a brief explanation (1-2 sentences), then write your final answer concisely on its own line starting with \"ANSWER: \". "
        "Base your answer only on the information provided in the context paragraphs."
    )


def morehopqa_full(record: Record) -> str:
    ctx = _context_block(record.context_payload["context"])
    return (
        "You are given a set of context passages and a multi-hop reasoning question. "
        "The question may require combining information from more than one passage to "
        "arrive at the final answer.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {record.question}\n\n"
        "Think step by step briefly (1-2 sentences), then write your final answer concisely on its own line starting with \"ANSWER: \". "
        "Base your answer only on the context above."
    )


def stepcot_full(record: Record) -> str:
    report = record.context_payload["report"]
    options = record.context_payload.get("final_options") or []
    opts_block = "\n".join(options)
    origin = record.origin or "unknown"
    return (
        f"You are a radiologist assistant. You are given a radiology report from the {origin} dataset. "
        "Read the report carefully and answer the diagnostic question by selecting one of the provided options.\n\n"
        f"Report:\n{report}\n\n"
        f"Question: {record.question}\n\n"
        f"Options:\n{opts_block}\n\n"
        "Select the single best option letter and briefly explain why (1-2 sentences). "
        "Then write your final answer on its own line starting with \"ANSWER: \" followed by just the option letter."
    )


BUILDERS = {
    "musique": musique_full,
    "morehopqa": morehopqa_full,
    "stepcot": stepcot_full,
}


def build(record: Record) -> str:
    return BUILDERS[record.dataset](record)
