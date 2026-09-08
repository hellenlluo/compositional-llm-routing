"""Prometheus-7B-v2.0 absolute-grading prompt builders.

Prometheus was fine-tuned on a specific template. Generic JSON-output prompts
don't produce reliable scores — stick to this format:

  ###Task Description:
  ...
  ###The instruction to evaluate: {question}
  ###Response to evaluate: {response}
  ###Reference Answer (Score 5): {gold}
  ###Score Rubrics:
  [criterion]
  Score 1: ...
  Score 5: ...
  ###Feedback:

Output: "Feedback: <reason> [RESULT] <1-5>"
"""
from __future__ import annotations

import re


_TASK_DESCRIPTION = (
    "An instruction, a response, a reference answer and a score rubric are given.\n"
    "1. Write a detailed feedback that assesses the quality of the response strictly "
    "based on the given score rubric, not evaluating in general.\n"
    "2. After writing a feedback, write a score that is an integer between 1 and 5. "
    "You should refer to the score rubric.\n"
    "3. The output format should look as follows: \"Feedback: (write a feedback for "
    "criteria) [RESULT] (an integer number between 1 and 5)\"\n"
    "4. Please do not generate any other opening, closing, and explanations."
)


_MUSIQUE_RUBRIC = """[Is the model's answer semantically correct given the ground truth and any listed aliases?]
Score 1: Completely wrong, unrelated, or refuses to answer.
Score 2: Substantially wrong; may contain a fragment of truth but not the answer.
Score 3: Partially correct — close but missing or adding information.
Score 4: Essentially correct with minor surface-level issues (wording, trailing text).
Score 5: Fully correct — matches the reference answer (or any listed alias) semantically. Numeric answers must match exactly; "4" and "four" are both fine."""


_MOREHOPQA_RUBRIC = """[Is the model's answer correct given the ground truth?]
Score 1: Completely wrong, unrelated, or refuses to answer.
Score 2: Substantially wrong; only tangentially related.
Score 3: Partially correct but missing key information or containing a contradiction.
Score 4: Essentially correct with minor surface-level issues.
Score 5: Fully correct — matches the reference answer semantically. Numeric answers must match exactly."""


_STEPCOT_RUBRIC = """[Does the model's answer select the same option letter as the ground truth (or correctly say "N/A")?]
Score 1: Selects a different option letter, or refuses to commit to a single option.
Score 2: Ambiguous — mentions multiple options without clearly choosing one.
Score 3: (not used — treat borderline cases as 2 or 4).
Score 4: Selects the correct option letter but with significant unrelated commentary.
Score 5: Cleanly selects exactly the correct option letter (or correctly answers "N/A" when ground truth is "N/A")."""


_RUBRICS = {
    "musique": _MUSIQUE_RUBRIC,
    "morehopqa": _MOREHOPQA_RUBRIC,
    "stepcot": _STEPCOT_RUBRIC,
}


def build(dataset: str, question: str, response: str, gold: str,
          gold_aliases: list[str] | None = None,
          options: list[str] | None = None) -> str:
    rubric = _RUBRICS[dataset]
    ref_answer = gold
    if dataset == "musique" and gold_aliases:
        ref_answer = f"{gold} (also acceptable: {', '.join(gold_aliases)})"
    if dataset == "stepcot" and options:
        opts_block = "\n".join(options)
        question = f"{question}\nOptions:\n{opts_block}"

    return (
        f"###Task Description:\n{_TASK_DESCRIPTION}\n\n"
        f"###The instruction to evaluate:\n{question}\n\n"
        f"###Response to evaluate:\n{response}\n\n"
        f"###Reference Answer (Score 5):\n{ref_answer}\n\n"
        f"###Score Rubrics:\n{rubric}\n\n"
        f"###Feedback:"
    )


_RESULT_RE = re.compile(r"\[RESULT\]\s*([1-5])")


def parse_score(raw: str) -> tuple[int | None, str]:
    """Returns (score_or_None, feedback_text). On parse failure, score is None."""
    m = _RESULT_RE.search(raw)
    if not m:
        return None, raw.strip()
    score = int(m.group(1))
    feedback = raw[: m.start()].strip()
    if feedback.lower().startswith("feedback:"):
        feedback = feedback[len("feedback:"):].strip()
    return score, feedback
