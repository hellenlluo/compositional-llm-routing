"""
Add deterministic, $0-cost axes to every subtask in the cleaned datasets.

These axes are derived purely from the question text and gold answer — no LLM
call. They live alongside the existing "label" field written by
classify_subtasks.py and give downstream routing analyses additional
dimensions to slice on.

Axes added per subtask:
  hop_role        ∈ {seed, bridge, final}   — position in the chain
  coref_to_prior  ∈ {true, false}            — '#' in question text (depends on prior hop)
  is_schema_query ∈ {true, false}            — ' >> ' in question text (Wikidata-shape)
  answer_type     ∈ {number, date, letter, mcq_letter, n_a, entity, other}
  comp_subtype    (only when label == COMPUTATION)
                  ∈ {PARSE, STRING_OP, STRING_OP_PHONETIC, ARITHMETIC, DATE_OP,
                     ENCODING, NUMERIC_FACT, OTHER}

The COMPUTATION-subtype regexes were derived from the actual final-hop catalog
in morehopqa (every stem with N≥3 occurrences is matched). They cover ~99% of
morehopqa COMPUTATION subtasks; the remainder fall to OTHER. On musique the
bucket is tiny (37) and mostly NUMERIC_FACT (the LLM mislabeled retrieval-of-
numbers as computation).

Usage:
  python3 router_analysis/add_deterministic_axes.py
  python3 router_analysis/add_deterministic_axes.py --dataset musique
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data_preprocessing" / "cleaned_trimmed_data"

CLEANED_PATHS = {
    "morehopqa": DATA_DIR / "morehopqa_cleaned.json",
    "musique":   DATA_DIR / "musique_cleaned.jsonl",
    "stepcot":   DATA_DIR / "stepcot_cleaned.json",
}

# ---------------------------------------------------------------------------
# Axis classifiers
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"^(?:\d{1,2}[\s\-/]?)?(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|"
    r"Oct|Nov|Dec)\b|^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
    re.IGNORECASE,
)
_NUMBER_RE  = re.compile(r"^[\-+]?\d[\d,]*\.?\d*\s*[%]?\s*[A-Za-z\$]*$")
_LETTER_RE  = re.compile(r"^[a-zA-Z]$")
_MCQ_RE     = re.compile(r"^[A-F]$")  # stepcot multi-choice letters
_YEAR_RE    = re.compile(r"^\d{4}$")


def classify_answer_type(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return "other"
    if a.upper() == "N/A":
        return "n_a"
    if _LETTER_RE.match(a):
        # Single letter — could be mcq_letter (A-F in stepcot) or letter (lowercase final hop in morehopqa)
        return "mcq_letter" if _MCQ_RE.match(a) else "letter"
    if _DATE_RE.match(a) or _YEAR_RE.match(a):
        return "date"
    # Try number
    cleaned = a.replace(",", "").rstrip("%").strip()
    try:
        float(cleaned)
        return "number"
    except ValueError:
        pass
    return "entity"


# COMPUTATION subtype regexes — applied in order; first match wins.
# Patterns derived from the morehopqa final-hop catalog. Order is significant:
# more specific patterns must match before broader ones (e.g. STRING_OP_PHONETIC
# before STRING_OP so syllables don't fall into letter-counting; STRING_OP
# before ARITHMETIC so "total number of letters" stays in letter-counting).
_ORD = (r"first|last|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
        r"tenth|\d+(?:st|nd|rd|th)")

_COMP_SUBTYPES: list[tuple[str, re.Pattern]] = [
    # STRING_OP_PHONETIC — pronunciation-dependent. Checked first so it cannot
    # be swallowed by the generic letter-counting STRING_OP regex.
    ("STRING_OP_PHONETIC", re.compile(
        r"\bsyllabl",
        re.IGNORECASE,
    )),
    # PARSE — pure structural extraction, no real computation. Includes
    # ordinal-position digit/letter pulls (first/second/third/.../Nth digit).
    ("PARSE", re.compile(
        r"^(what (is|are) the (" + _ORD + r") (name|letter|digit|vowel|consonant)\b"
        r"|what (is|are) the initials\b"
        r"|what (is|are) the year of\b"
        r"|what (is|are) the month \(in (number|words?)\) of\b"
        r"|what (is|are) the day of\b"
        r"|what (is|are) the (" + _ORD + r") (10|\d+) (of|letters)\b)",
        re.IGNORECASE,
    )),
    # ENCODING — format conversions. Catches both phrasings of timezone-offset
    # arithmetic ("between the winter time zone of X and UTC", and the bare
    # "between America/Los_Angeles and UTC"); both compute a UTC offset from
    # a timezone identifier, which is structurally a name->offset encoding.
    ("ENCODING", re.compile(
        r"\b(ASCII|binary|base[- ]?\d+|hexadecimal|hex|time zone|timezone)\b"
        r"|^how many hours are there between\s+\S+/\S+\s+and\s+UTC\b",
        re.IGNORECASE,
    )),
    # DATE_OP — calendar arithmetic
    ("DATE_OP", re.compile(
        r"^(what (is|are) the date \w+ (year|month|day|week|days?|months?|years?|weeks?)\b"
        r"|what (is|are) the date \d+\b"
        r"|what (is|are) the year (before|following|after|\d+)\b"
        r"|which day of the week\b"
        r"|how many (days|hours|months|weeks|years) (lay |between |are there )?\b"
        r"|what (is|are) the submission deadline\b)",
        re.IGNORECASE,
    )),
    # STRING_OP — any character/letter/word manipulation. "Total number of
    # letters in X" is letter-counting and lives here, not in ARITHMETIC.
    ("STRING_OP", re.compile(
        r"^(how many (letters|vowels|consonants|unique|distinct|repeated|different) \w*"
        r"|how many letters [a-z]+ are\b"
        r"|how many letters does\b"
        r"|how many times does the letter\b"
        r"|how many distinct two-letter combinations\b"
        r"|what (is|are) the reverse order\b"
        r"|what (is|are) the closest palindrome\b"
        r"|what (is|are) the longest (substring|word|run)\b"
        r"|what (is|are) the shortest word\b"
        r"|what (is|are) the alphabetical order\b"
        r"|what (is|are) the concatenation of\b"
        r"|what (is|are) the combined length\b"
        r"|what (is|are) the length of\b"
        r"|what (is|are) the (number|total number) of letters\b"
        r"|.+ without spaces\?$"
        r"|what (is|are) the (first|last) (vowel|consonant)\b)",
        re.IGNORECASE,
    )),
    # ARITHMETIC — numeric operations. "total number of <X>" matches here
    # only when <X> is not "letters" (those land in STRING_OP above and are
    # consumed first because STRING_OP precedes ARITHMETIC in this list).
    ("ARITHMETIC", re.compile(
        r"^(what (is|are) the (sum|product|square|cube|difference|result) of\b"
        r"|what (is|are) the (sum|product|square|cube|difference|result|total)\b"
        r"|what (is|are) the total number of (?!letters\b)\w+"
        r"|what (are|is) the prime factors of\b"
        r"|how many more is\b"
        r"|by how much is\b)",
        re.IGNORECASE,
    )),
    # NUMERIC_FACT — "how many X in Y" / "what percentage" — really
    # retrieval-of-numbers, mislabeled as COMPUTATION in musique.
    ("NUMERIC_FACT", re.compile(
        r"^(how many (people|miles|meters|feet|kilometers|km|counties|states|sites|members)\b"
        r"|what percent(age)?\b"
        r"|how (much|far|close)\b"
        r"|what (is|are) the period of\b)",
        re.IGNORECASE,
    )),
]


def classify_comp_subtype(question: str) -> str:
    q = question.strip()
    for name, pat in _COMP_SUBTYPES:
        if pat.search(q):
            return name
    return "OTHER"


def hop_role(idx: int, total: int) -> str:
    if total <= 1:
        return "final"
    if idx == 0:
        return "seed"
    if idx == total - 1:
        return "final"
    return "bridge"


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_subtask(s: dict, idx: int, total: int) -> None:
    q = s.get("question", "")
    a = str(s.get("answer", ""))
    s["hop_role"]        = hop_role(idx, total)
    s["coref_to_prior"]  = "#" in q
    s["is_schema_query"] = " >> " in q
    s["answer_type"]     = classify_answer_type(a)
    if s.get("label") == "COMPUTATION":
        s["comp_subtype"] = classify_comp_subtype(q)


def annotate_dataset(dataset: str) -> None:
    path = CLEANED_PATHS[dataset]
    if not path.exists():
        print(f"[{dataset}] file missing: {path}")
        return

    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.open()]
    else:
        rows = json.loads(path.read_text())

    chain_key = "vqa_chain" if dataset == "stepcot" else "question_decomposition"

    n_subs = 0
    for entry in rows:
        chain = entry.get(chain_key, [])
        total = len(chain)
        for i, s in enumerate(chain):
            annotate_subtask(s, i, total)
            n_subs += 1

    if path.suffix == ".jsonl":
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    else:
        path.write_text(json.dumps(rows, indent=2))

    print(f"[{dataset}] annotated {n_subs} subtasks across {len(rows)} questions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CLEANED_PATHS) + ["all"], default="all")
    args = ap.parse_args()
    datasets = list(CLEANED_PATHS) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        annotate_dataset(ds)


if __name__ == "__main__":
    main()
