"""
Which subtask shapes does subtask routing rescue from full-task failure?

Slices the recovery rate (subtask oracle B given full-task A=0) by:
  1. Which comp_subtype is the "killer hop" — the COMPUTATION subtype that
     appears in the chain. If a chain has multiple comp_subtypes we credit
     each one (so a chain with PARSE + STRING_OP appears in both rows).
  2. For musique, the schema/coref axis on RETRIEVAL hops.

Run AFTER add_deterministic_axes.py.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MAT  = REPO / "outputs" / "updated_matrices"
DATA = REPO / "data_preprocessing" / "cleaned_trimmed_data"


def load_full(ds):    return json.loads((MAT / f"{ds}_full_binary.json").read_text())
def load_sub(ds):     return json.loads((MAT / f"{ds}_subtasks.json").read_text())
def load_cleaned(ds):
    p = DATA / f"{ds}_cleaned.{'json' if ds != 'musique' else 'jsonl'}"
    rows = (json.loads(open(p).read()) if p.suffix == ".json"
            else [json.loads(l) for l in open(p)])
    return {e["id"]: e for e in rows}


def question_score(qid, full, sub, row_idx):
    binary = np.array(sub[qid]["binary"], dtype=int)
    a = int(any(v == 1 for v in full["values"][row_idx[qid]]))
    b = int(all(np.any(binary[si] == 1) for si in range(binary.shape[0])))
    return a, b


def print_tab(rows, headers):
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*(str(c) for c in r)))


def recovery_morehopqa():
    print("=" * 110)
    print("MOREHOPQA — recovery rate of full-task failures, sliced by comp_subtype present in chain")
    print("=" * 110)
    full = load_full("morehopqa")
    sub  = load_sub("morehopqa")
    cleaned = load_cleaned("morehopqa")
    row_idx = {qid: i for i, qid in enumerate(full["rows"])}

    qids = [q for q in row_idx if q in sub and q in cleaned]
    full_fail = [q for q in qids if not question_score(q, full, sub, row_idx)[0]]
    print(f"Total: {len(qids)}  Full-task failures: {len(full_fail)}")

    # Bucket each failure by the comp_subtypes appearing in its chain
    by_subtype = defaultdict(list)   # comp_subtype → list of qids that contain it
    for qid in full_fail:
        chain = cleaned[qid]["question_decomposition"]
        present = {s.get("comp_subtype") for s in chain
                   if s.get("label") == "COMPUTATION" and s.get("comp_subtype")}
        if not present:
            by_subtype["(no COMPUTATION hop)"].append(qid)
        for st in present:
            by_subtype[st].append(qid)

    rows = []
    for st, qs in sorted(by_subtype.items(), key=lambda x: -len(x[1])):
        recovered = sum(question_score(q, full, sub, row_idx)[1] for q in qs)
        rows.append([st, len(qs), recovered, f"{100*recovered/len(qs):.1f}%"])
    print()
    print("Comp_subtype × full-task failure recovery (chain may contain multiple subtypes):")
    print_tab(rows, ["comp_subtype", "fail_chains", "recovered", "recovery %"])

    # Single-subtype chains only — for cleaner attribution
    print("\n\nSingle-subtype chains only (chain has exactly ONE comp_subtype):")
    by_single = defaultdict(list)
    for qid in full_fail:
        chain = cleaned[qid]["question_decomposition"]
        present = [s.get("comp_subtype") for s in chain
                   if s.get("label") == "COMPUTATION" and s.get("comp_subtype")]
        if not present:
            by_single["(no COMPUTATION hop)"].append(qid)
        elif len(set(present)) == 1:
            by_single[present[0]].append(qid)
    rows = []
    for st, qs in sorted(by_single.items(), key=lambda x: -len(x[1])):
        recovered = sum(question_score(q, full, sub, row_idx)[1] for q in qs)
        rows.append([st, len(qs), recovered, f"{100*recovered/len(qs):.1f}%"])
    print_tab(rows, ["sole comp_subtype", "fails", "recovered", "recovery %"])


def recovery_musique():
    print("\n\n" + "=" * 110)
    print("MUSIQUE — recovery rate of full-task failures, sliced by question composition")
    print("=" * 110)
    full = load_full("musique")
    sub  = load_sub("musique")
    cleaned = load_cleaned("musique")
    row_idx = {qid: i for i, qid in enumerate(full["rows"])}

    qids = [q for q in row_idx if q in sub and q in cleaned]
    full_fail = [q for q in qids if not question_score(q, full, sub, row_idx)[0]]
    print(f"Total: {len(qids)}  Full-task failures: {len(full_fail)}")

    # Slice: by hops with each axis
    print("\n— Per-Q: number of schema hops vs NL hops in the chain")
    by_shape = defaultdict(list)
    for qid in full_fail:
        chain = cleaned[qid]["question_decomposition"]
        n = len(chain)
        n_schema = sum(1 for s in chain if s.get("is_schema_query"))
        n_coref  = sum(1 for s in chain if s.get("coref_to_prior"))
        if n_schema == n:    bucket = "all schema"
        elif n_schema == 0:  bucket = "all NL"
        else:                bucket = "mixed schema/NL"
        by_shape[bucket].append(qid)
    rows = []
    for bucket in ["all schema", "mixed schema/NL", "all NL"]:
        qs = by_shape[bucket]
        if not qs: continue
        recovered = sum(question_score(q, full, sub, row_idx)[1] for q in qs)
        rows.append([bucket, len(qs), recovered, f"{100*recovered/len(qs):.1f}%"])
    print_tab(rows, ["chain shape", "fails", "recovered", "recovery %"])

    print("\n— Per-Q: by chain length and heterogeneity (label type-set)")
    rows = []
    for n_hops in (2, 3, 4):
        for ts_label in ("homogeneous", "heterogeneous"):
            qs = []
            for qid in full_fail:
                chain = cleaned[qid]["question_decomposition"]
                if len(chain) != n_hops: continue
                ts = {s.get("label", "UNK") for s in chain}
                if (len(ts) == 1) == (ts_label == "homogeneous"):
                    qs.append(qid)
            if not qs: continue
            recovered = sum(question_score(q, full, sub, row_idx)[1] for q in qs)
            rows.append([f"{n_hops} hops", ts_label, len(qs), recovered, f"{100*recovered/len(qs):.1f}%"])
    print_tab(rows, ["chain length", "type-set", "fails", "recovered", "recovery %"])

    print("\n— Per-Q: by retrieval_subtype presence (genealogy vs not)")
    print("  (only meaningful AFTER you have re-run classify_subtasks.py with the new prompt)")
    by_rs = defaultdict(list)
    has_rs = sum(1 for q in cleaned.values() for s in q["question_decomposition"]
                 if s.get("retrieval_subtype"))
    if has_rs:
        for qid in full_fail:
            chain = cleaned[qid]["question_decomposition"]
            has_geneal = any(s.get("retrieval_subtype") == "GENEALOGY" for s in chain)
            by_rs["has GENEALOGY hop" if has_geneal else "no GENEALOGY hop"].append(qid)
        rows = []
        for k, qs in by_rs.items():
            recovered = sum(question_score(q, full, sub, row_idx)[1] for q in qs)
            rows.append([k, len(qs), recovered, f"{100*recovered/len(qs):.1f}%"])
        print_tab(rows, ["bucket", "fails", "recovered", "recovery %"])
    else:
        print("  retrieval_subtype not yet populated — re-run classify_subtasks.py with new prompt")


def headline():
    print()
    print("#" * 110)
    print("HEADLINE — for each dataset: of the questions full-task routing CANNOT solve,")
    print("           what fraction can subtask routing recover, and where is the recovery")
    print("           concentrated by chain shape?")
    print("#" * 110)


if __name__ == "__main__":
    headline()
    recovery_morehopqa()
    recovery_musique()
