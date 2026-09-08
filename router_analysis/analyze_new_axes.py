"""
Cross-tab the new deterministic axes against routing performance to see if
any of them creates a meaningful per-bucket recovery-rate split.

Run AFTER add_deterministic_axes.py.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
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


def section(label):
    print()
    print("=" * 110)
    print(label)
    print("=" * 110)


def print_tab(rows, headers):
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*(str(c) for c in r)))


# ---------------------------------------------------------------------------
# 1) morehopqa COMPUTATION subtype × per-hop coverage
# ---------------------------------------------------------------------------

def morehopqa_comp_subtype():
    section("morehopqa — per-hop coverage by comp_subtype")
    sub = load_sub("morehopqa")
    cleaned = load_cleaned("morehopqa")
    cols = sub[next(iter(sub))]["cols"]

    by_sub = defaultdict(lambda: [0, 0, defaultdict(int)])  # [n_subs, n_any_correct, per_model_wins]
    for qid, q in sub.items():
        binary = np.array(q["binary"], dtype=int)
        labels = cleaned[qid]["question_decomposition"]
        for si, s in enumerate(labels):
            if s.get("label") != "COMPUTATION": continue
            stype = s.get("comp_subtype", "OTHER")
            row = binary[si]
            by_sub[stype][0] += 1
            if row.sum() > 0:
                by_sub[stype][1] += 1
            for mi, m in enumerate(cols):
                if row[mi] == 1:
                    by_sub[stype][2][m] += 1

    rows = []
    for stype, (n, ok, wins) in sorted(by_sub.items(), key=lambda x: -x[1][0]):
        leader, lead_n = max(wins.items(), key=lambda x: x[1]) if wins else ("—", 0)
        rows.append([stype, n, f"{100*ok/n:.1f}%", f"{leader} ({100*lead_n/n:.1f}%)"])
    print_tab(rows, ["comp_subtype", "n_hops", "any-correct", "leader (per-hop accuracy)"])


# ---------------------------------------------------------------------------
# 2) morehopqa question-level recovery, partitioned by which comp_subtypes appear
# ---------------------------------------------------------------------------

def morehopqa_question_by_subtype():
    section("morehopqa — full-task failure recovery by COMPUTATION subtype set in chain")
    full = load_full("morehopqa")
    sub  = load_sub("morehopqa")
    cleaned = load_cleaned("morehopqa")
    row_idx = {qid: i for i, qid in enumerate(full["rows"])}

    by_set = defaultdict(list)
    for qid in row_idx:
        if qid not in sub or qid not in cleaned: continue
        types = frozenset(s.get("comp_subtype")
                          for s in cleaned[qid]["question_decomposition"]
                          if s.get("label") == "COMPUTATION" and s.get("comp_subtype"))
        if not types: continue
        by_set[types].append(qid)

    rows = []
    for types, qids in sorted(by_set.items(), key=lambda x: -len(x[1])):
        if len(qids) < 20: continue
        a_b = [question_score(q, full, sub, row_idx) for q in qids]
        n = len(qids)
        a = sum(x[0] for x in a_b) / n
        b = sum(x[1] for x in a_b) / n
        fails = [q for q, ab in zip(qids, a_b) if ab[0] == 0]
        recov = sum(question_score(q, full, sub, row_idx)[1] for q in fails)
        recov_rate = (recov / len(fails)) if fails else 0.0
        rows.append([
            ", ".join(sorted(types)) if len(types) <= 3 else f"{len(types)}-mix",
            n, f"{100*a:.1f}%", f"{100*b:.1f}%", f"{100*(b-a):+.1f}%",
            len(fails), recov, f"{100*recov_rate:.1f}%",
        ])
    print_tab(rows, ["comp_subtype set", "n", "A", "B", "B-A", "fails", "recovered", "recovery %"])


# ---------------------------------------------------------------------------
# 3) musique RETRIEVAL split by is_schema_query × coref_to_prior
# ---------------------------------------------------------------------------

def musique_retrieval_axes():
    section("musique — RETRIEVAL hop coverage by is_schema_query × coref_to_prior")
    sub = load_sub("musique")
    cleaned = load_cleaned("musique")
    cols = sub[next(iter(sub))]["cols"]

    by_axis = defaultdict(lambda: [0, 0, defaultdict(int)])  # [n, ok, per_model_correct]
    for qid, q in sub.items():
        if qid not in cleaned: continue
        binary = np.array(q["binary"], dtype=int)
        for si, s in enumerate(cleaned[qid]["question_decomposition"]):
            if s.get("label") != "RETRIEVAL": continue
            key = (s.get("is_schema_query", False), s.get("coref_to_prior", False))
            row = binary[si]
            by_axis[key][0] += 1
            if row.sum() > 0:
                by_axis[key][1] += 1
            for mi, m in enumerate(cols):
                if row[mi] == 1:
                    by_axis[key][2][m] += 1

    rows = []
    for (schema, coref), (n, ok, wins) in sorted(by_axis.items()):
        if not wins: continue
        ranking = sorted(wins.items(), key=lambda x: -x[1])
        leader, ln = ranking[0]
        second, sn = ranking[1] if len(ranking) > 1 else ("—", 0)
        rows.append([
            f"schema={schema}", f"coref={coref}",
            n, f"{100*ok/n:.1f}%",
            f"{leader} ({100*ln/n:.1f}%)",
            f"{second} ({100*sn/n:.1f}%)",
        ])
    print_tab(rows, ["axis_a", "axis_b", "n", "any-correct", "leader", "runner-up"])


# ---------------------------------------------------------------------------
# 4) musique question-level recovery split by majority-schema vs majority-NL
# ---------------------------------------------------------------------------

def musique_question_by_schema():
    section("musique — full-task failure recovery split by question composition")
    full = load_full("musique")
    sub  = load_sub("musique")
    cleaned = load_cleaned("musique")
    row_idx = {qid: i for i, qid in enumerate(full["rows"])}

    buckets = {
        "all schema":  [],
        "mostly schema":  [],
        "mostly NL":  [],
        "all NL":  [],
    }
    for qid in row_idx:
        if qid not in sub or qid not in cleaned: continue
        chain = cleaned[qid]["question_decomposition"]
        n = len(chain)
        n_schema = sum(1 for s in chain if s.get("is_schema_query"))
        if   n_schema == n:   buckets["all schema"].append(qid)
        elif n_schema > n//2: buckets["mostly schema"].append(qid)
        elif n_schema == 0:   buckets["all NL"].append(qid)
        else:                 buckets["mostly NL"].append(qid)

    rows = []
    for name, qids in buckets.items():
        if not qids: continue
        a_b = [question_score(q, full, sub, row_idx) for q in qids]
        n = len(qids)
        a = sum(x[0] for x in a_b) / n
        b = sum(x[1] for x in a_b) / n
        fails = [q for q, ab in zip(qids, a_b) if ab[0] == 0]
        recov = sum(question_score(q, full, sub, row_idx)[1] for q in fails)
        recov_rate = (recov / len(fails)) if fails else 0.0
        rows.append([name, n, f"{100*a:.1f}%", f"{100*b:.1f}%", f"{100*(b-a):+.1f}%",
                     len(fails), recov, f"{100*recov_rate:.1f}%"])
    print_tab(rows, ["bucket", "n", "A", "B", "B-A", "fails", "recovered", "recovery %"])


# ---------------------------------------------------------------------------
# 5) Per-hop_role accuracy by dataset
# ---------------------------------------------------------------------------

def hop_role_table():
    section("hop_role × any-correct (all datasets)")
    rows = []
    for ds in ["morehopqa", "musique", "stepcot"]:
        sub = load_sub(ds)
        cleaned = load_cleaned(ds)
        chain_key = "vqa_chain" if ds == "stepcot" else "question_decomposition"
        by_role = defaultdict(lambda: [0, 0])
        for qid, q in sub.items():
            if qid not in cleaned: continue
            binary = np.array(q["binary"], dtype=int)
            for si, s in enumerate(cleaned[qid][chain_key]):
                role = s.get("hop_role", "—")
                by_role[role][0] += 1
                if binary[si].sum() > 0:
                    by_role[role][1] += 1
        for role in ["seed", "bridge", "final"]:
            if role not in by_role: continue
            n, ok = by_role[role]
            rows.append([ds, role, n, f"{100*ok/n:.1f}%"])
    print_tab(rows, ["dataset", "hop_role", "n_hops", "any-correct"])


if __name__ == "__main__":
    morehopqa_comp_subtype()
    morehopqa_question_by_subtype()
    musique_retrieval_axes()
    musique_question_by_schema()
    hop_role_table()
