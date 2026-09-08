#!/usr/bin/env python3
"""
Statistical significance tests for oracle coverage comparisons.

Tests:
  1. McNemar's test: does the chained subtask oracle cover significantly
     more questions than the full-task oracle?
  2. Two-proportion z-test + Fisher's exact test: does the chained oracle
     recover heterogeneous failures at a significantly higher rate than
     homogeneous failures?

Usage:
    python -m router_analysis.oracle_significance_tests
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parent.parent

MODEL_SETS = [
    ("small4",   "4-model set"),
    ("no_qwen3", "8-model set"),
]

SCOPES = [
    ("morehopqa", "MoreHopQA"),
    ("musique",   "MuSiQue"),
]


def mcnemar(n_full_only: int, n_chain_only: int, label: str):
    """McNemar's test with continuity correction."""
    discordant = n_full_only + n_chain_only
    if discordant == 0:
        print(f"  {label}: no discordant pairs — test not applicable")
        return
    chi2 = (abs(n_full_only - n_chain_only) - 1) ** 2 / discordant
    p = stats.chi2.sf(chi2, df=1)
    direction = "chained > full" if n_chain_only > n_full_only else "full > chained"
    print(f"  {label}")
    print(f"    full-only={n_full_only}  chained-only={n_chain_only}")
    print(f"    McNemar χ²={chi2:.3f}  p={p:.4g}  ({direction})")


def two_prop_and_fisher(n1: int, k1: int, n2: int, k2: int,
                        label1: str, label2: str, context: str):
    """
    Two-proportion z-test and Fisher's exact test.
    n1/n2 = group sizes, k1/k2 = successes (recoveries).
    """
    if n1 == 0 or n2 == 0:
        print(f"  {context}: skipped (empty group)")
        return
    p1, p2 = k1 / n1, k2 / n2
    # Fisher's exact
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    _, p_fisher = fisher_exact(table, alternative="two-sided")
    # Two-proportion z-test
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    p_z = 2 * stats.norm.sf(abs(z))
    print(f"  {context}: {label1} vs {label2}")
    print(f"    {label1}: {k1}/{n1} = {p1*100:.1f}%")
    print(f"    {label2}: {k2}/{n2} = {p2*100:.1f}%")
    print(f"    Two-prop z={z:.3f}  p={p_z:.4g}")
    print(f"    Fisher's exact p={p_fisher:.4g}")


def analyse(ms_key: str, ms_label: str, data: dict):
    print(f"\n{'='*60}")
    print(f"  {ms_label}")
    print(f"{'='*60}")

    for scope_key, scope_label in SCOPES:
        d = data[scope_key]
        print(f"\n  --- {scope_label} ---")

        all_n      = d["all"]["n"]
        full_acc   = d["all"]["full"]
        chain_acc  = d["all"]["strict"]

        n_full_ok  = round(full_acc  * all_n)
        n_chain_ok = round(chain_acc * all_n)

        # Failures under full-task oracle
        n_full_fail = d["full_fail"]["n"]
        chain_on_fail = d["full_fail"]["strict"]   # recovery rate
        n_chain_recovers = round(chain_on_fail * n_full_fail)

        # Chained correct on full-task successes
        # chain_ok_total = chain_on_successes + chain_on_failures
        n_chain_on_success = n_chain_ok - n_chain_recovers
        n_full_only  = n_full_ok - n_chain_on_success   # full correct, chain wrong
        n_chain_only = n_chain_recovers                  # chain correct, full wrong

        print(f"\n  [1] McNemar — chained vs. full oracle coverage")
        mcnemar(max(n_full_only, 0), max(n_chain_only, 0),
                f"{ms_label} / {scope_label}")

        # Het vs hom recovery rates within full-task failures
        # Use complex-question failures (het_full_fail / hom_full_fail)
        het_fail = d.get("het_full_fail", {})
        hom_fail = d.get("hom_full_fail", {})

        n_het = het_fail.get("n", 0)
        n_hom = hom_fail.get("n", 0)
        k_het = round(het_fail.get("strict", 0) * n_het) if n_het else 0
        k_hom = round(hom_fail.get("strict", 0) * n_hom) if n_hom else 0

        print(f"\n  [2] Het vs. hom failure recovery")
        two_prop_and_fisher(n_het, k_het, n_hom, k_hom,
                            "Heterogeneous failures",
                            "Homogeneous failures",
                            f"{ms_label} / {scope_label}")

        # R+C vs. R+Reasoning (within heterogeneous questions)
        rc  = d.get("het_RC",   {})
        rr  = d.get("het_RRsn", {})
        n_rc = rc.get("n", 0);  n_rr = rr.get("n", 0)
        # "strict" here is chained accuracy on ALL questions in group (not just failures)
        k_rc = round(rc.get("strict", 0) * n_rc) if n_rc else 0
        k_rr = round(rr.get("strict", 0) * n_rr) if n_rr else 0

        print(f"\n  [3] R+C vs. R+Reasoning chained coverage")
        two_prop_and_fisher(n_rc, k_rc, n_rr, k_rr,
                            "R+C", "R+Reasoning",
                            f"{ms_label} / {scope_label}")


def main():
    for ms_key, ms_label in MODEL_SETS:
        path = REPO / "outputs" / f"oracle_analysis_{ms_key}.json"
        data = json.loads(path.read_text())
        analyse(ms_key, ms_label, data)

    print("\nSignificance thresholds: * p<0.05  ** p<0.01  *** p<0.001")


if __name__ == "__main__":
    main()
