"""Build the artifacts described in Context.md from the judgments.

Sources:
  outputs/judgments/<judge_id>/<dataset>/<model>/{full,subtask}.jsonl
  (also reads outputs/cache.db for any cached judgments — judgments tree wins)

Outputs (Git LFS via *.json rule):

  outputs/accuracy_table.json
    Per (dataset, model_id): n_total, n_judged, n_correct, accuracy.

  outputs/matrices/<dataset>_full_binary.json
    Full-question Nq × Nm matrix:
      {"rows": [question_id, ...],         # length Nq, ordered
       "cols": [model_id, ...],            # length Nm
       "values": [[0|1|null, ...], ...]}   # values[i][j] = correctness of model j on question i

  outputs/matrices/<dataset>_full_per_active_b.json
  outputs/matrices/<dataset>_full_per_total_b.json
    Same shape as binary, but values divided by the column model's params_*_b.

  outputs/matrices/<dataset>_subtasks.json
    Per-question subtask matrices, keyed by question_id:
      {"<question_id>":
          {"rows": [0, 1, 2, ...],         # subtask_idx labels (0-based)
           "cols": [model_id, ...],
           "binary": [[...], ...],
           "per_active_b": [[...], ...],
           "per_total_b": [[...], ...]}}

  outputs/matrices/model_params.json
    The model_id ↔ params_active_b / params_total_b key for traceability.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from eval.cache import Cache
from eval.run_inference import CACHE_DEFAULT, REPO_ROOT, load_cfg


JUDGMENTS_ROOT = REPO_ROOT / "outputs" / "judgments"


def _model_params(models_cfg: dict) -> pd.DataFrame:
    rows = [
        {
            "model_id": m["id"],
            "params_active_b": m["params_active_b"],
            "params_total_b": m["params_total_b"],
        }
        for m in models_cfg["models"]
    ]
    return pd.DataFrame(rows)


def _judgments_df(judge_id: str | None = None) -> pd.DataFrame:
    """Walk outputs/judgments/<judge>/**/*.jsonl into a long DataFrame."""
    rows = []
    if not JUDGMENTS_ROOT.exists():
        return pd.DataFrame(columns=["dataset", "question_id", "subtask_idx", "model_id", "correct", "judge_id"])
    judge_dirs = [JUDGMENTS_ROOT / judge_id] if judge_id else list(JUDGMENTS_ROOT.iterdir())
    for jd in judge_dirs:
        if not jd.is_dir():
            continue
        for jl in jd.rglob("*.jsonl"):
            with jl.open() as f:
                for line in f:
                    r = json.loads(line)
                    r.setdefault("judge_id", jd.name)
                    rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["dataset", "question_id", "subtask_idx", "model_id", "correct", "judge_id"])
    return pd.DataFrame(rows)


def _cache_df(cache: Cache) -> pd.DataFrame:
    rows = list(cache.iter_rows("correct IS NOT NULL"))
    if not rows:
        return pd.DataFrame(columns=["dataset", "question_id", "subtask_idx", "model_id", "correct"])
    return pd.DataFrame(rows)[["dataset", "question_id", "subtask_idx", "model_id", "correct"]]


def _combined_df(cache_db: Path, judge_id: str | None) -> pd.DataFrame:
    """Merge judgments-tree + cache-table judgments, preferring the JSONL tree."""
    j = _judgments_df(judge_id)
    if not j.empty:
        j = j[["dataset", "question_id", "subtask_idx", "model_id", "correct"]]
    c = _cache_df(Cache(cache_db))
    if j.empty:
        return c
    if c.empty:
        return j
    # Prefer JSONL tree on conflict (drop cache rows that have a corresponding JSONL row).
    key = ["dataset", "question_id", "subtask_idx", "model_id"]
    merged = pd.concat([j, c.merge(j[key], on=key, how="left", indicator=True).query('_merge == "left_only"').drop(columns="_merge")], ignore_index=True)
    return merged


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset", "model_id", "n_total", "n_judged", "n_correct", "accuracy"])
    g = df.groupby(["dataset", "model_id"], as_index=False).agg(
        n_total=("correct", "size"),
        n_judged=("correct", lambda s: s.notna().sum()),
        n_correct=("correct", lambda s: (s == 1).sum()),
    )
    g["accuracy"] = g.apply(
        lambda r: (r["n_correct"] / r["n_judged"]) if r["n_judged"] else None, axis=1
    )
    return g


def _matrix_from_pivot(pivot: pd.DataFrame) -> dict:
    """Convert a pandas pivot to {rows, cols, values} dict, preserving NaN as None."""
    return {
        "rows": pivot.index.tolist(),
        "cols": pivot.columns.tolist(),
        "values": [[None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v)) for v in row] for row in pivot.values],
    }


def _matrix_from_pivot_float(pivot: pd.DataFrame) -> dict:
    return {
        "rows": pivot.index.tolist(),
        "cols": pivot.columns.tolist(),
        "values": [[None if pd.isna(v) else float(v) for v in row] for row in pivot.values],
    }


def build_full_matrices(df: pd.DataFrame, params_df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    sel = df[(df["dataset"] == dataset) & (df["subtask_idx"] == -1)].copy()
    if sel.empty:
        return
    binary_pivot = sel.pivot_table(index="question_id", columns="model_id", values="correct", aggfunc="first").sort_index().sort_index(axis=1)
    binary = _matrix_from_pivot(binary_pivot)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{dataset}_full_binary.json").write_text(json.dumps(binary, indent=2))

    pmap = params_df.set_index("model_id")
    for col_field, file_suffix in [("params_active_b", "per_active_b"), ("params_total_b", "per_total_b")]:
        scaled_pivot = binary_pivot.copy()
        for model in scaled_pivot.columns:
            denom = pmap.loc[model, col_field] if model in pmap.index else None
            if denom and denom > 0:
                scaled_pivot[model] = scaled_pivot[model].astype("float64") / float(denom)
            else:
                scaled_pivot[model] = float("nan")
        (out_dir / f"{dataset}_full_{file_suffix}.json").write_text(
            json.dumps(_matrix_from_pivot_float(scaled_pivot), indent=2)
        )
    print(f"  wrote {dataset} full matrices ({binary_pivot.shape[0]} qids × {binary_pivot.shape[1]} models)")


def build_subtask_matrices(df: pd.DataFrame, params_df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    sel = df[(df["dataset"] == dataset) & (df["subtask_idx"] >= 0)].copy()
    if sel.empty:
        return
    pmap = params_df.set_index("model_id")
    per_q: dict[str, dict] = {}
    for qid, sub in sel.groupby("question_id"):
        binary_pivot = sub.pivot_table(index="subtask_idx", columns="model_id", values="correct", aggfunc="first").sort_index().sort_index(axis=1)
        entry = {
            "rows": [int(i) for i in binary_pivot.index.tolist()],
            "cols": binary_pivot.columns.tolist(),
            "binary": _matrix_from_pivot(binary_pivot)["values"],
        }
        for col_field, key in [("params_active_b", "per_active_b"), ("params_total_b", "per_total_b")]:
            scaled = binary_pivot.copy()
            for model in scaled.columns:
                denom = pmap.loc[model, col_field] if model in pmap.index else None
                if denom and denom > 0:
                    scaled[model] = scaled[model].astype("float64") / float(denom)
                else:
                    scaled[model] = float("nan")
            entry[key] = _matrix_from_pivot_float(scaled)["values"]
        per_q[qid] = entry
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{dataset}_subtasks.json").write_text(json.dumps(per_q, indent=2))
    print(f"  wrote {dataset} subtask matrices ({len(per_q)} questions, {sum(len(v['rows']) for v in per_q.values())} subtask rows total)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-db", default=str(CACHE_DEFAULT))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--judge-id", default=None,
                    help="Restrict to one judge subdir under outputs/judgments/. Default: all judges combined (last write wins).")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "matrices").mkdir(parents=True, exist_ok=True)

    models_cfg, _ = load_cfg()
    params_df = _model_params(models_cfg)
    (out / "matrices" / "model_params.json").write_text(
        json.dumps(params_df.to_dict(orient="records"), indent=2)
    )

    df = _combined_df(Path(args.cache_db), args.judge_id)
    acc = accuracy_table(df)
    acc.to_json(out / "accuracy_table.json", orient="records", indent=2)
    print(f"wrote accuracy_table.json ({len(acc)} rows)")

    for ds in (df["dataset"].dropna().unique() if not df.empty else []):
        build_full_matrices(df, params_df, ds, out / "matrices")
        build_subtask_matrices(df, params_df, ds, out / "matrices")


if __name__ == "__main__":
    main()
