"""Build matrix artifacts from judgments_4k_qwen into outputs/updated-matrices/.

Sources:
  outputs/judgments_4k_qwen/<judge_id>/<dataset>/<model>/{full,subtask}.jsonl

Outputs (same schema as build_matrices.py):

  outputs/updated-matrices/model_params.json
    The model_id ↔ params_active_b / params_total_b key for traceability.

  outputs/updated-matrices/<dataset>_full_binary.json
    Full-question Nq × Nm matrix:
      {"rows": [question_id, ...],         # length Nq, ordered
       "cols": [model_id, ...],            # length Nm
       "values": [[0|1|null, ...], ...]}   # values[i][j] = correctness of model j on question i

  outputs/updated-matrices/<dataset>_full_per_active_b.json
  outputs/updated-matrices/<dataset>_full_per_total_b.json
    Same shape as binary, but values divided by the column model's params_*_b.

  outputs/updated-matrices/<dataset>_subtasks.json
    Per-question subtask matrices, keyed by question_id:
      {"<question_id>":
          {"rows": [0, 1, 2, ...],         # subtask_idx labels (0-based)
           "cols": [model_id, ...],
           "binary": [[...], ...],
           "per_active_b": [[...], ...],
           "per_total_b": [[...], ...]}}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.run_inference import REPO_ROOT, load_cfg


DEFAULT_JUDGMENTS_ROOT = REPO_ROOT / "outputs" / "judgments_4k_qwen"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"


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


def _judgments_df(judgments_root: Path, judge_id: str | None = None) -> pd.DataFrame:
    """Walk <judgments_root>/<judge>/**/*.jsonl into a long DataFrame."""
    rows = []
    if not judgments_root.exists():
        print(f"[warn] judgments root not found: {judgments_root}")
        return pd.DataFrame(columns=["dataset", "question_id", "subtask_idx", "model_id", "correct", "judge_id"])

    judge_dirs = [judgments_root / judge_id] if judge_id else list(judgments_root.iterdir())
    for jd in judge_dirs:
        if not jd.is_dir():
            continue
        for jl in jd.rglob("*.jsonl"):
            with jl.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    r.setdefault("judge_id", jd.name)
                    rows.append(r)

    if not rows:
        return pd.DataFrame(columns=["dataset", "question_id", "subtask_idx", "model_id", "correct", "judge_id"])
    return pd.DataFrame(rows)


def _matrix_from_pivot(pivot: pd.DataFrame) -> dict:
    """Convert a pandas pivot to {rows, cols, values} dict, preserving NaN as None."""
    return {
        "rows": pivot.index.tolist(),
        "cols": pivot.columns.tolist(),
        "values": [
            [None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v)) for v in row]
            for row in pivot.values
        ],
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
        print(f"  [skip] no full-question rows for dataset={dataset}")
        return
    binary_pivot = (
        sel.pivot_table(index="question_id", columns="model_id", values="correct", aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )
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
        print(f"  [skip] no subtask rows for dataset={dataset}")
        return
    pmap = params_df.set_index("model_id")
    per_q: dict[str, dict] = {}
    for qid, sub in sel.groupby("question_id"):
        binary_pivot = (
            sub.pivot_table(index="subtask_idx", columns="model_id", values="correct", aggfunc="first")
            .sort_index()
            .sort_index(axis=1)
        )
        entry: dict = {
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
    print(
        f"  wrote {dataset} subtask matrices "
        f"({len(per_q)} questions, {sum(len(v['rows']) for v in per_q.values())} subtask rows total)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build updated matrix artifacts from judgments_4k_qwen."
    )
    ap.add_argument(
        "--judgments-root",
        default=str(DEFAULT_JUDGMENTS_ROOT),
        help="Root directory containing judge subdirectories with JSONL files. "
             f"Default: {DEFAULT_JUDGMENTS_ROOT}",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Root output directory. Matrices go into <out-dir>/updated-matrices/. Default: {DEFAULT_OUT_DIR}",
    )
    ap.add_argument(
        "--judge-id",
        default=None,
        help="Restrict to one judge subdir. Default: all judges combined.",
    )
    args = ap.parse_args()

    judgments_root = Path(args.judgments_root)
    out = Path(args.out_dir)
    matrices_dir = out / "updated-matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    models_cfg, _ = load_cfg()
    params_df = _model_params(models_cfg)
    (matrices_dir / "model_params.json").write_text(
        json.dumps(params_df.to_dict(orient="records"), indent=2)
    )
    print(f"wrote model_params.json ({len(params_df)} models)")

    df = _judgments_df(judgments_root, args.judge_id)
    if df.empty:
        print("No judgment rows found — exiting.")
        return

    df = df[["dataset", "question_id", "subtask_idx", "model_id", "correct"]].copy()

    for ds in sorted(df["dataset"].dropna().unique()):
        print(f"building matrices for dataset: {ds}")
        build_full_matrices(df, params_df, ds, matrices_dir)
        build_subtask_matrices(df, params_df, ds, matrices_dir)

    print("done.")


if __name__ == "__main__":
    main()
