"""Run Prometheus-7B-v2.0 over exported run JSONLs and write judgments.

Input:   outputs/runs/<dataset>/<model_id>/<phase>.jsonl   (from export_runs.py)
Output:  outputs/judgments/<judge_id>/<dataset>/<model_id>/<phase>.jsonl

Each judgment line:
  {question_id, subtask_idx, model_id, dataset, judge_id,
   score, correct, feedback}

Also updates outputs/cache.db so progress survives restarts.

Usage:
  python -m eval.run_judge                              # judge every unjudged row
  python -m eval.run_judge --model qwen3-8b             # one model
  python -m eval.run_judge --dataset musique --limit 10 # 10 rows from one dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eval.cache import Cache, prompt_hash
from eval.inference import make_backend, timed_generate
from eval.prompts import judge as judge_prompts
from eval.run_inference import CACHE_DEFAULT, REPO_ROOT


JUDGE_CFG_PATH = REPO_ROOT / "eval" / "configs" / "judge.yaml"


def _load_judge_cfg() -> dict:
    return yaml.safe_load(JUDGE_CFG_PATH.read_text())["judge"]


def _iter_runs_files(runs_root: Path, model: str | None, dataset: str | None):
    for ds_dir in sorted(runs_root.iterdir()) if runs_root.exists() else []:
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        if dataset and ds_name != dataset:
            continue
        for model_dir in sorted(ds_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_id = model_dir.name
            if model and model_id != model:
                continue
            for phase_file in sorted(model_dir.glob("*.jsonl")):
                yield ds_name, model_id, phase_file


def _already_judged(out_path: Path) -> set[tuple[str, int]]:
    """Return set of (question_id, subtask_idx) already present in the judgments file."""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            r = json.loads(line)
            done.add((r["question_id"], r["subtask_idx"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-db", default=str(CACHE_DEFAULT))
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "outputs" / "runs"))
    ap.add_argument("--judgments-dir", default=str(REPO_ROOT / "outputs" / "judgments"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Cap total rows judged (smoke test).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--backend", choices=["auto", "hf", "vllm"], default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--judge-config", default=str(JUDGE_CFG_PATH),
                    help="Path to judge YAML (default: eval/configs/judge.yaml). "
                         "Set to eval/configs/judge_qwen.yaml to use Qwen JSON-style judge.")
    args = ap.parse_args()

    import yaml as _yaml
    judge_cfg = _yaml.safe_load(open(args.judge_config).read())["judge"]
    judge_id = judge_cfg["id"]
    threshold = judge_cfg.get("correct_threshold", 4)
    use_json_output = judge_cfg.get("use_json_output", False)
    if use_json_output:
        from eval.prompts import judge_qwen as judge_prompts_impl
    else:
        judge_prompts_impl = judge_prompts

    # Collect work items before loading the model so we can bail early if empty.
    work: list[tuple[str, str, Path, Path, dict]] = []
    for ds_name, model_id, runs_file in _iter_runs_files(Path(args.runs_dir), args.model, args.dataset):
        out_path = Path(args.judgments_dir) / judge_id / ds_name / model_id / runs_file.name
        done = _already_judged(out_path)
        with runs_file.open() as f:
            for line in f:
                row = json.loads(line)
                key = (row["question_id"], row["subtask_idx"])
                if key in done:
                    continue
                work.append((ds_name, model_id, runs_file, out_path, row))
                if args.limit and len(work) >= args.limit:
                    break
        if args.limit and len(work) >= args.limit:
            break

    if not work:
        print("nothing to judge (all rows already have judgments).")
        return
    print(f"judging {len(work)} rows with {judge_id}")

    model_cfg = {
        "hf_repo": judge_cfg["hf_repo"],
        "dtype": judge_cfg.get("dtype", "bfloat16"),
        "max_model_len": judge_cfg.get("max_model_len", 4096),
        "tensor_parallel_size": judge_cfg.get("tensor_parallel_size", 1),
    }
    backend = make_backend(model_cfg, backend=args.backend, device=args.device)
    backend.gen_kwargs.max_new_tokens = judge_cfg.get("max_new_tokens", 512)
    backend.gen_kwargs.temperature = 0.0
    backend.gen_kwargs.seed = 42

    # No SQLite cache write — judgments live in the JSONL tree only.

    # Extract only the ANSWER: line for the judge. No fallback — if the
    # model didn't emit ANSWER:, we mark the row incorrect without calling
    # the judge (no answer = no score).
    def _extract_answer_line(resp: str) -> str | None:
        if not resp:
            return None
        idx = resp.rfind("ANSWER:")
        if idx < 0:
            return None
        line = resp[idx:].split("\n", 1)[0].strip()
        # Cap to 1000 chars to keep judge prompts bounded (some models emit
        # multi-paragraph "ANSWER:" lines that blow past Prometheus's context).
        return line[:1000]

    # Split work: rows with ANSWER: go to the judge; rows without → auto-wrong.
    judgeable = []      # list of (orig_tuple, answer_line) for judge
    no_answer_rows = [] # list of orig_tuple for auto correct=0
    for item in work:
        row = item[4]
        ans = _extract_answer_line(row.get("response") or "")
        if ans is None:
            no_answer_rows.append(item)
        else:
            judgeable.append((item, ans))
    print(f"  judgeable (has ANSWER:): {len(judgeable)}")
    print(f"  no-answer (auto correct=0): {len(no_answer_rows)}")

    # Build judge prompts for judgeable rows only.
    prompts = []
    for item, ans in judgeable:
        row = item[4]
        p = judge_prompts_impl.build(
            dataset=row["dataset"],
            question=row["question"],
            response=ans,
            gold=row["gold"] or "",
            gold_aliases=row.get("gold_aliases") or [],
            options=row.get("options"),
        )
        prompts.append(p)

    # Emit auto correct=0 records immediately for no-answer rows.
    open_files: dict[Path, any] = {}
    for (ds_name, model_id, _rf, out_path, row) in no_answer_rows:
        rec = {
            "dataset": row["dataset"],
            "question_id": row["question_id"],
            "subtask_idx": row["subtask_idx"],
            "model_id": row["model_id"],
            "judge_id": judge_id,
            "score": None,
            "correct": 0,
            "feedback": "NO_ANSWER_LINE: model did not emit an ANSWER: line; auto-marked incorrect.",
        }
        if out_path not in open_files:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            open_files[out_path] = out_path.open("a")
        open_files[out_path].write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Batched generate for judgeable rows.
    try:
        bs = args.batch_size
        for i in range(0, len(judgeable), bs):
            chunk = judgeable[i : i + bs]
            chunk_prompts = prompts[i : i + bs]
            outs, _walls = timed_generate(backend, chunk_prompts)
            for ((item, _ans), raw) in zip(chunk, outs):
                ds_name, model_id, _rf, out_path, row = item
                if use_json_output:
                    correct_bool, feedback = judge_prompts_impl.parse_correct(raw)
                    score = None
                    correct = None if correct_bool is None else int(correct_bool)
                else:
                    score, feedback = judge_prompts.parse_score(raw)
                    correct = None if score is None else int(score >= threshold)
                record = {
                    "dataset": row["dataset"],
                    "question_id": row["question_id"],
                    "subtask_idx": row["subtask_idx"],
                    "model_id": row["model_id"],
                    "judge_id": judge_id,
                    "score": score,
                    "correct": correct,
                    "feedback": feedback if correct is not None else f"PARSE_FAILED:{feedback[:200]}",
                }
                if out_path not in open_files:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    open_files[out_path] = out_path.open("a")
                open_files[out_path].write(json.dumps(record, ensure_ascii=False) + "\n")
                # NOTE: we deliberately don't write back to the home SQLite cache
                # here. NFS + SQLite + concurrent writers from many compute nodes
                # = "disk I/O error" failures. The judgments JSONL tree is the
                # source of truth; build_matrices.py reads it directly.
            print(f"  {min(i + bs, len(work))}/{len(work)}")
    finally:
        for fh in open_files.values():
            fh.close()
        backend.close()

    print(f"done. judgments written under {args.judgments_dir}/{judge_id}/")


if __name__ == "__main__":
    main()
