"""SQLite cache for model generations.

Keyed by (dataset, question_id, subtask_idx, model_id, prompt_hash) so:
 - repeated runs are idempotent
 - any prompt-template change forces recomputation (different hash)
 - a future swap to Hellen's DB only needs to match this interface.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  dataset       TEXT NOT NULL,
  question_id   TEXT NOT NULL,
  subtask_idx   INTEGER NOT NULL,
  model_id      TEXT NOT NULL,
  prompt_hash   TEXT NOT NULL,
  prompt        TEXT NOT NULL,
  response      TEXT,
  gold          TEXT,
  gold_aliases  TEXT,
  gen_kwargs    TEXT,
  wall_ms       INTEGER,
  created_at    TEXT,
  correct       INTEGER,
  judge_id      TEXT,
  judge_reason  TEXT,
  PRIMARY KEY (dataset, question_id, subtask_idx, model_id, prompt_hash)
);
CREATE INDEX IF NOT EXISTS idx_model_dataset ON runs(model_id, dataset);
CREATE INDEX IF NOT EXISTS idx_correct_null ON runs(dataset, model_id) WHERE correct IS NULL;
"""


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class Cache:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.db_path, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield con
        finally:
            con.close()

    def has(
        self,
        dataset: str,
        question_id: str,
        subtask_idx: int,
        model_id: str,
        p_hash: str,
    ) -> bool:
        with self._conn() as c:
            r = c.execute(
                "SELECT 1 FROM runs WHERE dataset=? AND question_id=? AND subtask_idx=? "
                "AND model_id=? AND prompt_hash=? LIMIT 1",
                (dataset, question_id, subtask_idx, model_id, p_hash),
            ).fetchone()
        return r is not None

    def upsert(
        self,
        *,
        dataset: str,
        question_id: str,
        subtask_idx: int,
        model_id: str,
        prompt: str,
        response: str,
        gold: str,
        gold_aliases: list[str],
        gen_kwargs: dict,
        wall_ms: int,
    ) -> None:
        p_hash = prompt_hash(prompt)
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO runs
                (dataset, question_id, subtask_idx, model_id, prompt_hash,
                 prompt, response, gold, gold_aliases, gen_kwargs, wall_ms, created_at,
                 correct, judge_id, judge_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        (SELECT correct FROM runs WHERE dataset=? AND question_id=? AND subtask_idx=? AND model_id=? AND prompt_hash=?),
                        (SELECT judge_id FROM runs WHERE dataset=? AND question_id=? AND subtask_idx=? AND model_id=? AND prompt_hash=?),
                        (SELECT judge_reason FROM runs WHERE dataset=? AND question_id=? AND subtask_idx=? AND model_id=? AND prompt_hash=?))
                """,
                (
                    dataset, question_id, subtask_idx, model_id, p_hash,
                    prompt, response, gold, json.dumps(gold_aliases), json.dumps(gen_kwargs),
                    wall_ms, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    dataset, question_id, subtask_idx, model_id, p_hash,
                    dataset, question_id, subtask_idx, model_id, p_hash,
                    dataset, question_id, subtask_idx, model_id, p_hash,
                ),
            )

    def mark_correct(
        self,
        *,
        dataset: str,
        question_id: str,
        subtask_idx: int,
        model_id: str,
        p_hash: str,
        correct: bool,
        judge_id: str,
        judge_reason: str,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET correct=?, judge_id=?, judge_reason=? "
                "WHERE dataset=? AND question_id=? AND subtask_idx=? AND model_id=? AND prompt_hash=?",
                (int(correct), judge_id, judge_reason,
                 dataset, question_id, subtask_idx, model_id, p_hash),
            )

    def iter_rows(self, where: str = "", params: tuple = ()) -> Iterable[dict]:
        q = "SELECT * FROM runs"
        if where:
            q += f" WHERE {where}"
        with self._conn() as c:
            cur = c.execute(q, params)
            cols = [d[0] for d in cur.description]
            for row in cur:
                yield dict(zip(cols, row))

    def count(self, where: str = "", params: tuple = ()) -> int:
        q = "SELECT COUNT(*) FROM runs"
        if where:
            q += f" WHERE {where}"
        with self._conn() as c:
            return c.execute(q, params).fetchone()[0]
