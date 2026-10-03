"""recorder.py — Trace recorder and content-hashed call cache for Black Box."""
from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

DB_PATH = Path("data/blackbox.db")

# ── schema ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    question_id TEXT,
    question TEXT,
    gold TEXT,
    final_answer TEXT,
    success INTEGER,
    fault_type TEXT,
    fault_step INTEGER,
    parent_run_id TEXT,
    split TEXT,
    n_reused INTEGER,
    n_reexecuted INTEGER,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT,
    step_idx INTEGER,
    type TEXT,
    input_json TEXT,
    output_json TEXT,
    state_before_json TEXT,
    parents_json TEXT,
    cache_key TEXT,
    cache_hit INTEGER,
    latency_ms INTEGER,
    error TEXT,
    PRIMARY KEY (run_id, step_idx)
);

CREATE TABLE IF NOT EXISTS cache (
    cache_key TEXT PRIMARY KEY,
    output_json TEXT
);
"""


@contextmanager
def _get_conn() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success (roll back on error) and always close it."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript(SCHEMA)


# ── cache ────────────────────────────────────────────────────────────────────

PROMPT_VERSION = "v1"


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def make_cache_key(step_type: str, model: str, input_data: Any) -> str:
    raw = f"{step_type}|{PROMPT_VERSION}|{model}|{_canonical(input_data)}"
    return hashlib.sha1(raw.encode()).hexdigest()


def cache_get(key: str) -> tuple[bool, Any]:
    with _get_conn() as conn:
        row = conn.execute("SELECT output_json FROM cache WHERE cache_key = ?", (key,)).fetchone()
    if row is not None:
        return True, json.loads(row["output_json"])
    return False, None


def cache_set(key: str, output: Any) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, output_json) VALUES (?, ?)",
            (key, json.dumps(output, ensure_ascii=False)),
        )


# ── step recorder ────────────────────────────────────────────────────────────


def record_step(
    run_id: str,
    idx: int,
    step_type: str,
    input_data: Any,
    fn: Callable[[Any], Any],
    state_before: dict,
    parents: list[int],
    model: str = "tool",
    override: Any = None,
    recorded: dict | None = None,
) -> Any:
    """Execute one agent step, managing cache and recording.

    Precedence: `override` (a patched output) > `recorded` (the original run's step,
    reused only if this step's input is unchanged, so a replay reproduces the original
    run exactly, injected faults included) > cache > real call.
    """
    cache_key = make_cache_key(step_type, model, input_data)
    cache_hit = 0
    error = None
    output = None

    t0 = time.monotonic()
    try:
        if override is not None:
            output = override
            cache_hit = 0
        elif recorded is not None and _canonical(recorded["input"]) == _canonical(input_data):
            output = recorded["output"]
            error = recorded.get("error")
            cache_hit = 1
        else:
            is_hit, cached = cache_get(cache_key)
            if is_hit:
                output = cached
                cache_hit = 1
            else:
                output = fn(input_data)
                cache_set(cache_key, output)
                cache_hit = 0
    except Exception as exc:
        error = str(exc)
        output = None
    latency_ms = int((time.monotonic() - t0) * 1000)

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO steps
              (run_id, step_idx, type, input_json, output_json, state_before_json,
               parents_json, cache_key, cache_hit, latency_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, idx, step_type,
                json.dumps(input_data, ensure_ascii=False),
                json.dumps(output, ensure_ascii=False),
                json.dumps(state_before, ensure_ascii=False),
                json.dumps(parents, ensure_ascii=False),
                cache_key, cache_hit, latency_ms, error,
            ),
        )
    return output


# ── run management ───────────────────────────────────────────────────────────


def save_run(
    run_id: str,
    question_id: str,
    question: str,
    gold: str,
    final_answer: str | None,
    success: bool,
    split: str,
    fault_type: str | None = None,
    fault_step: int | None = None,
    parent_run_id: str | None = None,
) -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
              (run_id, question_id, question, gold, final_answer, success,
               fault_type, fault_step, parent_run_id, split, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, question_id, question, gold,
                final_answer, int(success),
                fault_type, fault_step, parent_run_id, split,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )


def update_run_counts(run_id: str, n_reused: int, n_reexecuted: int) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE runs SET n_reused = ?, n_reexecuted = ? WHERE run_id = ?",
            (n_reused, n_reexecuted, run_id),
        )


def get_run(run_id: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_steps(run_id: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_idx", (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_runs(split: str | None = None, success: int | None = None) -> list[dict]:
    with _get_conn() as conn:
        clauses, params = [], []
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if success is not None:
            clauses.append("success = ?")
            params.append(success)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM runs {where} ORDER BY created_at DESC", params
        ).fetchall()
    return [dict(r) for r in rows]


def delete_run(run_id: str) -> None:
    """Remove a run and its steps (used before re-recording a replay with a fixed ID)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM steps WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def cache_stats(run_id: str) -> dict[str, int]:
    """Return count of reused (cache hit) and re-executed steps for a run."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT SUM(cache_hit) as hits, SUM(1-cache_hit) as misses FROM steps WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return {"reused": row["hits"] or 0, "reexecuted": row["misses"] or 0}
