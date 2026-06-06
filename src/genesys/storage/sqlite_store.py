"""SQLiteStore — structured metadata and result storage backed by SQLite.

Every pipeline run writes three things:
  - A row in `pipeline_runs` (id, name, started_at, succeeded)
  - One row per node in `node_results`  (run_id, node_id, component, status, message)
  - One row per model output in `model_outputs` (run_id, model_name, version, weights JSON, bias)

The store can also be used as a general key-value log via `save_record` /
`query` for anything else the pipeline needs to persist.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


class SQLiteStore:
    """Thread-safe SQLite storage for pipeline runs and model outputs."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, db_path: str = "genesys.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # One lock per store instance — protects multi-statement transactions
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    started_at  TEXT NOT NULL,
                    finished_at TEXT,
                    succeeded   INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS node_results (
                    id           TEXT PRIMARY KEY,
                    run_id       TEXT NOT NULL REFERENCES pipeline_runs(id),
                    node_id      TEXT NOT NULL,
                    component    TEXT NOT NULL,
                    status       TEXT NOT NULL,
                    message      TEXT,
                    recorded_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_outputs (
                    id          TEXT PRIMARY KEY,
                    run_id      TEXT NOT NULL REFERENCES pipeline_runs(id),
                    model_name  TEXT NOT NULL,
                    version     TEXT,
                    weights     TEXT,   -- JSON array
                    bias        REAL,
                    extra       TEXT,   -- JSON for anything else
                    saved_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS records (
                    id         TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    data       TEXT NOT NULL,   -- JSON
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_node_results_run  ON node_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_model_outputs_run ON model_outputs(run_id);
                CREATE INDEX IF NOT EXISTS idx_records_table     ON records(table_name);
            """)

    # ------------------------------------------------------------------
    # Pipeline run persistence
    # ------------------------------------------------------------------

    def save_run(
        self,
        name: str,
        node_results: list[dict[str, Any]],
        model_outputs: list[dict[str, Any]] | None = None,
    ) -> str:
        """Persist a complete pipeline run. Returns the new run_id."""
        run_id = str(uuid.uuid4())
        now = _now()

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO pipeline_runs (id, name, started_at, succeeded) VALUES (?, ?, ?, ?)",
                (run_id, name, now, 1),
            )

            for nr in node_results:
                conn.execute(
                    "INSERT INTO node_results "
                    "(id, run_id, node_id, component, status, message, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        run_id,
                        nr.get("node_id", ""),
                        nr.get("component", ""),
                        nr.get("status", ""),
                        nr.get("message", ""),
                        now,
                    ),
                )

            for mo in (model_outputs or []):
                conn.execute(
                    "INSERT INTO model_outputs "
                    "(id, run_id, model_name, version, weights, bias, extra, saved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        run_id,
                        mo.get("model_name", ""),
                        mo.get("version", ""),
                        json.dumps(mo.get("weights")),
                        mo.get("bias"),
                        json.dumps(mo.get("extra")) if mo.get("extra") else None,
                        now,
                    ),
                )

        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a run's metadata + node results + model outputs, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            run = dict(row)
            run["node_results"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM node_results WHERE run_id = ? ORDER BY rowid",
                    (run_id,),
                ).fetchall()
            ]
            run["model_outputs"] = [
                dict(m)
                for m in conn.execute(
                    "SELECT * FROM model_outputs WHERE run_id = ? ORDER BY rowid",
                    (run_id,),
                ).fetchall()
            ]
        return run

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent run summaries (no node/model details)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Generic record storage (for anything else)
    # ------------------------------------------------------------------

    def save_record(self, table_name: str, data: dict[str, Any]) -> str:
        """Persist an arbitrary dict under a logical table name. Returns record_id."""
        record_id = str(uuid.uuid4())
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO records (id, table_name, data, created_at) VALUES (?, ?, ?, ?)",
                (record_id, table_name, json.dumps(data), _now()),
            )
        return record_id

    def query(self, table_name: str) -> list[dict[str, Any]]:
        """Return all records saved under a logical table name."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, data, created_at FROM records WHERE table_name = ? ORDER BY rowid",
                (table_name,),
            ).fetchall()
        return [
            {"id": r["id"], "created_at": r["created_at"], **json.loads(r["data"])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent reads
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
