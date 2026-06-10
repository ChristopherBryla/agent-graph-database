"""RunStore — persistent, queryable record of every pipeline run.

Runs are first-class, addressable objects (the substrate for the platform's
north-star workflows):

  - The node dev loop replays one node against a recorded run's inputs.
  - Historical backfill fans a pipeline out over many recorded windows.
  - run_type lets downstream consumers separate 'production' (customer-grade)
    data from every team's own test/verification runs.

Schema:
  runs       — one row per run: identity, classification, status, timings,
               the request payload, and the LRO log.
  run_nodes  — one row per node: status, message, and the RESOLVED inputs and
               outputs (this is what makes node-level replay possible).

Status lifecycle:  running → completed   (pipeline finished; `succeeded`
                                          records whether all nodes passed)
                   running → error       (infrastructure failure — the
                                          pipeline itself could not run)

All methods are thread-safe: async runs complete on background threads while
API reads happen concurrently.
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Thread-safe SQLite storage for pipeline run records."""

    def __init__(self, db_path: str = "data/genesys_runs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       TEXT PRIMARY KEY,
                    runtime_type TEXT NOT NULL,
                    run_type     TEXT NOT NULL,
                    status       TEXT NOT NULL,   -- running | completed | error
                    succeeded    INTEGER,         -- NULL while running
                    request      TEXT NOT NULL,   -- JSON request payload
                    lros         TEXT,            -- JSON LRO log
                    error        TEXT,            -- set when status = error
                    started_at   TEXT NOT NULL,
                    finished_at  TEXT
                );

                CREATE TABLE IF NOT EXISTS run_nodes (
                    id          TEXT PRIMARY KEY,
                    run_id      TEXT NOT NULL REFERENCES runs(run_id),
                    node_id     TEXT NOT NULL,
                    component   TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    message     TEXT,
                    inputs      TEXT,             -- JSON resolved inputs
                    output      TEXT,             -- JSON node output
                    recorded_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_runtime  ON runs(runtime_type);
                CREATE INDEX IF NOT EXISTS idx_runs_run_type ON runs(run_type);
                CREATE INDEX IF NOT EXISTS idx_runs_started  ON runs(started_at);
                CREATE INDEX IF NOT EXISTS idx_run_nodes_run ON run_nodes(run_id);
            """)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        runtime_type: str,
        run_type: str,
        request_payload: dict[str, Any],
    ) -> None:
        """Record a new run in 'running' state."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, runtime_type, run_type, status, request, started_at) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                (run_id, runtime_type, run_type, json.dumps(request_payload), _now()),
            )

    def complete_run(self, run_id: str, result: dict[str, Any]) -> None:
        """Mark a run completed and store its full result.

        *result* is a RunResult.to_dict(): {succeeded, nodes, lros, ...}.
        """
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status='completed', succeeded=?, lros=?, finished_at=? "
                "WHERE run_id=?",
                (
                    1 if result.get("succeeded") else 0,
                    json.dumps(result.get("lros", [])),
                    now,
                    run_id,
                ),
            )
            for node in result.get("nodes", []):
                conn.execute(
                    "INSERT INTO run_nodes "
                    "(id, run_id, node_id, component, status, message, inputs, output, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        run_id,
                        node.get("node_id", ""),
                        node.get("component", ""),
                        node.get("status", ""),
                        node.get("message", ""),
                        json.dumps(node.get("inputs")),
                        json.dumps(node.get("output")),
                        now,
                    ),
                )

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as an infrastructure-level error (pipeline did not run)."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status='error', succeeded=0, error=?, finished_at=? "
                "WHERE run_id=?",
                (error, _now(), run_id),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Full run record including per-node inputs/outputs, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            node_rows = conn.execute(
                "SELECT * FROM run_nodes WHERE run_id=? ORDER BY recorded_at, node_id",
                (run_id,),
            ).fetchall()

        record = self._run_metadata(row)
        record["request"] = json.loads(row["request"])
        record["lros"] = json.loads(row["lros"]) if row["lros"] else []
        record["error"] = row["error"]
        record["nodes"] = [
            {
                "node_id": n["node_id"],
                "component": n["component"],
                "status": n["status"],
                "message": n["message"],
                "inputs": json.loads(n["inputs"]) if n["inputs"] else None,
                "output": json.loads(n["output"]) if n["output"] else None,
            }
            for n in node_rows
        ]
        return record

    def list_runs(
        self,
        runtime_type: str | None = None,
        run_type: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Run metadata (no node payloads), newest first, with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if runtime_type is not None:
            clauses.append("runtime_type = ?")
            params.append(runtime_type)
        if run_type is not None:
            clauses.append("run_type = ?")
            params.append(run_type)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 500)))

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs {where} ORDER BY started_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._run_metadata(r) for r in rows]

    @staticmethod
    def _run_metadata(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "runtime_type": row["runtime_type"],
            "run_type": row["run_type"],
            "status": row["status"],
            "succeeded": None if row["succeeded"] is None else bool(row["succeeded"]),
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
