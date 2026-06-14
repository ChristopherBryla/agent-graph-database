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
import math
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
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
                    parent_run_id TEXT,
                    backfill_id  TEXT,
                    backfill_window_id TEXT,
                    window_start TEXT,
                    window_end   TEXT,
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
                    inputs_lossy INTEGER NOT NULL DEFAULT 0,
                    started_at  REAL,
                    finished_at REAL,
                    inputs      TEXT,             -- JSON resolved inputs
                    output      TEXT,             -- JSON node output
                    recorded_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_runtime  ON runs(runtime_type);
                CREATE INDEX IF NOT EXISTS idx_runs_run_type ON runs(run_type);
                CREATE INDEX IF NOT EXISTS idx_runs_parent   ON runs(parent_run_id);
                CREATE INDEX IF NOT EXISTS idx_runs_backfill ON runs(backfill_id);
                CREATE INDEX IF NOT EXISTS idx_runs_started  ON runs(started_at);
                CREATE INDEX IF NOT EXISTS idx_run_nodes_run ON run_nodes(run_id);

                CREATE TABLE IF NOT EXISTS backfills (
                    backfill_id          TEXT PRIMARY KEY,
                    runtime_type         TEXT NOT NULL,
                    status               TEXT NOT NULL,
                    run_type             TEXT NOT NULL,
                    failure_mode         TEXT NOT NULL,
                    max_concurrent_runs  INTEGER NOT NULL,
                    total_windows        INTEGER NOT NULL,
                    succeeded_windows    INTEGER NOT NULL DEFAULT 0,
                    failed_windows       INTEGER NOT NULL DEFAULT 0,
                    started_at           TEXT NOT NULL,
                    finished_at          TEXT,
                    error                TEXT
                );

                CREATE TABLE IF NOT EXISTS schedules (
                    schedule_id         TEXT PRIMARY KEY,
                    runtime_type        TEXT NOT NULL,
                    payload             TEXT NOT NULL,
                    run_type            TEXT NOT NULL,
                    interval_seconds    INTEGER NOT NULL,
                    enabled             INTEGER NOT NULL DEFAULT 1,
                    overlap_policy      TEXT NOT NULL,
                    next_run_at         TEXT NOT NULL,
                    last_run_at         TEXT,
                    last_run_id         TEXT,
                    last_error          TEXT,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_schedules_next_run ON schedules(next_run_at);
                CREATE INDEX IF NOT EXISTS idx_schedules_enabled ON schedules(enabled);
            """)
            self._ensure_column(conn, "runs", "parent_run_id", "TEXT")
            self._ensure_column(conn, "runs", "backfill_id", "TEXT")
            self._ensure_column(conn, "runs", "backfill_window_id", "TEXT")
            self._ensure_column(conn, "runs", "window_start", "TEXT")
            self._ensure_column(conn, "runs", "window_end", "TEXT")
            self._ensure_column(conn, "run_nodes", "inputs_lossy", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_nodes", "started_at", "REAL")
            self._ensure_column(conn, "run_nodes", "finished_at", "REAL")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        runtime_type: str,
        run_type: str,
        request_payload: dict[str, Any],
        parent_run_id: str | None = None,
        backfill_id: str | None = None,
        backfill_window_id: str | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> None:
        """Record a new run in 'running' state."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO runs ("
                "run_id, runtime_type, run_type, parent_run_id, backfill_id, backfill_window_id, "
                "window_start, window_end, status, request, started_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)",
                (
                    run_id,
                    runtime_type,
                    run_type,
                    parent_run_id,
                    backfill_id,
                    backfill_window_id,
                    window_start,
                    window_end,
                    json.dumps(request_payload),
                    _now(),
                ),
            )

    def create_backfill(
        self,
        backfill_id: str,
        runtime_type: str,
        run_type: str,
        failure_mode: str,
        max_concurrent_runs: int,
        total_windows: int,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO backfills ("
                "backfill_id, runtime_type, status, run_type, failure_mode, "
                "max_concurrent_runs, total_windows, started_at"
                ") VALUES (?, ?, 'running', ?, ?, ?, ?, ?)",
                (
                    backfill_id,
                    runtime_type,
                    run_type,
                    failure_mode,
                    max_concurrent_runs,
                    total_windows,
                    _now(),
                ),
            )

    def update_backfill_progress(
        self,
        backfill_id: str,
        *,
        succeeded_delta: int = 0,
        failed_delta: int = 0,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE backfills "
                "SET succeeded_windows = succeeded_windows + ?, failed_windows = failed_windows + ? "
                "WHERE backfill_id = ?",
                (succeeded_delta, failed_delta, backfill_id),
            )

    def complete_backfill(
        self,
        backfill_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE backfills SET status=?, finished_at=?, error=? WHERE backfill_id=?",
                (status, _now(), error, backfill_id),
            )

    def get_backfill(self, backfill_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM backfills WHERE backfill_id=?",
                (backfill_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "backfill_id": row["backfill_id"],
            "runtime_type": row["runtime_type"],
            "status": row["status"],
            "run_type": row["run_type"],
            "failure_mode": row["failure_mode"],
            "max_concurrent_runs": row["max_concurrent_runs"],
            "total_windows": row["total_windows"],
            "succeeded_windows": row["succeeded_windows"],
            "failed_windows": row["failed_windows"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": row["error"],
        }

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
                    "(id, run_id, node_id, component, status, message, inputs_lossy, started_at, finished_at, inputs, output, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        run_id,
                        node.get("node_id", ""),
                        node.get("component", ""),
                        node.get("status", ""),
                        node.get("message", ""),
                        1 if node.get("inputs_lossy") else 0,
                        node.get("started_at"),
                        node.get("finished_at"),
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
                # rowid preserves insertion order, which complete_run writes in
                # pipeline-config order (recorded_at is identical for every
                # node of a run, so sorting by it would fall back to
                # alphabetical node_id and scramble the pipeline order).
                "SELECT * FROM run_nodes WHERE run_id=? ORDER BY rowid",
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
                "inputs_lossy": bool(n["inputs_lossy"]),
                "started_at": n["started_at"],
                "finished_at": n["finished_at"],
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
        backfill_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Run metadata (no node payloads), newest first, with optional filters.

        Returns (runs, count) where count is the total matching rows before
        LIMIT/OFFSET pagination.
        """
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
        if backfill_id is not None:
            clauses.append("backfill_id = ?")
            params.append(backfill_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        clamped_limit = max(1, min(limit, 500))
        clamped_offset = max(0, offset)

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS total FROM runs {where}",
                params,
            ).fetchone()["total"]
            rows = conn.execute(
                f"SELECT * FROM runs {where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                [*params, clamped_limit, clamped_offset],
            ).fetchall()
        return [self._run_metadata(r) for r in rows], int(total)

    def create_schedule(
        self,
        schedule_id: str,
        runtime_type: str,
        payload: dict[str, Any],
        run_type: str,
        interval_seconds: int,
        overlap_policy: str,
        next_run_at: str | None = None,
    ) -> None:
        """Create a new schedule."""
        now = _now()
        effective_next = next_run_at if next_run_at else now
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO schedules ("
                "schedule_id, runtime_type, payload, run_type, interval_seconds, "
                "enabled, overlap_policy, next_run_at, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                (
                    schedule_id,
                    runtime_type,
                    json.dumps(payload),
                    run_type,
                    interval_seconds,
                    overlap_policy,
                    effective_next,
                    now,
                    now,
                ),
            )

    def list_schedules(self) -> list[dict[str, Any]]:
        """List all schedules (enabled and disabled)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM schedules ORDER BY created_at DESC"
            ).fetchall()
        return [self._schedule_record(r) for r in rows]

    def get_schedule(self, schedule_id: str) -> dict[str, Any] | None:
        """Fetch one schedule by id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM schedules WHERE schedule_id = ?",
                (schedule_id,),
            ).fetchone()
        if row is None:
            return None
        return self._schedule_record(row)

    def update_schedule(
        self,
        schedule_id: str,
        *,
        payload: dict[str, Any] | None = None,
        interval_seconds: int | None = None,
        enabled: int | None = None,
        overlap_policy: str | None = None,
    ) -> None:
        """Update schedule fields."""
        updates: list[str] = ["updated_at = ?"]
        params: list[Any] = [_now()]
        if payload is not None:
            updates.append("payload = ?")
            params.append(json.dumps(payload))
        if interval_seconds is not None:
            updates.append("interval_seconds = ?")
            params.append(interval_seconds)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(enabled)
        if overlap_policy is not None:
            updates.append("overlap_policy = ?")
            params.append(overlap_policy)
        params.append(schedule_id)

        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE schedules SET {', '.join(updates)} WHERE schedule_id = ?",
                params,
            )

    def delete_schedule(self, schedule_id: str) -> None:
        """Delete a schedule."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM schedules WHERE schedule_id = ?", (schedule_id,))

    def record_schedule_run(
        self,
        schedule_id: str,
        run_id: str,
    ) -> None:
        """Update schedule after a run was launched."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE schedules SET last_run_at = ?, last_run_id = ?, last_error = NULL "
                "WHERE schedule_id = ?",
                (_now(), run_id, schedule_id),
            )

    def record_schedule_error(
        self,
        schedule_id: str,
        error: str,
    ) -> None:
        """Record an error on a schedule (e.g. contract validation failed)."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE schedules SET last_error = ? WHERE schedule_id = ?",
                (error, schedule_id),
            )

    def get_due_schedules(self) -> list[dict[str, Any]]:
        """Get all enabled schedules where next_run_at <= now."""
        now = _now()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM schedules WHERE enabled = 1 AND next_run_at <= ? ORDER BY next_run_at",
                (now,),
            ).fetchall()
        return [self._schedule_record(r) for r in rows]

    def advance_schedule(self, schedule_id: str, interval_seconds: int) -> None:
        """Advance a schedule's next_run_at by interval."""
        with self._lock, self._connect() as conn:
            current_row = conn.execute(
                "SELECT next_run_at FROM schedules WHERE schedule_id = ?",
                (schedule_id,),
            ).fetchone()
            if current_row is None:
                return
            current_ts = datetime.fromisoformat(current_row["next_run_at"])
            new_ts = current_ts + timedelta(seconds=interval_seconds)
            conn.execute(
                "UPDATE schedules SET next_run_at = ? WHERE schedule_id = ?",
                (new_ts.isoformat(), schedule_id),
            )

    def get_in_flight_run_for_schedule(self, schedule_id: str) -> dict[str, Any] | None:
        """Get the last_run_id from the schedule and fetch its status."""
        schedule = self.get_schedule(schedule_id)
        if schedule is None or schedule["last_run_id"] is None:
            return None
        return self.get_run(schedule["last_run_id"])

    @staticmethod
    def _schedule_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schedule_id": row["schedule_id"],
            "runtime_type": row["runtime_type"],
            "payload": json.loads(row["payload"]),
            "run_type": row["run_type"],
            "interval_seconds": row["interval_seconds"],
            "enabled": bool(row["enabled"]),
            "overlap_policy": row["overlap_policy"],
            "next_run_at": row["next_run_at"],
            "last_run_at": row["last_run_at"],
            "last_run_id": row["last_run_id"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_runtime_stats(
        self,
        runtime_type: str,
        *,
        run_type: str | None = None,
        since: str | None = None,
        limit_runs: int = 200,
    ) -> dict[str, Any]:
        """Aggregate run and node-duration stats for one runtime.

        Windowing supports either:
        - latest N runs (`limit_runs`), optionally narrowed by run_type
        - a started_at lower bound (`since`) plus limit_runs cap
        """
        clauses = ["runtime_type = ?"]
        params: list[Any] = [runtime_type]
        if run_type is not None:
            clauses.append("run_type = ?")
            params.append(run_type)
        if since is not None:
            clauses.append("started_at >= ?")
            params.append(since)
        where = " AND ".join(clauses)
        clamped_limit = max(1, min(limit_runs, 1000))

        with self._connect() as conn:
            selected_runs = conn.execute(
                f"SELECT run_id, status, succeeded, run_type FROM runs "
                f"WHERE {where} ORDER BY started_at DESC LIMIT ?",
                [*params, clamped_limit],
            ).fetchall()

            total_runs = len(selected_runs)
            if total_runs == 0:
                return {
                    "runtime_type": runtime_type,
                    "window": {
                        "since": since,
                        "limit_runs": clamped_limit,
                        "selected_runs": 0,
                    },
                    "runs": {
                        "total": 0,
                        "completed": 0,
                        "running": 0,
                        "error": 0,
                        "failed": 0,
                        "failure_rate": 0.0,
                        "by_run_type": {},
                    },
                    "node_durations": [],
                }

            run_ids = [r["run_id"] for r in selected_runs]
            run_placeholders = ",".join("?" for _ in run_ids)

            counts = conn.execute(
                f"SELECT "
                f"COUNT(*) AS total, "
                f"SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed, "
                f"SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running, "
                f"SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error, "
                f"SUM(CASE WHEN status = 'error' OR (status = 'completed' AND COALESCE(succeeded, 0) = 0) THEN 1 ELSE 0 END) AS failed "
                f"FROM runs WHERE run_id IN ({run_placeholders})",
                run_ids,
            ).fetchone()

            by_run_type_rows = conn.execute(
                f"SELECT run_type, COUNT(*) AS c "
                f"FROM runs WHERE run_id IN ({run_placeholders}) "
                f"GROUP BY run_type ORDER BY run_type",
                run_ids,
            ).fetchall()

            duration_rows = conn.execute(
                f"SELECT node_id, (finished_at - started_at) AS duration "
                f"FROM run_nodes "
                f"WHERE run_id IN ({run_placeholders}) "
                f"AND started_at IS NOT NULL AND finished_at IS NOT NULL "
                f"AND finished_at >= started_at",
                run_ids,
            ).fetchall()

        durations_by_node: dict[str, list[float]] = {}
        for row in duration_rows:
            node_id = str(row["node_id"])
            durations_by_node.setdefault(node_id, []).append(float(row["duration"]))

        node_durations: list[dict[str, Any]] = []
        for node_id in sorted(durations_by_node.keys()):
            values = sorted(durations_by_node[node_id])
            count = len(values)
            avg_value = sum(values) / count
            p95_index = max(0, math.ceil(0.95 * count) - 1)
            node_durations.append(
                {
                    "node_id": node_id,
                    "count": count,
                    "avg_seconds": round(avg_value, 6),
                    "p95_seconds": round(values[p95_index], 6),
                }
            )

        failed = int(counts["failed"] or 0)
        total = int(counts["total"] or 0)
        failure_rate = (failed / total) if total else 0.0

        return {
            "runtime_type": runtime_type,
            "window": {
                "since": since,
                "limit_runs": clamped_limit,
                "selected_runs": total,
            },
            "runs": {
                "total": total,
                "completed": int(counts["completed"] or 0),
                "running": int(counts["running"] or 0),
                "error": int(counts["error"] or 0),
                "failed": failed,
                "failure_rate": round(failure_rate, 6),
                "by_run_type": {str(r["run_type"]): int(r["c"]) for r in by_run_type_rows},
            },
            "node_durations": node_durations,
        }

    @staticmethod
    def _run_metadata(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "runtime_type": row["runtime_type"],
            "run_type": row["run_type"],
            "parent_run_id": row["parent_run_id"],
            "backfill_id": row["backfill_id"],
            "backfill_window_id": row["backfill_window_id"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "status": row["status"],
            "succeeded": None if row["succeeded"] is None else bool(row["succeeded"]),
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
