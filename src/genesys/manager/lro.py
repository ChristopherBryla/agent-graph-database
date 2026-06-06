"""LROCollector — Long-Running Operation tracker.

A single LROCollector flows through every node of an orchestrator run. Components
register their async work with `start(...)` and report terminal states via
`complete(op_id)` / `fail(op_id, error)`. The orchestrator can then block on a
specific node's operations before scheduling its dependents (set
`wait_for_lros: true` on that node in the pipeline YAML).

Why this exists:
  Some components kick off work that outlives the synchronous return of their
  execute() call (an upload, a queued job, an external write). Downstream nodes
  must not run until that work has actually finished, otherwise they read stale
  or missing data. The collector also doubles as an audit trail returned in the
  HTTP response so callers can poll or replay.

All operations are thread-safe: the orchestrator runs same-level nodes in
parallel and components inside one node may also spawn parallel work.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LROEntry:
    op_id: str
    node_id: str
    component: str
    kind: str
    ref: str | None
    state: str = "pending"   # pending | completed | failed
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "node_id": self.node_id,
            "component": self.component,
            "kind": self.kind,
            "ref": self.ref,
            "state": self.state,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": (
                None if self.finished_at is None
                else round(self.finished_at - self.started_at, 4)
            ),
        }


class LROCollector:
    def __init__(self) -> None:
        self._ops: dict[str, LROEntry] = {}
        self._lock = threading.Lock()
        # one Event per op_id — set when the op reaches a terminal state
        self._events: dict[str, threading.Event] = {}

    # ------------------------------------------------------------------
    # Component-facing API
    # ------------------------------------------------------------------

    def start(
        self,
        *,
        node_id: str,
        component: str,
        kind: str,
        ref: str | None = None,
    ) -> str:
        """Register a long-running operation. Returns the op_id."""
        op_id = str(uuid.uuid4())
        entry = LROEntry(
            op_id=op_id,
            node_id=node_id,
            component=component,
            kind=kind,
            ref=ref,
        )
        with self._lock:
            self._ops[op_id] = entry
            self._events[op_id] = threading.Event()
        return op_id

    def complete(self, op_id: str) -> None:
        with self._lock:
            entry = self._ops.get(op_id)
            if entry is None or entry.state != "pending":
                return
            entry.state = "completed"
            entry.finished_at = time.time()
            self._events[op_id].set()

    def fail(self, op_id: str, error: str) -> None:
        with self._lock:
            entry = self._ops.get(op_id)
            if entry is None or entry.state != "pending":
                return
            entry.state = "failed"
            entry.error = error
            entry.finished_at = time.time()
            self._events[op_id].set()

    # ------------------------------------------------------------------
    # Orchestrator-facing API
    # ------------------------------------------------------------------

    def wait_for_node(self, node_id: str, timeout: float = 60.0) -> tuple[bool, list[str]]:
        """Block until every op started by `node_id` is in a terminal state.

        Returns (all_ok, failed_op_ids). If any op timed out it counts as failed.
        """
        with self._lock:
            ops = [(op_id, self._events[op_id]) for op_id, e in self._ops.items() if e.node_id == node_id]

        deadline = time.time() + timeout
        failed_ops: list[str] = []
        for op_id, event in ops:
            remaining = max(0.0, deadline - time.time())
            if not event.wait(timeout=remaining):
                # treat timeout as failure so dependents won't proceed silently
                self.fail(op_id, f"timeout after {timeout}s")
                failed_ops.append(op_id)
                continue
            with self._lock:
                if self._ops[op_id].state == "failed":
                    failed_ops.append(op_id)
        return (len(failed_ops) == 0, failed_ops)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return all operations as JSON-serializable dicts (for the response)."""
        with self._lock:
            return [e.to_dict() for e in self._ops.values()]
