"""ModelRegistryStore — lightweight versioned model metadata + blob pointers."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from genesys.storage.blob_store import BlobStore


class ModelRegistryStore:
    """Thread-safe model version registry backed by SQLite + BlobStore."""

    def __init__(
        self,
        db_path: str = "data/genesys_models.db",
        blob_path: str = "data/blobs",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(base_path=blob_path)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id           TEXT PRIMARY KEY,
                    name               TEXT NOT NULL,
                    version            INTEGER NOT NULL,
                    blob_key           TEXT NOT NULL,
                    metrics            TEXT,
                    created_by_run_id  TEXT NOT NULL,
                    created_at         TEXT NOT NULL,
                    UNIQUE(name, version)
                );

                CREATE INDEX IF NOT EXISTS idx_models_name_created
                ON models(name, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_models_run
                ON models(created_by_run_id);
                """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_model(
        self,
        *,
        name: str,
        artifact_bytes: bytes,
        created_by_run_id: str,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_version FROM models WHERE name = ?",
                (name,),
            ).fetchone()
            version = int(row["max_version"]) + 1
            model_id = str(uuid.uuid4())
            blob_key = self.blobs.save(
                artifact_bytes,
                content_type="application/python-pickle",
                tags={"name": name, "version": str(version), "model_id": model_id},
            )
            created_at = _now()
            conn.execute(
                "INSERT INTO models (model_id, name, version, blob_key, metrics, created_by_run_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    name,
                    version,
                    blob_key,
                    json.dumps(metrics) if metrics is not None else None,
                    created_by_run_id,
                    created_at,
                ),
            )

        return {
            "model_id": model_id,
            "name": name,
            "version": version,
            "blob_key": blob_key,
            "created_by_run_id": created_by_run_id,
            "created_at": created_at,
        }

    def load_model(self, *, name: str, version: int | None = None) -> dict[str, Any] | None:
        with self._connect() as conn:
            if version is None:
                row = conn.execute(
                    "SELECT * FROM models WHERE name = ? ORDER BY version DESC LIMIT 1",
                    (name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM models WHERE name = ? AND version = ? LIMIT 1",
                    (name, version),
                ).fetchone()
        if row is None:
            return None

        artifact_bytes = self.blobs.load(row["blob_key"])
        return {
            "model_id": row["model_id"],
            "name": row["name"],
            "version": row["version"],
            "blob_key": row["blob_key"],
            "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
            "created_by_run_id": row["created_by_run_id"],
            "created_at": row["created_at"],
            "artifact_bytes": artifact_bytes,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
