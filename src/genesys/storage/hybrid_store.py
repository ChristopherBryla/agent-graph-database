"""HybridStore — SQLite metadata + BlobStore binary data.

Use this when an artifact has two parts:
  - Queryable metadata (model name, run id, timestamp, tags, content type)
  - A binary payload that's too large or unstructured for a database column
    (images, serialized model checkpoints, CSV exports, PDFs, etc.)

The record in SQLite always contains a `blob_id` column that points to the
corresponding file in the BlobStore. If no binary data is needed, `blob`
may be omitted and `blob_id` will be None.

Typical use:
    store = HybridStore(db_path="genesys.db", blob_path="blobs")

    # Save an image with metadata
    record_id = store.save(
        metadata={"run_id": "...", "model": "cnn", "label": "cat"},
        blob=image_bytes,
        content_type="image/jpeg",
    )

    # Query by metadata
    records = store.query(model="cnn")

    # Retrieve the image later
    image_bytes = store.load_blob(record_id)
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

from genesys.storage.blob_store import BlobStore


class HybridStore:
    """Queryable metadata in SQLite, binary payloads on the filesystem."""

    def __init__(
        self,
        db_path: str = "genesys.db",
        blob_path: str = "blobs",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(base_path=blob_path)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id           TEXT PRIMARY KEY,
                    blob_id      TEXT,           -- NULL when no binary data
                    content_type TEXT,
                    metadata     TEXT NOT NULL,  -- JSON dict of user-supplied tags
                    created_at   TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created_at);
            """)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        metadata: dict[str, Any],
        blob: bytes | None = None,
        content_type: str = "application/octet-stream",
        tags: dict[str, str] | None = None,
    ) -> str:
        """Save an artifact. Returns the record_id.

        Args:
            metadata:     Queryable key-value pairs stored in SQLite.
            blob:         Optional binary data stored in the BlobStore.
            content_type: MIME type for the blob (e.g. "image/jpeg").
            tags:         Extra tags attached to the BlobStore entry.
        """
        record_id = str(uuid.uuid4())
        blob_id: str | None = None

        if blob is not None:
            blob_id = self.blobs.save(
                data=blob,
                content_type=content_type,
                tags={"record_id": record_id, **(tags or {})},
            )

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (id, blob_id, content_type, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (record_id, blob_id, content_type, json.dumps(metadata), _now()),
            )

        return record_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, record_id: str) -> dict[str, Any] | None:
        """Return the full record (metadata + blob_id), or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE id = ?", (record_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        return result

    def query(self, **filters: Any) -> list[dict[str, Any]]:
        """Return records whose metadata contains all supplied key=value pairs.

        Example:
            store.query(model="linear_regression", run_id="abc123")
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts ORDER BY created_at DESC"
            ).fetchall()

        results = []
        for row in rows:
            record = dict(row)
            meta = json.loads(record["metadata"])
            if all(meta.get(k) == v for k, v in filters.items()):
                record["metadata"] = meta
                results.append(record)
        return results

    def load_blob(self, record_id: str) -> bytes | None:
        """Return the binary data for a record, or None if there is none."""
        record = self.get(record_id)
        if record is None or record.get("blob_id") is None:
            return None
        return self.blobs.load(record["blob_id"])

    def list_artifacts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent artifacts (metadata only, no blob bytes)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            record = dict(row)
            record["metadata"] = json.loads(record["metadata"])
            results.append(record)
        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, record_id: str) -> None:
        """Remove the metadata row and, if present, the blob."""
        record = self.get(record_id)
        if record is None:
            return
        if record.get("blob_id"):
            self.blobs.delete(record["blob_id"])
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM artifacts WHERE id = ?", (record_id,))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
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
