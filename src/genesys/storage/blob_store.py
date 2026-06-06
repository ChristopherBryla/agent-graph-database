"""BlobStore — filesystem-backed binary object storage.

Each blob is stored as a flat file under `base_path/` named by a UUID.
A companion `.meta` JSON file sits next to it with content_type and tags.

This local implementation is intentionally swap-friendly: replace the
`save` / `load` / `delete` methods with S3, Azure Blob, or GCS calls
and the HybridStore (and everything above it) keeps working unchanged.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class BlobStore:
    """Local filesystem blob storage.

    Blobs are stored as `{base_path}/{blob_id}`.
    Metadata is stored as `{base_path}/{blob_id}.meta` (JSON).
    """

    def __init__(self, base_path: str = "blobs") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        data: bytes,
        content_type: str = "application/octet-stream",
        tags: dict[str, str] | None = None,
    ) -> str:
        """Persist binary data and return a unique blob_id."""
        blob_id = str(uuid.uuid4())
        blob_path = self.base_path / blob_id
        meta_path = self.base_path / f"{blob_id}.meta"

        blob_path.write_bytes(data)
        meta: dict = {
            "blob_id": blob_id,
            "content_type": content_type,
            "size_bytes": len(data),
            "tags": tags or {},
            "created_at": _now(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return blob_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, blob_id: str) -> bytes:
        """Load raw bytes for a blob. Raises FileNotFoundError if missing."""
        return (self.base_path / blob_id).read_bytes()

    def meta(self, blob_id: str) -> dict:
        """Return the metadata dict for a blob."""
        meta_path = self.base_path / f"{blob_id}.meta"
        if not meta_path.exists():
            raise FileNotFoundError(f"no blob metadata for id '{blob_id}'")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def exists(self, blob_id: str) -> bool:
        return (self.base_path / blob_id).exists()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, blob_id: str) -> None:
        """Remove a blob and its metadata. Silently ignores missing ids."""
        (self.base_path / blob_id).unlink(missing_ok=True)
        (self.base_path / f"{blob_id}.meta").unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_blobs(self) -> list[dict]:
        """Return metadata for all stored blobs."""
        blobs = []
        for meta_path in self.base_path.glob("*.meta"):
            try:
                blobs.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
        return sorted(blobs, key=lambda b: b.get("created_at", ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
