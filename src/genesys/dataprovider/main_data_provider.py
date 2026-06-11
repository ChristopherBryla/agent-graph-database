from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from sample.raw_input_data.sample_input import generate_sample_data


class DataProvider:

    def get_training_data(
        self,
        source: str,
        entity_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[list[dict[str, float]], list[float]]:
        """Fetch training data for a given source and optional entity.

        Args:
            source:    Data source type — "sample", "csv", or "database".
            entity_id: Optional entity scope (e.g. a location ID). When
                       provided, only data belonging to that entity is
                       returned. For the sample source this is simulated
                       via a deterministic per-entity seed so each entity
                       gets distinct but reproducible training data.
            start:     Optional ISO8601 lower bound for sample data windows.
            end:       Optional ISO8601 upper bound for sample data windows.
        """
        start_dt, end_dt = _validate_window(start, end)

        if source == "sample":
            seed = _entity_seed(entity_id) if entity_id is not None else 42
            data, targets = generate_sample_data(n=100, seed=seed)
            if start_dt is None and end_dt is None:
                return data, targets
            return _filter_sample_window(data, targets, start_dt, end_dt)
        elif source == "csv":
            raise NotImplementedError("CSV source not yet implemented")
        elif source == "database":
            raise NotImplementedError("database source not yet implemented")
        else:
            raise ValueError(f"unknown data source: '{source}'")


def _entity_seed(entity_id: str) -> int:
    """Convert an entity ID string to a stable integer seed for reproducible data."""
    return int(hashlib.md5(entity_id.encode(), usedforsecurity=False).hexdigest(), 16) % (2**31)


def _parse_iso8601(value: str, field: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO8601, got '{value}'") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _validate_window(start: str | None, end: str | None) -> tuple[datetime | None, datetime | None]:
    start_dt = _parse_iso8601(start, "start") if start is not None else None
    end_dt = _parse_iso8601(end, "end") if end is not None else None
    if start_dt is not None and end_dt is not None and end_dt <= start_dt:
        raise ValueError("end must be greater than start")
    return start_dt, end_dt


def _filter_sample_window(
    data: list[dict[str, float]],
    targets: list[float],
    start: datetime | None,
    end: datetime | None,
) -> tuple[list[dict[str, float]], list[float]]:
    # Synthetic daily timestamps provide deterministic window filtering for
    # sample data without introducing a real time column dependency.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    out_data: list[dict[str, float]] = []
    out_targets: list[float] = []
    for idx, (row, target) in enumerate(zip(data, targets)):
        ts = base + timedelta(days=idx)
        if start is not None and ts < start:
            continue
        if end is not None and ts >= end:
            continue
        out_data.append(row)
        out_targets.append(target)
    return out_data, out_targets
