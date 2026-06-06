from __future__ import annotations

import hashlib

from sample.raw_input_data.sample_input import generate_sample_data


class DataProvider:

    def get_training_data(
        self, source: str, entity_id: str | None = None
    ) -> tuple[list[dict[str, float]], list[float]]:
        """Fetch training data for a given source and optional entity.

        Args:
            source:    Data source type — "sample", "csv", or "database".
            entity_id: Optional entity scope (e.g. a location ID). When
                       provided, only data belonging to that entity is
                       returned. For the sample source this is simulated
                       via a deterministic per-entity seed so each entity
                       gets distinct but reproducible training data.
        """
        if source == "sample":
            seed = _entity_seed(entity_id) if entity_id is not None else 42
            return generate_sample_data(n=100, seed=seed)
        elif source == "csv":
            raise NotImplementedError("CSV source not yet implemented")
        elif source == "database":
            raise NotImplementedError("database source not yet implemented")
        else:
            raise ValueError(f"unknown data source: '{source}'")


def _entity_seed(entity_id: str) -> int:
    """Convert an entity ID string to a stable integer seed for reproducible data."""
    return int(hashlib.md5(entity_id.encode(), usedforsecurity=False).hexdigest(), 16) % (2**31)
