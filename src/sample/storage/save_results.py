"""SaveResults — pipeline storage component for node type 'storage'.

Receives the outputs of all upstream model nodes (trained component
instances or None for skipped nodes) and persists them via a configured
store backend.

Supports three backends:
  - SQLiteStore  → structured rows only
  - HybridStore  → structured rows + optional binary blobs
  - BlobStore    → binary-only (use when models export raw checkpoints)

Wire it up in main.py / app.py lifespan:

    from genesys.storage.sqlite_store import SQLiteStore
    from sample.storage.save_results import SaveResults

    store = SQLiteStore(db_path="data/genesys.db")
    manager.register("save_results", SaveResults(store=store, pipeline_name="default"))
"""

from __future__ import annotations

from typing import Any


class SaveResults:
    """Storage pipeline component.

    `run()` is called by the component manager with all upstream model
    node outputs as positional arguments. Each output is either a trained
    component instance or None (node was skipped).
    """

    def __init__(self, store: Any, pipeline_name: str = "default") -> None:
        """
        Args:
            store:         Any of SQLiteStore, HybridStore, or BlobStore.
            pipeline_name: Label written into every run record for easy querying.
        """
        self._store = store
        self._pipeline_name = pipeline_name

    # ------------------------------------------------------------------
    # Pipeline interface — called by ComponentManager._execute_node
    # ------------------------------------------------------------------

    def run(self, *model_outputs: Any) -> dict[str, Any]:
        """Persist model outputs. Returns a summary dict.

        Each positional argument is the return value of a model node:
        - A trained component instance if the node completed.
        - None if the node was skipped.

        The component manager collects node outputs in dependency order so
        `model_outputs[0]` corresponds to the first dependency listed in
        the YAML `depends_on` for this storage node.
        """
        model_records = []
        node_results_summary = []

        for i, output in enumerate(model_outputs):
            if output is None:
                node_results_summary.append({"node_index": i, "status": "skipped"})
                continue

            record = _extract_model_record(output)
            model_records.append(record)
            node_results_summary.append({"node_index": i, "status": "saved", "model": record.get("model_name")})

        run_id = self._store.save_run(
            name=self._pipeline_name,
            node_results=node_results_summary,
            model_outputs=model_records,
        )

        summary = {
            "run_id": run_id,
            "pipeline": self._pipeline_name,
            "models_saved": len(model_records),
            "models_skipped": len(model_outputs) - len(model_records),
        }
        print(f"[storage] saved run {run_id!r} — {len(model_records)} model(s) persisted")
        return summary


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _extract_model_record(component: Any) -> dict[str, Any]:
    """Serialize a trained component into a storable dict.

    Works with any component that exposes the expected attributes.
    Unknown components are stored as an empty record with just their
    class name so the run is never silently dropped.
    """
    record: dict[str, Any] = {
        "model_name": type(component).__name__.lower(),
        "version": None,
        "weights": None,
        "bias": None,
        "extra": None,
    }

    # LinearRegression (and any model that follows the same convention)
    if hasattr(component, "weights"):
        record["weights"] = list(component.weights)
    if hasattr(component, "bias"):
        record["bias"] = float(component.bias)

    # DataContract version if declared on the class
    contract = getattr(component, "contract", None)
    if contract is not None:
        record["model_name"] = contract.model_name
        record["version"] = contract.version

    return record
