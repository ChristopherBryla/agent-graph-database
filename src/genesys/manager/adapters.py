"""Component adapters for the v2 orchestrator.

Each adapter wraps an existing in-process component (DataProvider,
LinearRegression, SaveResults) so it conforms to the v2 contract:

    def execute(self, *, lro_collector, node_id, **inputs) -> dict

Adapters are intentionally thin so the underlying components stay unchanged
and the legacy v1 ComponentManager continues to work.

Each adapter also demonstrates LRO usage so you can see what shows up in the
`lros` field of the orchestrator response.
"""

from __future__ import annotations

import pickle
import threading
from typing import Any

from genesys.contract.data_contract import ContractViolationError
from genesys.dataprovider.main_data_provider import DataProvider
from genesys.manager.lro import LROCollector
from genesys.storage.model_registry_store import ModelRegistryStore
from sample.ml_model.ml_sample_model1 import LinearRegression


# ---------------------------------------------------------------------------
# Data provider adapter
# ---------------------------------------------------------------------------

class DataProviderAdapter:
    """Fetches training data and returns it as a dict in the run_context."""

    def __init__(self, provider: DataProvider | None = None) -> None:
        self._provider = provider or DataProvider()

    def execute(
        self,
        *,
        lro_collector: LROCollector,
        node_id: str,
        source: str = "sample",
        entity_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        data: list[dict[str, float]] | None = None,
        targets: list[float] | None = None,
    ) -> dict[str, Any]:
        op_id = lro_collector.start(
            node_id=node_id, component="data_provider", kind="fetch", ref=source,
        )
        try:
            if source == "passthrough":
                if data is None or targets is None:
                    raise ValueError("passthrough source requires 'data' and 'targets' inputs")
                result_data = data
                result_targets = list(targets)
            else:
                result_data, result_targets = self._provider.get_training_data(
                    source,
                    entity_id=entity_id,
                    start=start,
                    end=end,
                )
                result_targets = list(result_targets)
            lro_collector.complete(op_id)
        except Exception as exc:
            lro_collector.fail(op_id, str(exc))
            raise

        return {
            "data": result_data,
            "targets": result_targets,
            "row_count": len(result_data),
            "window": {"start": start, "end": end},
        }


# ---------------------------------------------------------------------------
# Linear regression adapter
# ---------------------------------------------------------------------------

class LinearRegressionAdapter:
    """Trains a fresh LinearRegression every call and returns serialized weights."""

    def execute(
        self,
        *,
        lro_collector: LROCollector,
        node_id: str,
        data: list[dict[str, float]],
        targets: list[float],
        learning_rate: float = 0.01,
        epochs: int = 500,
    ) -> dict[str, Any]:
        op_id = lro_collector.start(
            node_id=node_id, component="linear_regression", kind="train",
            ref=f"epochs={epochs}",
        )
        try:
            model = LinearRegression()

            # Enforce input contract — hard fail on any violation
            violations = model.contract.validate_batch(data)
            if violations:
                flat = [msg for vs in violations.values() for msg in vs]
                raise ContractViolationError(model.contract.model_name, flat)

            # Field order comes from the contract so this adapter never
            # needs to hardcode feature names
            field_names = [f.name for f in model.contract.fields]
            model.fit(
                [tuple(row[k] for k in field_names) for row in data],
                targets,
                learning_rate=learning_rate,
                epochs=epochs,
            )

            result = {
                "weights": list(model.weights),
                "bias": float(model.bias),
                "epochs": epochs,
                "learning_rate": learning_rate,
            }

            # Enforce output contract
            out_violations = model.output_contract.validate(result)
            if out_violations:
                raise ContractViolationError(model.output_contract.model_name, out_violations)

            lro_collector.complete(op_id)
        except Exception as exc:
            lro_collector.fail(op_id, str(exc))
            raise

        return result


# ---------------------------------------------------------------------------
# Save results adapter (in-memory by default — easy to inspect in tests)
# ---------------------------------------------------------------------------

class InMemorySaveResultsAdapter:
    """Persists model snapshots to an in-memory list. Inspect via `.records`.

    This demonstrates `wait_for_lros: true`: the save uses an LRO so the
    pipeline only marks the node 'completed' (and lets dependents go) once
    the write is fully recorded.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def execute(
        self,
        *,
        lro_collector: LROCollector,
        node_id: str,
        weights: list[float],
        bias: float,
        label: str = "default",
    ) -> dict[str, Any]:
        op_id = lro_collector.start(
            node_id=node_id, component="save_results", kind="persist", ref=label,
        )
        try:
            record = {"label": label, "weights": list(weights), "bias": float(bias)}
            with self._lock:
                self.records.append(record)
            lro_collector.complete(op_id)
        except Exception as exc:
            lro_collector.fail(op_id, str(exc))
            raise

        return {
            "label": label,
            "record_count": len(self.records),
        }


# ---------------------------------------------------------------------------
# Model lifecycle adapters
# ---------------------------------------------------------------------------

class SaveModelAdapter:
    """Save a versioned model artifact and return its model reference."""

    def __init__(self, registry: ModelRegistryStore) -> None:
        self._registry = registry

    def execute(
        self,
        *,
        lro_collector: LROCollector,
        node_id: str,
        name: str,
        artifact: Any,
        created_by_run_id: str,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not created_by_run_id:
            raise ValueError("save_model requires 'created_by_run_id' (pass ${run.run_id})")

        op_id = lro_collector.start(
            node_id=node_id, component="save_model", kind="persist", ref=name,
        )
        try:
            metadata = self._registry.save_model(
                name=name,
                artifact_bytes=pickle.dumps(artifact),
                created_by_run_id=created_by_run_id,
                metrics=metrics,
            )
            lro_collector.complete(op_id)
        except Exception as exc:
            lro_collector.fail(op_id, str(exc))
            raise

        return {
            "model_id": metadata["model_id"],
            "name": metadata["name"],
            "version": metadata["version"],
            "blob_key": metadata["blob_key"],
            "created_by_run_id": metadata["created_by_run_id"],
        }


class LoadModelAdapter:
    """Load a model artifact by name (and optional version)."""

    def __init__(self, registry: ModelRegistryStore) -> None:
        self._registry = registry

    def execute(
        self,
        *,
        lro_collector: LROCollector,
        node_id: str,
        name: str,
        version: int | None = None,
    ) -> dict[str, Any]:
        ref = f"{name}:{version}" if version is not None else f"{name}:latest"
        op_id = lro_collector.start(
            node_id=node_id, component="load_model", kind="load", ref=ref,
        )
        try:
            record = self._registry.load_model(name=name, version=version)
            if record is None:
                raise FileNotFoundError(f"no model found for name='{name}' version='{version}'")
            model = pickle.loads(record["artifact_bytes"])
            lro_collector.complete(op_id)
        except Exception as exc:
            lro_collector.fail(op_id, str(exc))
            raise

        return {
            "model_id": record["model_id"],
            "name": record["name"],
            "version": int(record["version"]),
            "model": model,
        }


class LinearPredictAdapter:
    """Generate predictions from a serialized linear regression artifact."""

    def execute(
        self,
        *,
        lro_collector: LROCollector,
        node_id: str,
        model: dict[str, Any],
        data: list[dict[str, float]],
    ) -> dict[str, Any]:
        op_id = lro_collector.start(
            node_id=node_id, component="linear_predict", kind="predict", ref=f"rows={len(data)}",
        )
        try:
            weights = model.get("weights")
            bias = model.get("bias")
            if not isinstance(weights, list) or len(weights) != 4:
                raise ValueError("linear_predict expected model.weights as a list of 4 numbers")
            if bias is None:
                raise ValueError("linear_predict expected model.bias")

            lr = LinearRegression()
            lr.weights = [float(w) for w in weights]
            lr.bias = float(bias)

            violations = lr.contract.validate_batch(data)
            if violations:
                flat = [msg for vs in violations.values() for msg in vs]
                raise ContractViolationError(lr.contract.model_name, flat)

            field_names = [f.name for f in lr.contract.fields]
            predictions = [
                float(lr.predict(*(float(row[k]) for k in field_names)))
                for row in data
            ]
            lro_collector.complete(op_id)
        except Exception as exc:
            lro_collector.fail(op_id, str(exc))
            raise

        return {
            "predictions": predictions,
            "row_count": len(predictions),
        }
