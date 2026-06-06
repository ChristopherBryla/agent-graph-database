"""Orchestrator (v2 ComponentManager).

Responsibilities:
  1. Load a per-runtime-type pipeline YAML (one file per runtime).
  2. Build the dependency graph and group nodes into parallel-safe levels.
  3. Resolve every node's `inputs:` block by interpolating ${...} references
     against a shared run_context (request body + completed node outputs).
  4. Apply caller-supplied overrides:
        - X-Debug-Component-List → only run nodes whose component is in the list
        - X-Fail-Fast            → stop scheduling once any node fails
        - X-Trace-Failures       → include Python traceback in error output
  5. Pass an LROCollector into every component; before scheduling a node's
     dependents, optionally block on `wait_for_lros`.
  6. Return a structured result with per-node status, outputs, and the LRO log.

Components are looked up in a registry keyed by component name. Each component
must implement:

    def execute(self, *, lro_collector, node_id, **inputs) -> dict
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from genesys.manager.lro import LROCollector


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class NodeOutcome:
    node_id: str
    component: str
    status: str   # completed | skipped | failed
    output: dict[str, Any] | None = None
    message: str = ""
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "node_id": self.node_id,
            "component": self.component,
            "status": self.status,
            "output": self.output,
            "message": self.message,
        }
        if self.traceback is not None:
            d["traceback"] = self.traceback
        return d


@dataclass
class RunResult:
    runtime_type: str
    succeeded: bool
    nodes: list[NodeOutcome] = field(default_factory=list)
    lros: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_type": self.runtime_type,
            "succeeded": self.succeeded,
            "nodes": [n.to_dict() for n in self.nodes],
            "lros": self.lros,
        }


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

class ContractValidationError(Exception):
    """Raised when the request body does not satisfy the pipeline contract."""
    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("request body failed contract validation: " + "; ".join(violations))


def _validate_contract(contract_cfg: list[dict], payload: dict[str, Any]) -> list[str]:
    """Validate *payload* against a list of contract field definitions.

    Each entry in contract_cfg may have:
        name:     str   (required)
        type:     str   — "list" | "dict" | "float" | "int" | "str" | "any"  (default "any")
        required: bool  (default true)
        description: str (ignored, documentation only)
    """
    _type_map: dict[str, type | None] = {
        "list": list, "dict": dict,
        "float": float, "int": int, "str": str,
        "any": None,
    }
    violations: list[str] = []
    for field_def in contract_cfg:
        name = field_def.get("name", "")
        required = field_def.get("required", True)
        expected_type_str = field_def.get("type", "any")
        expected_type = _type_map.get(expected_type_str)

        if name not in payload:
            if required:
                violations.append(f"required field '{name}' is missing from request body")
            continue

        value = payload[name]
        if expected_type is not None and not isinstance(value, expected_type):
            violations.append(
                f"field '{name}' expected {expected_type_str}, "
                f"got {type(value).__name__}"
            )
    return violations


# ---------------------------------------------------------------------------
# Reference resolution: ${request.x} / ${node_id.output.field}
# ---------------------------------------------------------------------------

class ReferenceError(Exception):
    """Raised when a ${...} reference cannot be resolved."""


def resolve_value(value: Any, context: dict[str, Any]) -> Any:
    """Recursively resolve ${...} references inside dicts/lists/strings."""
    if isinstance(value, str):
        stripped = value.strip()
        # Whole-string reference: preserve original type (list, dict, number, ...)
        if stripped.startswith("${") and stripped.endswith("}") and stripped.count("${") == 1:
            return _lookup(stripped[2:-1].strip(), context)
        # Embedded reference inside a longer string → coerce to str
        if "${" in value:
            out = ""
            i = 0
            while i < len(value):
                if value[i:i + 2] == "${":
                    end = value.find("}", i + 2)
                    if end == -1:
                        raise ReferenceError(f"unterminated reference in {value!r}")
                    out += str(_lookup(value[i + 2:end].strip(), context))
                    i = end + 1
                else:
                    out += value[i]
                    i += 1
            return out
        return value
    if isinstance(value, dict):
        return {k: resolve_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_value(v, context) for v in value]
    return value


def _lookup(path: str, context: dict[str, Any]) -> Any:
    parts = path.split(".")
    current: Any = context
    walked: list[str] = []
    for part in parts:
        walked.append(part)
        if isinstance(current, dict):
            if part not in current:
                raise ReferenceError(
                    f"reference ${{{path}}} failed at '{'.'.join(walked)}' — key not found"
                )
            current = current[part]
        else:
            raise ReferenceError(
                f"reference ${{{path}}} failed at '{'.'.join(walked)}' — not a dict"
            )
    return current


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class RunOptions:
    debug_components: set[str] | None = None  # None = run everything
    fail_fast: bool = False
    trace_failures: bool = False


class Manager:
    def __init__(self, config_dir: str | Path) -> None:
        self._config_dir = Path(config_dir)
        self._registry: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register(self, name: str, component: Any) -> None:
        if not hasattr(component, "execute"):
            raise TypeError(
                f"component '{name}' must implement execute(*, lro_collector, node_id, **inputs)"
            )
        self._registry[name] = component

    def list_registered(self) -> list[str]:
        return sorted(self._registry.keys())

    def list_runtime_types(self) -> list[str]:
        if not self._config_dir.exists():
            return []
        return sorted(p.stem for p in self._config_dir.glob("*.yaml"))

    def get_contract(self, runtime_type: str) -> dict[str, Any]:
        """Return the contract block for a runtime_type (empty list if none declared)."""
        config_path = self._config_dir / f"{runtime_type}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"no pipeline config for runtime_type='{runtime_type}'")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return {
            "runtime_type": runtime_type,
            "contract": config.get("contract", []),
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        runtime_type: str,
        request_payload: dict[str, Any],
        options: RunOptions | None = None,
    ) -> RunResult:
        options = options or RunOptions()

        config_path = self._config_dir / f"{runtime_type}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"no pipeline config for runtime_type='{runtime_type}' "
                f"(expected {config_path})"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # Validate request body against the pipeline's contract (if declared).
        contract_cfg = config.get("contract", [])
        if contract_cfg:
            violations = _validate_contract(contract_cfg, request_payload)
            if violations:
                raise ContractValidationError(violations)

        nodes_cfg = config.get("nodes", [])
        if not nodes_cfg:
            return RunResult(runtime_type=runtime_type, succeeded=True)

        nodes_by_id: dict[str, dict] = {n["id"]: n for n in nodes_cfg}
        deps: dict[str, list[str]] = {n["id"]: n.get("depends_on", []) for n in nodes_cfg}
        levels = _build_levels(deps)

        lro = LROCollector()
        run_context: dict[str, Any] = {"request": request_payload}
        outcomes: dict[str, NodeOutcome] = {}
        ctx_lock = threading.Lock()
        any_failure = threading.Event()

        for level_index, level in enumerate(levels):
            if any_failure.is_set() and options.fail_fast:
                # Mark remaining nodes as skipped due to fail-fast
                for nid in level:
                    cfg_node = nodes_by_id[nid]
                    outcomes[nid] = NodeOutcome(
                        node_id=nid,
                        component=cfg_node["component"],
                        status="skipped",
                        message="fail-fast: an earlier node failed",
                    )
                continue

            print(f"[manager] level {level_index}: {level}")

            def execute_one(node_id: str) -> NodeOutcome:
                cfg_node = nodes_by_id[node_id]
                component_name = cfg_node["component"]

                # Debug filter
                if options.debug_components is not None and component_name not in options.debug_components:
                    return NodeOutcome(
                        node_id=node_id,
                        component=component_name,
                        status="skipped",
                        message="not in X-Debug-Component-List",
                    )

                component = self._registry.get(component_name)
                if component is None:
                    return NodeOutcome(
                        node_id=node_id,
                        component=component_name,
                        status="skipped",
                        message=f"component '{component_name}' not registered",
                    )

                # Resolve inputs against the shared context
                with ctx_lock:
                    snapshot_ctx = dict(run_context)
                try:
                    raw_inputs = cfg_node.get("inputs", {}) or {}
                    inputs = resolve_value(raw_inputs, snapshot_ctx)
                except ReferenceError as exc:
                    return NodeOutcome(
                        node_id=node_id,
                        component=component_name,
                        status="failed",
                        message=str(exc),
                        traceback=traceback.format_exc() if options.trace_failures else None,
                    )

                # Execute
                try:
                    output = component.execute(
                        lro_collector=lro,
                        node_id=node_id,
                        **inputs,
                    )
                    if output is not None and not isinstance(output, dict):
                        raise TypeError(
                            f"component '{component_name}' returned {type(output).__name__}; "
                            "execute() must return a dict (or None)"
                        )
                    return NodeOutcome(
                        node_id=node_id,
                        component=component_name,
                        status="completed",
                        output=output or {},
                    )
                except Exception as exc:
                    return NodeOutcome(
                        node_id=node_id,
                        component=component_name,
                        status="failed",
                        message=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc() if options.trace_failures else None,
                    )

            # Run this level in parallel
            with ThreadPoolExecutor(max_workers=max(len(level), 1)) as pool:
                futures = {pool.submit(execute_one, nid): nid for nid in level}
                for fut in as_completed(futures):
                    outcome = fut.result()
                    outcomes[outcome.node_id] = outcome
                    with ctx_lock:
                        run_context[outcome.node_id] = {"output": outcome.output or {}}
                    if outcome.status == "failed":
                        any_failure.set()

            # After the level finishes, honor wait_for_lros for any node in it
            for nid in level:
                if not nodes_by_id[nid].get("wait_for_lros"):
                    continue
                if outcomes[nid].status != "completed":
                    continue
                ok, failed_ops = lro.wait_for_node(nid)
                if not ok:
                    outcomes[nid].status = "failed"
                    outcomes[nid].message = (
                        f"wait_for_lros: {len(failed_ops)} operation(s) did not complete: {failed_ops}"
                    )
                    any_failure.set()

        # Build final result preserving config order for readability
        ordered = [outcomes[n["id"]] for n in nodes_cfg if n["id"] in outcomes]
        succeeded = all(o.status != "failed" for o in ordered)
        return RunResult(
            runtime_type=runtime_type,
            succeeded=succeeded,
            nodes=ordered,
            lros=lro.snapshot(),
        )


# ---------------------------------------------------------------------------
# Topological level grouping (same algorithm as the v1 manager)
# ---------------------------------------------------------------------------

def _build_levels(deps: dict[str, list[str]]) -> list[list[str]]:
    node_level: dict[str, int] = {}

    def get_level(node_id: str, visiting: set[str]) -> int:
        if node_id in node_level:
            return node_level[node_id]
        if node_id in visiting:
            raise ValueError(f"circular dependency at node '{node_id}'")
        visiting = visiting | {node_id}
        lvl = 0
        for dep in deps.get(node_id, []):
            if dep not in deps:
                raise ValueError(f"node '{node_id}' depends on unknown node '{dep}'")
            lvl = max(lvl, get_level(dep, visiting) + 1)
        node_level[node_id] = lvl
        return lvl

    for nid in deps:
        get_level(nid, set())

    if not node_level:
        return []
    max_lvl = max(node_level.values())
    levels: list[list[str]] = [[] for _ in range(max_lvl + 1)]
    for nid, lvl in node_level.items():
        levels[lvl].append(nid)
    return levels
