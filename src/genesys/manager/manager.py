"""Orchestrator (v2 ComponentManager).

Responsibilities:
  1. Load a per-runtime-type pipeline YAML (one file per runtime).
  2. Build the dependency graph and schedule nodes with a ready-set scheduler:
     a node is submitted to the thread pool the moment ALL of its dependencies
     have finished — it never waits for unrelated nodes.
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

import logging
import math
import re
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from genesys.manager.lro import LROCollector


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Run classification. `run_type` is a free-form label so teams can define
# their own (e.g. "team_a_check", "chris_experiment") — but with two rules:
#   1. Format is enforced (lowercase slug) so labels stay queryable and a
#      stray space/typo can't create unfilterable values downstream.
#   2. "production" is the single reserved label business consumers treat as
#      customer-grade/verified. It is never a default — always an explicit
#      assertion. Everything else is non-production by definition.
RUN_TYPE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
PRODUCTION_RUN_TYPE = "production"
DEFAULT_RUN_TYPE = "test"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

def _json_safe_with_lossiness(value: Any) -> tuple[Any, bool]:
    """Snapshot *value* into JSON-safe data and report whether conversion was lossy."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        lossy = False
        for k, v in value.items():
            safe_v, child_lossy = _json_safe_with_lossiness(v)
            out[str(k)] = safe_v
            lossy = lossy or child_lossy
        return out, lossy
    if isinstance(value, (list, tuple)):
        out_list: list[Any] = []
        lossy = False
        for v in value:
            safe_v, child_lossy = _json_safe_with_lossiness(v)
            out_list.append(safe_v)
            lossy = lossy or child_lossy
        return out_list, lossy
    if isinstance(value, float):
        if math.isfinite(value):
            return value, False
        return None, True
    if value is None or isinstance(value, (str, int, bool)):
        return value, False
    return repr(value), True


def json_safe(value: Any) -> Any:
    """Public JSON-safe snapshot used when comparing or storing replay output."""
    safe_value, _ = _json_safe_with_lossiness(value)
    return safe_value


@dataclass
class NodeOutcome:
    node_id: str
    component: str
    status: str   # completed | skipped | failed
    output: dict[str, Any] | None = None
    message: str = ""
    traceback: str | None = None
    # Resolved inputs (after ${...} interpolation). Recorded so a stored run
    # can later be replayed node-by-node without re-running upstream nodes.
    inputs: dict[str, Any] | None = None
    # True when inputs were coerced for storage (e.g. NaN->None or repr(...)).
    # Replay requires lossless JSON inputs.
    inputs_lossy: bool = False
    # Epoch-second timestamps captured around node execution.
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        safe_output = None
        if self.output is not None:
            safe_output, _ = _json_safe_with_lossiness(self.output)

        safe_inputs = None
        inferred_inputs_lossy = False
        if self.inputs is not None:
            safe_inputs, inferred_inputs_lossy = _json_safe_with_lossiness(self.inputs)

        d = {
            "node_id": self.node_id,
            "component": self.component,
            "status": self.status,
            # Sanitize at the serialization boundary only — in-process the raw
            # objects (incl. NaN floats / rich objects) keep flowing between
            # nodes untouched.
            "output": safe_output,
            "message": self.message,
            "inputs": safe_inputs,
            "inputs_lossy": self.inputs_lossy or inferred_inputs_lossy,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.traceback is not None:
            d["traceback"] = self.traceback
        return d


@dataclass
class RunResult:
    runtime_type: str
    succeeded: bool
    run_type: str = DEFAULT_RUN_TYPE
    nodes: list[NodeOutcome] = field(default_factory=list)
    lros: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_type": self.runtime_type,
            "run_type": self.run_type,
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


def validate_pipeline_config(
    config: dict[str, Any],
    registered_components: set[str] | None = None,
) -> list[str]:
    """Statically validate a pipeline config dict. Returns a list of problems
    (empty = valid).

    Checks: node shape, duplicate ids, dependency graph (cycles / unknown
    deps), component existence (when *registered_components* is given), and
    the contract block shape. Used at save time so a data scientist gets
    instant feedback instead of a runtime failure.
    """
    problems: list[str] = []

    nodes = config.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return ["'nodes' must be a non-empty list"]

    seen_ids: set[str] = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            problems.append(f"node[{i}] must be an object")
            continue
        node_id = node.get("id")
        if not node_id or not isinstance(node_id, str):
            problems.append(f"node[{i}] must have a string 'id'")
            continue
        if node_id in seen_ids:
            problems.append(f"duplicate node id '{node_id}'")
        seen_ids.add(node_id)

        component = node.get("component")
        if not component or not isinstance(component, str):
            problems.append(f"node '{node_id}' must have a string 'component'")
        elif registered_components is not None and component not in registered_components:
            problems.append(
                f"node '{node_id}' uses unknown component '{component}' "
                f"(registered: {sorted(registered_components)})"
            )

        if "inputs" in node and node["inputs"] is not None and not isinstance(node["inputs"], dict):
            problems.append(f"node '{node_id}': 'inputs' must be a mapping")
        if "depends_on" in node and not isinstance(node["depends_on"], list):
            problems.append(f"node '{node_id}': 'depends_on' must be a list")

    if not problems:
        deps = {n["id"]: n.get("depends_on", []) or [] for n in nodes}
        try:
            _validate_graph(deps)
        except ValueError as exc:
            problems.append(str(exc))

    contract = config.get("contract")
    if contract is not None:
        if not isinstance(contract, list):
            problems.append("'contract' must be a list of field definitions")
        else:
            for i, field_def in enumerate(contract):
                if not isinstance(field_def, dict) or not field_def.get("name"):
                    problems.append(f"contract[{i}] must be an object with a 'name'")

    return problems


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
    run_type: str = DEFAULT_RUN_TYPE  # one of RUN_TYPES; see module docstring


class Manager:
    def __init__(self, config_dir: str | Path, max_workers: int = 8) -> None:
        self._config_dir = Path(config_dir)
        self._registry: dict[str, Any] = {}
        # Upper bound on threads per level — prevents a wide fan-out (e.g. a
        # 50-node level) from spawning 50 threads at once.
        self._max_workers = max(1, max_workers)

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

    def unregister(self, name: str) -> None:
        self._registry.pop(name, None)

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

    def validate_request(self, runtime_type: str, payload: dict[str, Any]) -> list[str]:
        """Validate *payload* against the pipeline's contract without running it.

        Returns a list of violations (empty = valid). Raises FileNotFoundError
        if the runtime_type has no config. Used to reject bad async runs
        before a run record is created.
        """
        contract_cfg = self.get_contract(runtime_type)["contract"]
        if not contract_cfg:
            return []
        return _validate_contract(contract_cfg, payload)

    def replay_node(
        self,
        runtime_type: str,
        node_id: str,
        resolved_inputs: dict[str, Any],
    ) -> tuple[NodeOutcome, list[dict[str, Any]]]:
        """Execute one node from the current runtime config using pinned inputs."""
        config_path = self._config_dir / f"{runtime_type}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"no pipeline config for runtime_type='{runtime_type}'")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        nodes_cfg = config.get("nodes", [])
        target_cfg = next((n for n in nodes_cfg if n.get("id") == node_id), None)
        if target_cfg is None:
            raise KeyError(f"node '{node_id}' does not exist in runtime '{runtime_type}'")

        component_name = target_cfg.get("component", "")
        component = self._registry.get(component_name)
        if component is None:
            raise LookupError(f"component '{component_name}' is not registered")

        lro = LROCollector()
        started_at = time.time()
        try:
            output = component.execute(
                lro_collector=lro,
                node_id=node_id,
                **resolved_inputs,
            )
            if output is not None and not isinstance(output, dict):
                raise TypeError(
                    f"component '{component_name}' returned {type(output).__name__}; "
                    "execute() must return a dict (or None)"
                )
            outcome = NodeOutcome(
                node_id=node_id,
                component=component_name,
                status="completed",
                output=output or {},
                inputs=resolved_inputs,
                started_at=started_at,
                finished_at=time.time(),
            )
        except Exception as exc:
            outcome = NodeOutcome(
                node_id=node_id,
                component=component_name,
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
                inputs=resolved_inputs,
                started_at=started_at,
                finished_at=time.time(),
            )
        return outcome, lro.snapshot()

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
        if not RUN_TYPE_RE.match(options.run_type):
            raise ValueError(
                f"invalid run_type '{options.run_type}' — must be a lowercase slug "
                "(letters, digits, '_', '-'; starting with a letter; max 64 chars)"
            )

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
            return RunResult(runtime_type=runtime_type, succeeded=True, run_type=options.run_type)

        nodes_by_id: dict[str, dict] = {n["id"]: n for n in nodes_cfg}
        deps: dict[str, list[str]] = {n["id"]: n.get("depends_on", []) for n in nodes_cfg}
        _validate_graph(deps)

        # Reverse edges + pending-dependency counts for ready-set scheduling.
        dependents: dict[str, list[str]] = {nid: [] for nid in deps}
        for nid, dep_list in deps.items():
            for dep in dep_list:
                dependents[dep].append(nid)
        pending_deps: dict[str, int] = {nid: len(dep_list) for nid, dep_list in deps.items()}

        lro = LROCollector()
        # "run" exposes run-level metadata to pipelines: ${run.run_type} lets a
        # node (e.g. save_results) tag persisted data with its classification.
        run_context: dict[str, Any] = {
            "request": request_payload,
            "run": {"run_type": options.run_type},
        }
        outcomes: dict[str, NodeOutcome] = {}
        ctx_lock = threading.Lock()
        any_failure = threading.Event()

        def execute_one(node_id: str) -> NodeOutcome:
            cfg_node = nodes_by_id[node_id]
            component_name = cfg_node["component"]
            started_at = time.time()

            # Debug filter
            if options.debug_components is not None and component_name not in options.debug_components:
                return NodeOutcome(
                    node_id=node_id,
                    component=component_name,
                    status="skipped",
                    message="not in X-Debug-Component-List",
                    started_at=started_at,
                    finished_at=time.time(),
                )

            component = self._registry.get(component_name)
            if component is None:
                return NodeOutcome(
                    node_id=node_id,
                    component=component_name,
                    status="skipped",
                    message=f"component '{component_name}' not registered",
                    started_at=started_at,
                    finished_at=time.time(),
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
                    started_at=started_at,
                    finished_at=time.time(),
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
                outcome = NodeOutcome(
                    node_id=node_id,
                    component=component_name,
                    status="completed",
                    output=output or {},
                    inputs=inputs,
                    started_at=started_at,
                    finished_at=time.time(),
                )
            except Exception as exc:
                return NodeOutcome(
                    node_id=node_id,
                    component=component_name,
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc() if options.trace_failures else None,
                    inputs=inputs,
                    started_at=started_at,
                    finished_at=time.time(),
                )

            # Honor wait_for_lros inside the worker: the node only counts as
            # finished (and its dependents only become ready) once all of its
            # long-running operations reach a terminal state.
            if cfg_node.get("wait_for_lros"):
                ok, failed_ops = lro.wait_for_node(node_id)
                if not ok:
                    outcome.status = "failed"
                    outcome.message = (
                        f"wait_for_lros: {len(failed_ops)} operation(s) did not complete: {failed_ops}"
                    )
            return outcome

        # Ready-set scheduler: submit a node as soon as ALL its dependencies
        # have finished. A slow node only delays its own dependents — unrelated
        # branches keep flowing through the (capped) thread pool.
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            in_flight: dict[Future[NodeOutcome], str] = {}

            def submit(node_id: str) -> None:
                logger.info(f"[manager] scheduling node '{node_id}'")
                in_flight[pool.submit(execute_one, node_id)] = node_id

            for nid, count in pending_deps.items():
                if count == 0:
                    submit(nid)

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    finished_id = in_flight.pop(fut)
                    outcome = fut.result()
                    outcomes[finished_id] = outcome
                    with ctx_lock:
                        run_context[finished_id] = {"output": outcome.output or {}}
                    if outcome.status == "failed":
                        any_failure.set()

                    for child in dependents[finished_id]:
                        pending_deps[child] -= 1
                        if pending_deps[child] == 0:
                            if any_failure.is_set() and options.fail_fast:
                                continue  # marked skipped below
                            submit(child)

        # Any node never scheduled (fail-fast cut its branch off) is skipped.
        for nid in deps:
            if nid not in outcomes:
                outcomes[nid] = NodeOutcome(
                    node_id=nid,
                    component=nodes_by_id[nid]["component"],
                    status="skipped",
                    message="fail-fast: an earlier node failed",
                )

        # Build final result preserving config order for readability
        ordered = [outcomes[n["id"]] for n in nodes_cfg if n["id"] in outcomes]
        succeeded = all(o.status != "failed" for o in ordered)
        return RunResult(
            runtime_type=runtime_type,
            succeeded=succeeded,
            run_type=options.run_type,
            nodes=ordered,
            lros=lro.snapshot(),
        )


# ---------------------------------------------------------------------------
# Graph validation (cycles + unknown dependencies)
# ---------------------------------------------------------------------------

def _validate_graph(deps: dict[str, list[str]]) -> None:
    """Raise ValueError on circular dependencies or references to unknown nodes."""
    visited: set[str] = set()

    def visit(node_id: str, visiting: set[str]) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise ValueError(f"circular dependency at node '{node_id}'")
        visiting = visiting | {node_id}
        for dep in deps.get(node_id, []):
            if dep not in deps:
                raise ValueError(f"node '{node_id}' depends on unknown node '{dep}'")
            visit(dep, visiting)
        visited.add(node_id)

    for nid in deps:
        visit(nid, set())
