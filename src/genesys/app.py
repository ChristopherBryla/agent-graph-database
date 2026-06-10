from __future__ import annotations

import importlib.util
import inspect
import os
import re
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from genesys.manager.adapters import (
    DataProviderAdapter,
    InMemorySaveResultsAdapter,
    LinearRegressionAdapter,
)
from genesys.manager.manager import (
    DEFAULT_RUN_TYPE,
    RUN_TYPE_RE,
    ContractValidationError,
    Manager,
    RunOptions,
    validate_pipeline_config,
)
from genesys.storage.run_store import RunStore

# Directory holding one YAML per runtime_type.
# Override with PIPELINE_CONFIGS_DIR env var (useful in tests or Docker).
PIPELINE_CONFIGS_DIR = os.getenv("PIPELINE_CONFIGS_DIR", "pipeline_configs")

# Directory for user-supplied plugin components (one .py file per component).
PLUGINS_DIR = Path(os.getenv("PLUGINS_DIR", "plugins"))

# Cap on concurrent node executions per pipeline level.
MANAGER_MAX_WORKERS = int(os.getenv("MANAGER_MAX_WORKERS", "8"))

# SQLite file recording every pipeline run (see genesys.storage.run_store).
RUNS_DB_PATH = os.getenv("GENESYS_RUNS_DB", "data/genesys_runs.db")

# Built-in infrastructure components — hidden from the pipeline node dropdown.
_INFRASTRUCTURE = {"data_provider", "save_results"}


# ---------------------------------------------------------------------------
# API key gate
#
# Pipelines and components are runtime-editable data — that is the product's
# core differentiator, and also its biggest liability: POST /v2/components
# executes caller-supplied Python. When GENESYS_API_KEY is set, every mutating
# endpoint (and any 'production' run) requires a matching X-API-Key header.
# When unset (local dev), everything stays open.
# ---------------------------------------------------------------------------

def _check_api_key(provided: str | None, action: str) -> None:
    required = os.getenv("GENESYS_API_KEY")
    if not required:
        return  # dev mode — no key configured
    if not provided or not secrets.compare_digest(provided, required):
        raise HTTPException(status_code=401, detail=f"{action} requires a valid X-API-Key")


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------

def _load_plugins(orch: Manager) -> list[str]:
    """Scan PLUGINS_DIR for .py files and register any class that has run() or execute().

    - Classes with run(**inputs) -> dict are auto-wrapped: the system adds all
      LRO tracking and error handling invisibly. Users only write their logic.
    - Classes with execute(*, lro_collector, node_id, **inputs) -> dict are
      registered as-is (advanced use / backward compatibility).

    The component name is the filename stem (e.g. my_model.py → my_model).
    Files are loaded fresh each call so saving a new version picks up immediately.
    """
    if not PLUGINS_DIR.exists():
        return []
    loaded: list[str] = []
    for py_file in sorted(PLUGINS_DIR.glob("*.py")):
        name = py_file.stem
        try:
            spec = importlib.util.spec_from_file_location(name, py_file)
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            for _cls_name, obj in inspect.getmembers(module, inspect.isclass):
                if callable(getattr(obj, "execute", None)):
                    # Already conforms to the adapter contract — register directly.
                    orch.register(name, obj())
                    loaded.append(name)
                    break
                if callable(getattr(obj, "run", None)):
                    # User plugin — wrap automatically.
                    orch.register(name, _AutoAdapter(name, obj()))
                    loaded.append(name)
                    break
        except Exception as exc:
            print(f"[plugins] failed to load {py_file.name}: {exc}")
    return loaded


class _AutoAdapter:
    """Wraps a user plugin class (with run()) into the orchestrator adapter contract.

    Users only write:
        class MyModel:
            def run(self, **inputs) -> dict:
                return {"result": ...}

    This wrapper handles LRO tracking and error handling automatically.
    """

    def __init__(self, name: str, instance: Any) -> None:
        self._name = name
        self._instance = instance

    def execute(self, *, lro_collector: Any, node_id: str, **inputs: Any) -> dict:
        from genesys.manager.lro import LROCollector
        op_id = lro_collector.start(
            node_id=node_id, component=self._name, kind="run", ref="v1",
        )
        try:
            result = self._instance.run(**inputs)
            if not isinstance(result, dict):
                raise TypeError(f"run() must return a dict, got {type(result).__name__}")
            lro_collector.complete(op_id)
        except Exception as exc:
            lro_collector.fail(op_id, str(exc))
            raise
        return result


_COMPONENT_SKELETON = '''\
class {class_name}:
    """Describe what this component does."""

    def run(self, **inputs) -> dict:
        # inputs come from the pipeline YAML inputs: block.
        # Return a dict — every key is reachable as ${{{name}.output.key}}.
        #
        # Examples:
        #   data  = inputs["data"]          # list of dicts from a previous node
        #   label = inputs.get("label", "default")
        #   result = some_calculation(data)
        #   return {{"score": result}}
        return {{}}
'''


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class _AppState:
    manager: Manager | None = None
    save_results_adapter: InMemorySaveResultsAdapter | None = None
    run_store: RunStore | None = None

state = _AppState()



@asynccontextmanager
async def lifespan(app: FastAPI):
    orch = Manager(config_dir=PIPELINE_CONFIGS_DIR, max_workers=MANAGER_MAX_WORKERS)
    orch.register("data_provider", DataProviderAdapter())
    orch.register("linear_regression", LinearRegressionAdapter())
    save_adapter = InMemorySaveResultsAdapter()
    orch.register("save_results", save_adapter)
    plugins = _load_plugins(orch)
    state.manager = orch
    state.save_results_adapter = save_adapter
    state.run_store = RunStore(db_path=RUNS_DB_PATH)
    print(f"[startup] manager runtimes: {orch.list_runtime_types()}")
    if plugins:
        print(f"[startup] plugins loaded: {plugins}")
    yield


app = FastAPI(title="genesys", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v2/runtimes")
def list_runtimes() -> dict[str, list[str]]:
    """List available runtime_type pipeline configs."""
    if state.manager is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialised")
    return {"runtime_types": state.manager.list_runtime_types()}


@app.get("/v2/components")
def list_v2_components() -> dict[str, Any]:
    """List components registered with the v2 orchestrator.

    Returns two lists:
      - ``all``: every registered component name
      - ``user``: components available for use in pipeline nodes
        (infrastructure components like data_provider/save_results excluded)
      - ``plugins``: names that came from the plugins/ folder
    """
    if state.manager is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialised")
    all_components = state.manager.list_registered()
    plugin_names = sorted(p.stem for p in PLUGINS_DIR.glob("*.py")) if PLUGINS_DIR.exists() else []
    return {
        "components": all_components,
        "user": [c for c in all_components if c not in _INFRASTRUCTURE],
        "plugins": plugin_names,
    }


@app.post("/v2/pipelines/{runtime_type}/run")
async def run_v2(
    runtime_type: str,
    request: Request,
    async_mode: bool = Query(default=False, alias="async"),
    x_debug_component_list: str | None = Header(default=None, alias="X-Debug-Component-List"),
    x_fail_fast: str | None = Header(default=None, alias="X-Fail-Fast"),
    x_trace_failures: str | None = Header(default=None, alias="X-Trace-Failures"),
    x_run_type: str | None = Header(default=None, alias="X-Run-Type"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Run a pipeline by runtime_type with caller-supplied JSON.

    Every run is recorded with a run_id (see GET /v2/runs). With ?async=true
    the request returns 202 + {run_id} immediately and the pipeline executes
    in the background — poll GET /v2/runs/{run_id} for status and results.

    Headers (all optional):
      - X-Debug-Component-List: comma-separated component names; only those run.
      - X-Fail-Fast: 'true' to stop scheduling once any node fails.
      - X-Trace-Failures: 'true' to include Python tracebacks in failed outcomes.
      - X-Run-Type: classification label for this run (default 'test'). Free
        choice — teams can define their own labels (e.g. 'team_a_check') to
        verify their changes against the output data. Must be a lowercase slug
        (letters, digits, '_', '-'). 'production' is the one reserved label
        downstream consumers treat as customer-grade; it must always be
        asserted explicitly and is never a default.

    Body: arbitrary JSON — accessible inside the pipeline as ${request.<field>}.
    """
    if state.manager is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialised")

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="request body must be a JSON object")

    debug_set: set[str] | None = None
    if x_debug_component_list:
        debug_set = {p.strip() for p in x_debug_component_list.split(",") if p.strip()}

    run_type = (x_run_type or DEFAULT_RUN_TYPE).strip().lower()
    if not RUN_TYPE_RE.match(run_type):
        raise HTTPException(
            status_code=422,
            detail=(
                f"invalid X-Run-Type '{x_run_type}' — must be a lowercase slug "
                "(letters, digits, '_', '-'; starting with a letter; max 64 chars)"
            ),
        )
    if run_type == "production":
        # 'production' marks output as customer-grade/verified — when an API
        # key is configured, asserting it requires authentication.
        _check_api_key(x_api_key, "a production run")

    options = RunOptions(
        debug_components=debug_set,
        fail_fast=(x_fail_fast or "").lower() == "true",
        trace_failures=(x_trace_failures or "").lower() == "true",
        run_type=run_type,
    )

    run_id = str(uuid.uuid4())

    if async_mode:
        # Validate everything we can up-front so async callers get an
        # immediate 4xx instead of a run record that errors out.
        try:
            violations = state.manager.validate_request(runtime_type, payload)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if violations:
            raise HTTPException(status_code=422, detail={"message": "request body failed contract validation", "violations": violations})

        state.run_store.create_run(run_id, runtime_type, run_type, payload)
        manager, run_store = state.manager, state.run_store

        def _execute_in_background() -> None:
            try:
                result = manager.run(runtime_type, payload, options)
                run_store.complete_run(run_id, result.to_dict())
            except Exception as exc:  # infra failure — record it for the poller
                run_store.fail_run(run_id, f"{type(exc).__name__}: {exc}")

        threading.Thread(target=_execute_in_background, daemon=True).start()
        return JSONResponse(
            status_code=202,
            content={"run_id": run_id, "status": "running", "poll": f"/v2/runs/{run_id}"},
        )

    try:
        # Run the (synchronous, potentially long) pipeline in a worker thread so
        # the event loop stays free to serve other requests (incl. /health).
        result = await run_in_threadpool(state.manager.run, runtime_type, payload, options)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ContractValidationError as exc:
        raise HTTPException(status_code=422, detail={"message": "request body failed contract validation", "violations": exc.violations})

    result_dict = result.to_dict()
    result_dict["run_id"] = run_id
    state.run_store.create_run(run_id, runtime_type, run_type, payload)
    state.run_store.complete_run(run_id, result_dict)
    return result_dict


@app.get("/v2/runs")
def list_runs(
    runtime_type: str | None = None,
    run_type: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Run history (newest first). Filter by runtime_type, run_type, status.

    run_type filtering is the point: downstream consumers ask for
    ?run_type=production to see only customer-grade runs, while a team asks
    for ?run_type=team_a_check to review their own verification runs.
    """
    if state.run_store is None:
        raise HTTPException(status_code=503, detail="run store not initialised")
    return {"runs": state.run_store.list_runs(runtime_type=runtime_type, run_type=run_type, status=status, limit=limit)}


@app.get("/v2/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """Full record for one run: status, request, per-node resolved inputs and outputs."""
    if state.run_store is None:
        raise HTTPException(status_code=503, detail="run store not initialised")
    record = state.run_store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no run with id '{run_id}'")
    return record


@app.get("/v2/pipelines/{runtime_type}/contract")
def get_pipeline_contract(runtime_type: str) -> dict[str, Any]:
    """Return the input contract for a pipeline.

    Upstream systems call this to know exactly what JSON shape to POST.
    Returns an empty contract list if the pipeline accepts any payload.
    """
    if state.manager is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialised")
    try:
        return state.manager.get_contract(runtime_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Pipeline config CRUD  (read / write YAML files)
# ---------------------------------------------------------------------------

_RUNTIME_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _config_path(runtime_type: str) -> Path:
    return Path(PIPELINE_CONFIGS_DIR) / f"{runtime_type}.yaml"


@app.get("/v2/pipelines/{runtime_type}/config")
def get_pipeline_config(runtime_type: str) -> dict[str, Any]:
    """Return the pipeline config for *runtime_type* as JSON."""
    path = _config_path(runtime_type)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no config for '{runtime_type}'")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@app.put("/v2/pipelines/{runtime_type}/config")
async def put_pipeline_config(
    runtime_type: str,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Overwrite the YAML for *runtime_type* from a JSON body.

    The body must be a JSON object with a ``nodes`` list (and optionally a
    ``contract`` list). The config is fully validated before writing —
    unknown components, duplicate ids, dependency cycles, and malformed
    blocks are rejected with 422 so a broken pipeline can never be saved.
    The next pipeline run uses the new config.
    """
    _check_api_key(x_api_key, "editing a pipeline config")
    if not _RUNTIME_NAME_RE.match(runtime_type):
        raise HTTPException(status_code=422, detail="runtime_type must be lowercase letters, digits, underscores, starting with a letter")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="body must be valid JSON")

    if not isinstance(body, dict) or "nodes" not in body:
        raise HTTPException(status_code=422, detail="body must be a JSON object with a 'nodes' key")

    registered = set(state.manager.list_registered()) if state.manager else None
    problems = validate_pipeline_config(body, registered_components=registered)
    if problems:
        raise HTTPException(status_code=422, detail={"message": "pipeline config is invalid", "problems": problems})

    config: dict[str, Any] = {"runtime_type": runtime_type}
    if body.get("contract"):
        config["contract"] = body["contract"]
    config["nodes"] = body["nodes"]
    path = _config_path(runtime_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return {"saved": True, "runtime_type": runtime_type, "node_count": len(body["nodes"])}


@app.post("/v2/pipelines/{runtime_type}/validate")
async def validate_pipeline(runtime_type: str, request: Request) -> dict[str, Any]:
    """Dry-run validation of a pipeline config — same checks as PUT, no write.

    Always returns 200 with {valid, problems} so UIs can validate live while
    the user edits.
    """
    try:
        body = await request.json()
    except Exception:
        return {"valid": False, "problems": ["body must be valid JSON"]}
    if not isinstance(body, dict):
        return {"valid": False, "problems": ["body must be a JSON object"]}

    registered = set(state.manager.list_registered()) if state.manager else None
    problems = validate_pipeline_config(body, registered_components=registered)
    return {"valid": not problems, "problems": problems}


@app.delete("/v2/pipelines/{runtime_type}/config")
def delete_pipeline_config(
    runtime_type: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Delete the YAML for *runtime_type*."""
    _check_api_key(x_api_key, "deleting a pipeline config")
    path = _config_path(runtime_type)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no config for '{runtime_type}'")
    path.unlink()
    return {"deleted": True, "runtime_type": runtime_type}


# ---------------------------------------------------------------------------
# Component plugin CRUD
# ---------------------------------------------------------------------------

_PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_component_source(source: str) -> str | None:
    """Execute *source* in an isolated namespace and check it defines a usable
    component class. Returns an error message, or None if valid.

    This runs the caller's code — which is the deliberate design of the plugin
    system (see _check_api_key for how it is gated outside local dev).
    """
    try:
        compiled = compile(source, "<component>", "exec")
        namespace: dict[str, Any] = {}
        exec(compiled, namespace)  # noqa: S102 — intentional, see docstring
    except Exception as exc:
        return f"source failed to load — {type(exc).__name__}: {exc}"

    for obj in namespace.values():
        if inspect.isclass(obj) and (
            callable(getattr(obj, "execute", None)) or callable(getattr(obj, "run", None))
        ):
            try:
                obj()
            except Exception as exc:
                return f"component class could not be instantiated — {type(exc).__name__}: {exc}"
            return None
    return (
        "source must define a class with a run(**inputs) -> dict method "
        "(or an execute(*, lro_collector, node_id, **inputs) -> dict method)"
    )


@app.get("/v2/components/skeleton")
def get_component_skeleton(name: str = "my_component") -> dict[str, Any]:
    """Return a starter skeleton for a new plugin component."""
    class_name = "".join(p.capitalize() for p in name.split("_"))
    return {"name": name, "source": _COMPONENT_SKELETON.format(name=name, class_name=class_name)}


@app.get("/v2/components/{name}/source")
def get_component_source(name: str) -> dict[str, Any]:
    """Return the Python source for a plugin component."""
    path = PLUGINS_DIR / f"{name}.py"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"'{name}' is a built-in component with no editable source")
    return {"name": name, "source": path.read_text(encoding="utf-8"), "is_plugin": True}


@app.post("/v2/components")
async def create_or_update_component(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Write a plugin component from Python source and register it immediately.

    The source is validated (it must load and define a class with run() or
    execute()) BEFORE anything is written, so a broken save can never knock
    out a previously working component.
    """
    _check_api_key(x_api_key, "creating or updating a component")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="body must be valid JSON")

    name = (body.get("name") or "").strip()
    source = (body.get("source") or "").strip()

    if not _PLUGIN_NAME_RE.match(name):
        raise HTTPException(status_code=422, detail="name must be lowercase letters, digits, underscores, starting with a letter")
    if name in _INFRASTRUCTURE:
        raise HTTPException(status_code=422, detail=f"'{name}' is a reserved built-in component name")
    if not source:
        raise HTTPException(status_code=422, detail="source cannot be empty")

    error = _validate_component_source(source)
    if error is not None:
        raise HTTPException(status_code=422, detail=error)

    if state.manager is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialised")

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLUGINS_DIR / f"{name}.py"
    path.write_text(source, encoding="utf-8")

    loaded = _load_plugins(state.manager)
    if name not in loaded:
        raise HTTPException(status_code=500, detail=f"component '{name}' validated but failed to register")

    return {"registered": name}


@app.delete("/v2/components/{name}")
def delete_component(
    name: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Delete a plugin component."""
    _check_api_key(x_api_key, "deleting a component")
    if name in _INFRASTRUCTURE:
        raise HTTPException(status_code=422, detail=f"'{name}' is a built-in and cannot be deleted")
    path = PLUGINS_DIR / f"{name}.py"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no plugin named '{name}'")
    path.unlink()
    return {"deleted": name}


# ---------------------------------------------------------------------------
# Frontend UI
# ---------------------------------------------------------------------------

_UI_FILE = Path(__file__).parent / "ui" / "index.html"


@app.get("/ui", response_class=HTMLResponse)
@app.get("/ui/", response_class=HTMLResponse)
def ui() -> FileResponse:
    return FileResponse(str(_UI_FILE))
