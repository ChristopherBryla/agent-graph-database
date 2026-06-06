from __future__ import annotations

import importlib.util
import inspect
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from genesys.manager.adapters import (
    DataProviderAdapter,
    InMemorySaveResultsAdapter,
    LinearRegressionAdapter,
)
from genesys.manager.manager import Manager, RunOptions, ContractValidationError

# Directory holding one YAML per runtime_type.
# Override with PIPELINE_CONFIGS_DIR env var (useful in tests or Docker).
PIPELINE_CONFIGS_DIR = os.getenv("PIPELINE_CONFIGS_DIR", "pipeline_configs")

# Directory for user-supplied plugin components (one .py file per component).
PLUGINS_DIR = Path(os.getenv("PLUGINS_DIR", "plugins"))

# Built-in infrastructure components — hidden from the pipeline node dropdown.
_INFRASTRUCTURE = {"data_provider", "save_results"}


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

state = _AppState()



@asynccontextmanager
async def lifespan(app: FastAPI):
    orch = Manager(config_dir=PIPELINE_CONFIGS_DIR)
    orch.register("data_provider", DataProviderAdapter())
    orch.register("linear_regression", LinearRegressionAdapter())
    save_adapter = InMemorySaveResultsAdapter()
    orch.register("save_results", save_adapter)
    plugins = _load_plugins(orch)
    state.manager = orch
    state.save_results_adapter = save_adapter
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
    x_debug_component_list: str | None = Header(default=None, alias="X-Debug-Component-List"),
    x_fail_fast: str | None = Header(default=None, alias="X-Fail-Fast"),
    x_trace_failures: str | None = Header(default=None, alias="X-Trace-Failures"),
) -> dict[str, Any]:
    """Run a pipeline by runtime_type with caller-supplied JSON.

    Headers (all optional):
      - X-Debug-Component-List: comma-separated component names; only those run.
      - X-Fail-Fast: 'true' to stop scheduling once any node fails.
      - X-Trace-Failures: 'true' to include Python tracebacks in failed outcomes.

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

    options = RunOptions(
        debug_components=debug_set,
        fail_fast=(x_fail_fast or "").lower() == "true",
        trace_failures=(x_trace_failures or "").lower() == "true",
    )

    try:
        result = state.manager.run(runtime_type, payload, options)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ContractValidationError as exc:
        raise HTTPException(status_code=422, detail={"message": "request body failed contract validation", "violations": exc.violations})

    return result.to_dict()


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
async def put_pipeline_config(runtime_type: str, request: Request) -> dict[str, Any]:
    """Overwrite the YAML for *runtime_type* from a JSON body.

    The body must be a JSON object with at least a ``nodes`` list.
    Each node must have ``id`` and ``component``.
    The file is written immediately; the next pipeline run uses the new config.
    """
    if not _RUNTIME_NAME_RE.match(runtime_type):
        raise HTTPException(status_code=422, detail="runtime_type must be lowercase letters, digits, underscores, starting with a letter")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="body must be valid JSON")

    if not isinstance(body, dict) or "nodes" not in body:
        raise HTTPException(status_code=422, detail="body must be a JSON object with a 'nodes' key")

    nodes = body["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise HTTPException(status_code=422, detail="'nodes' must be a non-empty list")

    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or not node.get("id") or not node.get("component"):
            raise HTTPException(status_code=422, detail=f"node[{i}] must have 'id' and 'component'")

    config = {"runtime_type": runtime_type, "nodes": nodes}
    path = _config_path(runtime_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return {"saved": True, "runtime_type": runtime_type, "node_count": len(nodes)}


@app.delete("/v2/pipelines/{runtime_type}/config")
def delete_pipeline_config(runtime_type: str) -> dict[str, Any]:
    """Delete the YAML for *runtime_type*."""
    path = _config_path(runtime_type)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no config for '{runtime_type}'")
    path.unlink()
    return {"deleted": True, "runtime_type": runtime_type}


# ---------------------------------------------------------------------------
# Component plugin CRUD
# ---------------------------------------------------------------------------

_PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


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
async def create_or_update_component(request: Request) -> dict[str, Any]:
    """Write a plugin component from Python source and register it immediately."""
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

    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLUGINS_DIR / f"{name}.py"
    path.write_text(source, encoding="utf-8")

    if state.manager is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialised")

    try:
        _load_plugins(state.manager)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"component loaded with error: {exc}")

    return {"registered": name}


@app.delete("/v2/components/{name}")
def delete_component(name: str) -> dict[str, Any]:
    """Delete a plugin component."""
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
