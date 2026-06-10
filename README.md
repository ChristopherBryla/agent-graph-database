# Genesys

Genesys turns ML models into reliable business products. It handles the data
pipelines, orchestration, monitoring, and operational plumbing required to make
ML continuously deliver value — replacing custom infrastructure with a
ready-to-use system.

## Positioning (read this before adding features)

Genesys is **not** an MLflow rebuild. The target user is a data scientist who
has a working model (Python, R, anything) but no MLOps expertise, and just
needs it running in a production-ready state. Design rules that follow:

- **Developer experience is the product.** The user writes a `run()` method
  (or stands up a model behind HTTP) — the platform supplies orchestration,
  contracts, LRO tracking, and error handling invisibly. Every feature should
  reduce, never grow, what a data scientist must learn.
- **Stay lite and opinionated.** Persist models and runs with sensible
  defaults rather than building general-purpose registries/trackers. If a team
  outgrows Genesys, integrate with MLflow — don't clone it.
- **Language-agnostic via HTTP components.** Models written in R (or any
  language) participate as out-of-process components behind an HTTP contract;
  the docker-compose layout anticipates this.

### How this differs from MLflow / Dagster / Prefect

- **Pipelines and components are data, not code deploys.** Incumbent tools
  require editing Python in a repo + CI + redeploy. Genesys pipelines are YAML
  edited over HTTP; components are hot-reloaded source. A data scientist
  changes their node from a browser and it's live.
- **Audience inversion.** Dagster/Prefect serve engineers (learn assets, ops,
  resources, deployments first). Genesys serves data scientists: write `run()`
  or answer HTTP — that's the whole contract.
- **The node dev loop is the proof point** (roadmap #2): edit your node,
  instantly diff output against a recorded run. No incumbent offers this as a
  first-class experience.
- **Feature decision rule:** if a platform engineer would love it but a data
  scientist wouldn't understand it, it belongs in an integration, not in
  Genesys. (Backfills especially: Dagster's flagship feature — win on
  simplicity, never on capability.)

### North-star workflows

The two workflows the platform exists to serve (modeled on the real product
this is based on):

1. **Node-scoped development loop.** A data scientist owns a single node in
   the pipeline. They edit only their code and immediately see how the change
   affects output data — by re-running their node against the *recorded
   outputs of a previous run* (pinned upstream inputs, no need to re-run the
   whole pipeline) and diffing new output vs old.
2. **Historical replay / backfill.** Rerun years of historical data through a
   pipeline (e.g. after a model change) by fanning the same pipeline out over
   many time windows. Retraining is the recurring special case of this.

Both workflows sit on the same foundation: **recorded runs as first-class,
addressable objects** (run_id, persisted config + per-node inputs/outputs).
That is why run persistence is roadmap item #1.

## Architecture at a glance

```
HTTP request (JSON)
      │
      ▼
FastAPI gateway (src/genesys/app.py)          ── /v2/* endpoints + /ui
      │  runs pipeline in a worker thread so the event loop stays free
      ▼
Manager / orchestrator (src/genesys/manager/manager.py)
      │  1. loads pipeline_configs/<runtime_type>.yaml
      │  2. validates request body against the pipeline contract
      │  3. validates the dependency graph (cycles, unknown deps)
      │  4. ready-set scheduling: each node runs as soon as ALL of its
      │     dependencies finish (ThreadPoolExecutor, capped worker count)
      │  5. resolves ${...} references from the shared run_context
      ▼
Components (looked up by name in a registry)
      ├─ data_provider      → DataProviderAdapter → DataProvider
      ├─ linear_regression  → LinearRegressionAdapter → LinearRegression
      ├─ save_results       → InMemorySaveResultsAdapter
      └─ plugins/*.py       → user components, auto-wrapped at load time
```

Key modules:

| Path | Purpose |
| --- | --- |
| `src/genesys/app.py` | FastAPI gateway: pipeline run/CRUD, component CRUD, plugin loading, UI |
| `src/genesys/manager/manager.py` | Orchestrator: DAG scheduling, contract validation, `${...}` resolution |
| `src/genesys/manager/adapters.py` | Thin wrappers that adapt existing components to the orchestrator contract |
| `src/genesys/manager/lro.py` | `LROCollector` — tracks long-running operations across a run |
| `src/genesys/dataprovider/` | Data access layer (sample/csv/database sources) |
| `src/genesys/contract/` | Data contracts for model inputs/outputs |
| `src/genesys/storage/` | Blob / SQLite / hybrid persistence |
| `pipeline_configs/` | One YAML per `runtime_type` |
| `plugins/` | User-supplied components (one `.py` file each) |

## Pipeline YAML format

One file per `runtime_type` in `pipeline_configs/`. Example
(`scoring_v2.yaml` — trains two models in parallel):

```yaml
runtime_type: scoring_v2

contract:                      # optional — validates the request body
  - name: source
    type: str                  # list | dict | float | int | str | any
    required: true

nodes:
  - id: fetch
    component: data_provider
    inputs:
      source: "${request.source}"

  - id: train_a
    component: linear_regression
    depends_on: [fetch]
    inputs:
      data:    "${fetch.output.data}"
      targets: "${fetch.output.targets}"
      epochs: 300

  - id: train_b                # no dependency on train_a → runs in parallel
    component: linear_regression
    depends_on: [fetch]
    inputs:
      data:    "${fetch.output.data}"
      targets: "${fetch.output.targets}"
      learning_rate: 0.05
      epochs: 300

  - id: persist
    component: save_results
    depends_on: [train_a, train_b]
    wait_for_lros: true        # block dependents until this node's async work finishes
    inputs:
      weights: "${train_a.output.weights}"
```

Reference syntax inside `inputs:`:

- `${request.<field>}` — a field from the POSTed JSON body
- `${<node_id>.output.<field>}` — output of a completed upstream node
- A whole-string reference preserves the original type (list/dict/number);
  references embedded in longer strings are coerced to `str`.

## Parallel execution model

- The orchestrator uses a **ready-set scheduler**: a node is submitted to the
  thread pool the moment all of its `depends_on` nodes have finished. A slow
  node only delays its own dependents — unrelated branches keep flowing.
- All execution happens in one `ThreadPoolExecutor` capped at
  `MANAGER_MAX_WORKERS` threads (env var, default 8).
- A preprocessing node that every downstream node depends on naturally acts as
  a gate: it runs first, then everything after it fans out in parallel.
- `wait_for_lros: true` on a node means the node only counts as finished (and
  its dependents only become ready) once all of its registered long-running
  operations reach a terminal state. The wait happens inside the node's worker
  thread, so other branches are unaffected.
- `X-Fail-Fast`: once any node fails, no *new* nodes are scheduled; in-flight
  nodes finish, and everything never scheduled is reported as `skipped`.

### Known design decisions / deferred work

- **Ready-set scheduling** (done): replaced the original level-based scheduler
  so a slow sibling can no longer delay nodes whose own dependencies are done.
- **GIL / CPU-bound training** (deferred until performance matters): threads
  give concurrency, not CPU parallelism, for pure-Python training. Options when
  needed: numpy-vectorized models, `ProcessPoolExecutor`, or out-of-process
  model services (docker-compose already points this way).
- **Event loop protection** (done): `POST /v2/pipelines/{rt}/run` executes the
  synchronous pipeline via `run_in_threadpool` so `/health` and other requests
  are never blocked by a long run.

## Roadmap (priority order)

1. **Run persistence + async run API** — runs currently live only inside one
   HTTP request. `POST /run` should return a `run_id` immediately, execute in
   the background, and expose `GET /v2/runs/{run_id}` for status/results plus a
   run-history listing. Persist per-node inputs/outputs so recorded runs can be
   used as fixtures (see north-star workflows), and include the `run_type`
   column (already threaded through the manager/API) so production data is
   distinguishable from test/training/historical runs. The storage layer
   (`src/genesys/storage/`) already exists; this is mostly wiring. Everything
   below depends on this.
2. **Node dev loop: pinned replay + output diff** — run a single node against
   the recorded run_context of a previous run (no upstream re-execution), and
   diff its output against what that run produced. This is the data
   scientist's daily driver; `X-Debug-Component-List` and plugin hot-reload
   are the existing building blocks.
3. **Model lifecycle: separate train from predict** — persist trained model
   artifacts with a version (minimal, opinionated — see Positioning), then add
   predict pipelines that load a stored model instead of retraining every call.
4. **Historical replay / backfill** — time-windowed data access in the
   DataProvider (`start`/`end` alongside `entity_id`) plus a batch driver that
   fans one pipeline out over N windows (naturally parallel under the
   ready-set scheduler). Scheduled retraining falls out of this.
5. **HTTP component adapter (language-agnostic components)** — a generic
   adapter that calls any model service behind an HTTP contract. This is what
   lets R (or any non-Python) models participate, unifies the in-process vs
   docker-compose (ports 8001/8002) split, and sidesteps the GIL since model
   calls become I/O-bound.
6. **Monitoring** — replace `print` with structured logging, record per-node
   timings in the run record (LRO durations already capture most of it), add a
   run-history endpoint. Drift/model-performance tracking comes after run
   history exists.
7. **Security** — partially done: `GENESYS_API_KEY` gates all mutating
   endpoints and production runs (see "Live editing & the API key gate").
   Still open: per-user/team identity (so run_type labels and component
   ownership map to real teams), and a sandboxing decision for plugin
   execution before multi-tenant deployment.
8. **Real data sources** — implement the `csv` and `database` DataProvider
   sources (currently `NotImplementedError`).

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| GET | `/v2/runtimes` | List available runtime types (one per YAML) |
| POST | `/v2/pipelines/{runtime_type}/run` | Run a pipeline with a JSON body |
| GET | `/v2/pipelines/{runtime_type}/contract` | Input contract for upstream callers |
| GET/PUT/DELETE | `/v2/pipelines/{runtime_type}/config` | Pipeline YAML CRUD (PUT fully validates before writing) |
| POST | `/v2/pipelines/{runtime_type}/validate` | Dry-run config validation — same checks as PUT, no write |
| GET | `/v2/components` | List registered components (`all` / `user` / `plugins`) |
| GET | `/v2/components/skeleton` | Starter source for a new plugin |
| GET | `/v2/components/{name}/source` | Read plugin source |
| POST | `/v2/components` | Create/update a plugin from source (registers immediately) |
| DELETE | `/v2/components/{name}` | Delete a plugin |
| GET | `/ui` | Browser UI |

Optional headers on `/run`:

- `X-Run-Type`: classification label for the run (default `test`). Teams
  choose their own labels — e.g. `team_a_check`, `chris_experiment`,
  `training`, `historical` — so multiple teams can run their own verification
  passes and find their output data downstream. Echoed in the response,
  exposed to pipelines as `${run.run_type}`, and stored on every run record
  once run persistence lands. **Rules:** labels must be lowercase slugs
  (letters, digits, `_`, `-`; 422 otherwise — a stray space or typo'd format
  can't create unqueryable values), and `production` is the single reserved
  label downstream business consumers treat as customer-grade/verified. It is
  never a default — callers must assert it explicitly; everything else is
  non-production by definition.
- `X-Debug-Component-List`: comma-separated component names; only those run,
  everything else is reported as `skipped`.
- `X-Fail-Fast`: `true` → stop scheduling new levels once any node fails.
- `X-Trace-Failures`: `true` → include Python tracebacks in failed outcomes.

Environment variables:

- `PIPELINE_CONFIGS_DIR` (default `pipeline_configs`)
- `PLUGINS_DIR` (default `plugins`)
- `MANAGER_MAX_WORKERS` (default `8`) — per-level thread cap
- `GENESYS_API_KEY` (unset by default) — see "Live editing & the API key gate"

## Live editing & the API key gate

Pipelines and components are runtime-editable data (the core differentiator).
Two mechanisms keep that loop safe:

**Save-time validation.** A broken artifact can never be saved:

- `PUT …/config` validates the full pipeline before writing — unknown
  components (with the list of registered ones in the error), duplicate node
  ids, dependency cycles, unknown `depends_on` targets, malformed
  `inputs`/`contract` blocks → 422 with a `problems` list. The `contract`
  block is preserved on save.
- `POST …/validate` runs the same checks without writing, so UIs can
  validate live while the user edits.
- `POST /v2/components` loads and inspects the source in isolation BEFORE
  writing: syntax errors, import failures, missing `run()`/`execute()` class,
  or a class that can't be instantiated → 422 with the actual error. A broken
  save can never knock out a previously working component.

**API key gate.** When `GENESYS_API_KEY` is set, every mutating endpoint
(config PUT/DELETE, component POST/DELETE) and any run asserting
`X-Run-Type: production` requires a matching `X-API-Key` header (401
otherwise). When unset, everything is open — intended for local dev only.
Remember: `POST /v2/components` executes caller-supplied Python by design;
never expose the API beyond localhost without setting the key.

## Writing plugin components

Drop a `.py` file in `plugins/` (or POST to `/v2/components`). The component
name is the filename stem. Minimal form:

```python
class MyComponent:
    def run(self, **inputs) -> dict:
        # inputs come from the pipeline YAML inputs: block
        return {"score": 0.9}   # reachable as ${<node_id>.output.score}
```

The system auto-wraps `run()` with LRO tracking and error handling. Advanced
plugins may instead implement the full contract directly:

```python
def execute(self, *, lro_collector, node_id, **inputs) -> dict: ...
```

**Thread-safety requirement:** a single component instance is shared across
all pipeline runs and may be called from multiple threads at once (parallel
nodes in the same level). Do not store per-call mutable state on `self` —
keep everything local to `run()`/`execute()`, or guard shared state with a
lock.

## Getting started (local)

1. Activate the virtual environment (`.venv`).
2. Install: `python -m pip install -e .` and `python -m pip install pytest`
3. Run tests: `python -m pytest`
4. Start the API: `uvicorn genesys.app:app --reload`
5. Open `http://localhost:8000/ui` or `http://localhost:8000/docs`

Example run:

```bash
curl -X POST http://localhost:8000/v2/pipelines/scoring_v2/run \
  -H "Content-Type: application/json" \
  -d '{"source": "sample"}'
```

## Run with Docker

`docker compose up --build` starts three services:

- `genesys-api` (port 8000) — gateway / ComponentManager
- `genesys-data-provider` (port 8001) — DataProvider microservice
- `genesys-linear-regression` (port 8002) — stateless model service

`./data` is mounted into the API container so SQLite writes survive restarts.
