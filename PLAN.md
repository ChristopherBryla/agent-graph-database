# Genesys — Next Steps Plan

Working plan for upcoming sessions. Tasks are tagged by who should do them:

- **[small]** — well-scoped, mechanical, or pattern-following. Hand to a
  cheaper/older model. Each one includes everything needed to execute cold.
- **[big]** — design decisions, cross-cutting changes, or anything touching
  the scheduler/positioning. Use a stronger model and review carefully.

Ground rules for ALL sessions (paste into context if the model can't see repo memory):

- Validate with `.\.venv\Scripts\python.exe -m pytest -q` — 96 tests must stay green.
- Tests use tmp_-prefixed runtime/component names + cleanup fixtures (see tests/test_api.py patterns).
- Do NOT touch src/genesys/ui/index.html.
- README.md is the canonical architecture doc — update it when behavior changes.
- Feature rule: if a platform engineer would love it but a data scientist
  wouldn't understand it, it belongs in an integration, not core.

---

## Milestone A — Node dev loop (roadmap #2) — the proof-point feature

The daily driver: a data scientist owns one node, edits its code, re-runs it
against a recorded run's pinned inputs, and diffs the output. Everything it
needs (per-node resolved inputs/outputs in RunStore) already exists.

### A1. [big] Replay endpoint design + implementation
Status: ✅ completed (June 10, 2026)

Decision lock (June 11, 2026):
- Keep replay runs visible in default run history (no hidden-by-default filtering).
- Keep replay strict-only: if recorded inputs were lossy, replay is rejected (no force mode).

`POST /v2/runs/{run_id}/replay/{node_id}` — load the recorded resolved inputs
for that node from RunStore, execute ONLY that node's component (current,
possibly hot-reloaded version), return the new output. No upstream execution.
Decisions to make: how to handle inputs that were repr-stringified by
`_json_safe` (probably reject replay for nodes with non-JSON inputs, with a
clear message); whether replays are themselves recorded (suggest yes, with
run_type `replay` and a `parent_run_id` column).

### A2. [big] Output diff
Status: ✅ completed (June 10, 2026)

Extend A1 response with a structural diff: `{added, removed, changed}` keyed
by JSON path, comparing new output vs the recorded output. Keep it dependency-
free (recursive dict/list compare). NaN→null normalization already happens at
record time, so compare the sanitized forms.

### A3. [small] Replay tests
Status: ✅ completed (June 10, 2026)

Follow tests/test_api.py patterns (module-scoped client). Cases: replay of a
completed node returns 200 + diff; unknown run_id → 404; unknown node_id →
404; node whose component no longer exists → 422 or 409 with clear message.

### A4. [small] README update for the dev loop
Status: ✅ completed (June 11, 2026)

Document the replay endpoint + diff in the API surface table and replace the
"north-star workflow 1" description with the concrete curl flow:
edit plugin → POST /v2/components → POST replay → read diff.

---

## Milestone B — Hygiene / small chores (good warm-ups for a small model)

### B1. [small] Structured logging
Status: ✅ completed (June 11, 2026)

Replace the `print(...)` calls in src/genesys/manager/manager.py and
src/genesys/app.py with the `logging` module (logger per module, INFO level,
same messages). No new dependencies. Tests must still pass (none assert on
stdout).

### B2. [small] .gitignore for artifacts
Status: ✅ completed (June 11, 2026)

Add/extend .gitignore: `__pycache__/`, `*.pyc`, `data/genesys_runs.db`,
`.pytest_cache/`. The repo currently commits .pyc files — remove them from
tracking (`git rm --cached`), but ask Chris before pushing.

### B3. [small] Run record timings
Status: ✅ completed (June 11, 2026)

LRO durations are already captured. Add `started_at`/`finished_at` per node
row in RunStore (record at execute start/end inside Manager — NodeOutcome
gains two optional float fields, threaded into run_nodes columns). Additive,
follow the existing NodeOutcome → complete_run flow.

### B4. [small] GET /v2/runs pagination polish
Status: ✅ completed (June 11, 2026)

Add `offset` param to RunStore.list_runs + endpoint, and return
`{"runs": [...], "count": N}`. Mechanical; mirror the existing limit clamp.

---

## Milestone C — Model lifecycle: train/predict split (roadmap #3)

### C1. [big] Design the minimal model registry
Status: ✅ completed (June 11, 2026)

Decide schema + flow before any code: a `models` table (model_id, name,
version, blob pointer via existing BlobStore/HybridStore, created_by_run_id).
Keep it lite — versioned save/load only; MLflow integration if outgrown.
Output of this task is a short design section in README, not code.

### C2. [small] Implement per C1's written design
Status: ✅ completed (June 11, 2026)

Only after C1 lands in README. SaveModelAdapter + LoadModelAdapter following
src/genesys/manager/adapters.py patterns, plus a `scoring_predict.yaml`
example pipeline.

---

## Milestone D — Historical replay / backfill (roadmap #4)

### D1. [big] Batch driver design
Status: ✅ completed (June 11, 2026)

Fan one pipeline over N time windows: request shape, concurrency (reuses the
ready-set scheduler per run; cap concurrent runs), how N runs link to one
backfill_id, partial-failure semantics. Design doc first (README section).

### D2. [small] DataProvider time windows
Status: ✅ completed (June 11, 2026)

Add optional `start`/`end` inputs to DataProviderAdapter (pass through to
sample source as a filter; csv/database still NotImplementedError). Additive,
contract-style validation, tests.

---

## Audit note (June 11, 2026)

Full A–D audit passed (15 live probes). One bug found + fixed: GET
/v2/runs/{run_id} returned nodes in alphabetical order, not pipeline order
(ORDER BY recorded_at degenerated — all nodes share one timestamp). Fixed to
ORDER BY rowid in run_store.py. Three tests added (node order regression,
lossy-replay 422, replay-run persistence): suite now 115.

---

## Milestone E — Backfill driver implementation (roadmap #4, D1 design → code)

The D1 design is fully written in README ("Backfill driver design (D1)") —
implement exactly that. D2's DataProvider windows are the per-run input
mechanism. Vision tie-in: "historical simulations" / "reprocessing".

### E1. [small] Backfill schema in RunStore
Status: ✅ completed (June 11, 2026)

Per README design: new `backfills` table + additive `runs` columns
(backfill_id, backfill_window_id, window_start, window_end). Follow the
existing `_ensure_column` migration pattern in
src/genesys/storage/run_store.py. Index runs.backfill_id. Pure storage —
no endpoint yet. Acceptance: schema created on init, existing tests green.

### E2. [big] Backfill driver + POST /v2/backfills
Status: ✅ completed (June 11, 2026)

Only after E1. The fan-out engine per README: child payload =
{**base_payload, start, end, window_id}; run-level concurrency cap
(max_concurrent_runs default 4, clamp 1..32) on top of node-level
MANAGER_MAX_WORKERS; failure_mode continue|stop; backfill status rules
(completed / completed_with_errors / failed); run_type default "historical".
Decisions in play: driver thread model (suggest one coordinator thread +
bounded pool of pipeline runs), idempotency (client_request_id — optional,
may defer), cancellation (may defer; document if so).

### E3. [small] Backfill read endpoints + tests
Status: ✅ completed (June 11, 2026)

GET /v2/backfills/{id} (status + window summary) and
GET /v2/backfills/{id}/runs (child run metadata). Tests follow
tests/test_api.py patterns: happy path over 3 windows, one bad window →
completed_with_errors with failure_mode=continue, stop mode halts early,
unknown backfill 404. tmp_-prefixed names + cleanup.

### E4. [small] README roadmap update
Status: ✅ completed (June 11, 2026)

Mark roadmap #4 done; move the D1 design section's "planned" wording to
present tense; add endpoints to the API surface table.

---

## Milestone F — Monitoring + scheduling (vision gaps: "no clock", "metrics go in, nothing comes out")

Product-vision alignment (June 11 review): timings/statuses are already
recorded but nothing analyzes them, and the platform only acts when called.
These two close the "continuously running" gap.

### F1. [big] Monitoring summary design
Status: ✅ completed (June 11, 2026)

Design-first (README section, like C1/D1). A read-only aggregate endpoint,
e.g. GET /v2/runtimes/{runtime_type}/stats: run counts by status/run_type,
failure rate, per-node duration aggregates (avg/p95 from run_nodes
started_at/finished_at), window param (last N runs or since timestamp).
Decide: SQL aggregates only (suggest yes — no new deps), and whether model
metrics from the registry's `metrics` field join in now or later.

### F2. [small] Implement per F1's written design
Status: ✅ completed (June 11, 2026)

Only after F1 lands in README. SQL aggregation queries in RunStore +
endpoint + tests (seed a few runs incl. one failure, assert counts/rates).

### F3. [big] Scheduler design
Status: ✅ completed (June 11, 2026)

The vision's word is "continuously" — pipelines need a clock. Design-first
(README section). Decisions: schedule storage (suggest a `schedules` table:
schedule_id, runtime_type, cron-ish interval, payload, run_type, enabled);
in-process timer thread vs external trigger (suggest in-process daemon
thread polling due schedules — single-process platform, keep it honest);
overlap policy (skip if previous still running vs queue); how scheduled
runs are labeled (suggest run_type "scheduled" or schedule-defined).

### F4. [small] Implement per F3's written design
Status: ✅ completed (June 11, 2026)

Schedules CRUD endpoints + the timer loop + tests. All 6 endpoints working
(POST /v2/schedules, GET /v2/schedules, GET /v2/schedules/{id}, PUT, DELETE,
POST run-now). Daemon thread polls every 1 second, respects overlap_policy
(skip vs queue), advances next_run_at after each execution. Test baseline:
96 tests (10 new scheduler tests added).

---

## Milestone G — Data ingestion (vision component 1 — design conversation first)

Chris (June 11): clients already have source data in many forms — blob
storage, SQL databases, even Excel files — and it needs a path INTO the
platform. Unresolved on purpose; G1 is a thinking task, not a coding task.

### G1. [big] Ingestion design discussion + decision lock
Status: not started

Work through with Chris before any code. Questions to resolve:
- Connector model: do sources stay live-pull (DataProvider fetches at run
  time, like today) or does ingestion LAND data into platform storage
  (BlobStore/SQLite) as versioned datasets that pipelines then reference?
  (Likely answer: both, but landed datasets are what enable backfill
  reproducibility, lineage, and offline clients.)
- Source coverage order: CSV/Excel files first (csv source already stubs
  NotImplementedError), SQL via connection string second, blob storage third.
- Schema handling at the boundary: reuse the pipeline contract mechanism for
  dataset validation? Where do quality checks run — at ingest or at fetch?
- Addressability: dataset_id + version (mirrors the model registry pattern);
  how ${...} references name a dataset in pipeline YAML.
- Security: SQL connection strings are credentials — storage and redaction.
Output: README design section + a decision lock in this file (C1/D1 pattern).

### G2. [small] First connector per G1's design
Status: not started — BLOCKED until G1's decision lock exists

Scope intentionally unknown until G1 resolves. Expected shape: implement the
`csv` source path (file upload or path-based) exactly as G1 specifies, with
contract-style validation + tests. Do not start from this block alone.

---

## Suggested order

1. ~~B1, B2 (small-model warm-ups, de-risk the delegation workflow itself)~~ ✅
2. ~~A1 → A2 (big) → A3, A4 (small) — ships the differentiator~~ ✅
3. ~~B3, B4 anytime as filler~~ ✅
4. ~~C1 (big design) → C2 (small impl)~~ ✅
5. ~~D1 (big design) → D2 (small impl)~~ ✅
6. E1 (small) → E2 (big) → E3, E4 (small) — ships backfill, design already done
7. F1 (big design) → F2 (small impl) — monitoring reads data we already record
8. G1 (big — discussion with Chris) in parallel with F anytime; G2 only after
9. F3 (big design) → F4 (small impl) — scheduler last of F; benefits from G1 clarity

## How to brief a small-model session

Paste: the task block from this file + the ground rules at the top. That's
deliberate — each [small] task names its files, its pattern to copy, and its
acceptance criteria so no architectural context is required.
