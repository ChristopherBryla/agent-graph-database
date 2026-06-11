# Genesys — Next Steps Plan

Working plan for upcoming sessions. Tasks are tagged by who should do them:

- **[small]** — well-scoped, mechanical, or pattern-following. Hand to a
  cheaper/older model. Each one includes everything needed to execute cold.
- **[big]** — design decisions, cross-cutting changes, or anything touching
  the scheduler/positioning. Use a stronger model and review carefully.

Ground rules for ALL sessions (paste into context if the model can't see repo memory):

- Validate with `.\.venv\Scripts\python.exe -m pytest -q` — 100 tests must stay green.
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
Decide schema + flow before any code: a `models` table (model_id, name,
version, blob pointer via existing BlobStore/HybridStore, created_by_run_id).
Keep it lite — versioned save/load only; MLflow integration if outgrown.
Output of this task is a short design section in README, not code.

### C2. [small] Implement per C1's written design
Only after C1 lands in README. SaveModelAdapter + LoadModelAdapter following
src/genesys/manager/adapters.py patterns, plus a `scoring_predict.yaml`
example pipeline.

---

## Milestone D — Historical replay / backfill (roadmap #4)

### D1. [big] Batch driver design
Fan one pipeline over N time windows: request shape, concurrency (reuses the
ready-set scheduler per run; cap concurrent runs), how N runs link to one
backfill_id, partial-failure semantics. Design doc first (README section).

### D2. [small] DataProvider time windows
Add optional `start`/`end` inputs to DataProviderAdapter (pass through to
sample source as a filter; csv/database still NotImplementedError). Additive,
contract-style validation, tests.

---

## Suggested order

1. B1, B2 (small-model warm-ups, de-risk the delegation workflow itself)
2. A1 → A2 (big) → A3, A4 (small) — ships the differentiator
3. B3, B4 anytime as filler
4. C1 (big design) → C2 (small impl)
5. D1 (big design) → D2 (small impl)

## How to brief a small-model session

Paste: the task block from this file + the ground rules at the top. That's
deliberate — each [small] task names its files, its pattern to copy, and its
acceptance criteria so no architectural context is required.
