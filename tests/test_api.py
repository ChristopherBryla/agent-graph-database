"""API tests for the v2 orchestrator endpoints.

Covers HTTP behaviour: status codes, response structure, header processing.
All tests run entirely in-process via FastAPI TestClient — no Docker needed.
"""

import time

import pytest
from fastapi.testclient import TestClient

from genesys.app import app, state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_body(self, client):
        assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_runtimes_returns_200(self, client):
        assert client.get("/v2/runtimes").status_code == 200

    def test_runtimes_lists_known_types(self, client):
        types = set(client.get("/v2/runtimes").json()["runtime_types"])
        assert {"scoring_v1", "scoring_v2"} <= types

    def test_components_returns_200(self, client):
        assert client.get("/v2/components").status_code == 200

    def test_components_lists_all_adapters(self, client):
        comps = set(client.get("/v2/components").json()["components"])
        assert {"data_provider", "linear_regression", "save_results"} <= comps


# ---------------------------------------------------------------------------
# Happy path — POST /v2/pipelines/{runtime_type}/run
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_scoring_v1_returns_200(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert r.status_code == 200

    def test_scoring_v1_succeeded(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert r.json()["succeeded"] is True

    def test_runtime_type_in_response(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert r.json()["runtime_type"] == "scoring_v1"

    def test_nodes_present(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert len(r.json()["nodes"]) == 3

    def test_all_nodes_have_required_fields(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        for node in r.json()["nodes"]:
            assert {"node_id", "component", "status", "output", "message"} <= node.keys()

    def test_all_nodes_completed(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        statuses = {n["status"] for n in r.json()["nodes"]}
        assert statuses == {"completed"}

    def test_lros_present(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert len(r.json()["lros"]) > 0

    def test_lros_all_completed(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        for op in r.json()["lros"]:
            assert op["state"] == "completed", f"LRO {op['op_id']} was {op['state']}"

    def test_unknown_runtime_returns_404(self, client):
        r = client.post("/v2/pipelines/no_such_type/run", json={"source": "sample"})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Parallel execution — scoring_v2
# ---------------------------------------------------------------------------

class TestParallelExecution:
    def test_scoring_v2_returns_200(self, client):
        r = client.post("/v2/pipelines/scoring_v2/run", json={"source": "sample"})
        assert r.status_code == 200

    def test_scoring_v2_both_train_nodes_complete(self, client):
        r = client.post("/v2/pipelines/scoring_v2/run", json={"source": "sample"})
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert nodes["train_a"]["status"] == "completed"
        assert nodes["train_b"]["status"] == "completed"

    def test_scoring_v2_has_four_lros(self, client):
        # fetch + train_a + train_b + persist = 4 LROs
        r = client.post("/v2/pipelines/scoring_v2/run", json={"source": "sample"})
        assert len(r.json()["lros"]) == 4

    def test_scoring_v2_parallel_lros_node_ids(self, client):
        r = client.post("/v2/pipelines/scoring_v2/run", json={"source": "sample"})
        node_ids = {op["node_id"] for op in r.json()["lros"]}
        assert node_ids == {"fetch", "train_a", "train_b", "persist"}


# ---------------------------------------------------------------------------
# X-Debug-Component-List header
# ---------------------------------------------------------------------------

class TestDebugFilter:
    def test_only_listed_component_runs(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Debug-Component-List": "data_provider"},
            json={"source": "sample"},
        )
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert nodes["fetch"]["status"] == "completed"
        assert nodes["train"]["status"] == "skipped"
        assert nodes["persist"]["status"] == "skipped"

    def test_skipped_message_explains_reason(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Debug-Component-List": "data_provider"},
            json={"source": "sample"},
        )
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert "X-Debug-Component-List" in nodes["train"]["message"]

    def test_multiple_components_in_list(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Debug-Component-List": "data_provider,linear_regression"},
            json={"source": "sample"},
        )
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert nodes["fetch"]["status"] == "completed"
        assert nodes["train"]["status"] == "completed"
        assert nodes["persist"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# X-Fail-Fast and X-Trace-Failures headers
# ---------------------------------------------------------------------------

class TestFailureBehavior:
    def test_missing_field_fails_fetch_node(self, client):
        # No 'source' field in body → ${request.source} cannot resolve
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Fail-Fast": "true"},
            json={},
        )
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert nodes["fetch"]["status"] == "failed"

    def test_fail_fast_skips_downstream(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Fail-Fast": "true"},
            json={},
        )
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert nodes["train"]["status"] == "skipped"
        assert nodes["persist"]["status"] == "skipped"

    def test_fail_fast_skip_message(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Fail-Fast": "true"},
            json={},
        )
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert "fail-fast" in nodes["train"]["message"]

    def test_succeeded_false_on_failure(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={})
        assert r.json()["succeeded"] is False

    def test_trace_failures_adds_traceback(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Fail-Fast": "true", "X-Trace-Failures": "true"},
            json={},
        )
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert nodes["fetch"].get("traceback") is not None

    def test_no_trace_by_default(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={})
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        failed = next((n for n in nodes.values() if n["status"] == "failed"), None)
        if failed:
            assert failed.get("traceback") is None


# ---------------------------------------------------------------------------
# X-Run-Type header
# ---------------------------------------------------------------------------

class TestRunType:
    def test_defaults_to_test(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert r.json()["run_type"] == "test"

    def test_production_is_explicit(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Run-Type": "production"},
            json={"source": "sample"},
        )
        assert r.json()["run_type"] == "production"

    def test_custom_team_label_accepted(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Run-Type": "team_a_check"},
            json={"source": "sample"},
        )
        assert r.status_code == 200
        assert r.json()["run_type"] == "team_a_check"

    def test_case_insensitive(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Run-Type": "Training"},
            json={"source": "sample"},
        )
        assert r.json()["run_type"] == "training"

    def test_malformed_run_type_rejected(self, client):
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Run-Type": "team a!!"},  # spaces/punctuation must not pass
            json={"source": "sample"},
        )
        assert r.status_code == 422
        assert "X-Run-Type" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Save-time pipeline config validation
# ---------------------------------------------------------------------------

_VALID_NODES = [
    {"id": "fetch", "component": "data_provider", "inputs": {"source": "sample"}},
    {
        "id": "train",
        "component": "linear_regression",
        "depends_on": ["fetch"],
        "inputs": {"data": "${fetch.output.data}", "targets": "${fetch.output.targets}"},
    },
]


class TestPipelineConfigValidation:
    RT = "tmp_cfg_validation"

    @pytest.fixture(autouse=True)
    def _cleanup(self, client):
        yield
        client.delete(f"/v2/pipelines/{self.RT}/config")

    def test_unknown_component_rejected(self, client):
        r = client.put(
            f"/v2/pipelines/{self.RT}/config",
            json={"nodes": [{"id": "a", "component": "does_not_exist"}]},
        )
        assert r.status_code == 422
        assert any("unknown component" in p for p in r.json()["detail"]["problems"])

    def test_duplicate_node_id_rejected(self, client):
        r = client.put(
            f"/v2/pipelines/{self.RT}/config",
            json={"nodes": [
                {"id": "a", "component": "data_provider"},
                {"id": "a", "component": "data_provider"},
            ]},
        )
        assert r.status_code == 422
        assert any("duplicate node id" in p for p in r.json()["detail"]["problems"])

    def test_dependency_cycle_rejected(self, client):
        r = client.put(
            f"/v2/pipelines/{self.RT}/config",
            json={"nodes": [
                {"id": "a", "component": "data_provider", "depends_on": ["b"]},
                {"id": "b", "component": "data_provider", "depends_on": ["a"]},
            ]},
        )
        assert r.status_code == 422
        assert any("circular" in p for p in r.json()["detail"]["problems"])

    def test_valid_config_saves(self, client):
        r = client.put(f"/v2/pipelines/{self.RT}/config", json={"nodes": _VALID_NODES})
        assert r.status_code == 200
        assert r.json()["saved"] is True

    def test_contract_block_is_preserved(self, client):
        contract = [{"name": "source", "type": "str", "required": True}]
        r = client.put(
            f"/v2/pipelines/{self.RT}/config",
            json={"contract": contract, "nodes": _VALID_NODES},
        )
        assert r.status_code == 200
        saved = client.get(f"/v2/pipelines/{self.RT}/config").json()
        assert saved["contract"] == contract

    def test_validate_endpoint_reports_problems_without_writing(self, client):
        r = client.post(
            f"/v2/pipelines/{self.RT}/validate",
            json={"nodes": [{"id": "a", "component": "nope"}]},
        )
        assert r.status_code == 200
        assert r.json()["valid"] is False
        assert r.json()["problems"]
        # nothing was written
        assert client.get(f"/v2/pipelines/{self.RT}/config").status_code == 404

    def test_validate_endpoint_accepts_valid_config(self, client):
        r = client.post(f"/v2/pipelines/{self.RT}/validate", json={"nodes": _VALID_NODES})
        assert r.status_code == 200
        assert r.json() == {"valid": True, "problems": []}


# ---------------------------------------------------------------------------
# Component source validation
# ---------------------------------------------------------------------------

class TestComponentValidation:
    NAME = "tmp_component_validation"

    @pytest.fixture(autouse=True)
    def _cleanup(self, client):
        yield
        client.delete(f"/v2/components/{self.NAME}")

    def test_syntax_error_rejected_and_not_written(self, client):
        r = client.post(
            "/v2/components",
            json={"name": self.NAME, "source": "class Broken(:\n    pass"},
        )
        assert r.status_code == 422
        assert "failed to load" in r.json()["detail"]
        # nothing was written
        assert client.get(f"/v2/components/{self.NAME}/source").status_code == 404

    def test_source_without_component_class_rejected(self, client):
        r = client.post(
            "/v2/components",
            json={"name": self.NAME, "source": "x = 1"},
        )
        assert r.status_code == 422
        assert "must define a class" in r.json()["detail"]

    def test_valid_component_registers_and_runs(self, client):
        source = (
            "class TmpComponent:\n"
            "    def run(self, **inputs) -> dict:\n"
            "        return {'echo': inputs.get('value', 'none')}\n"
        )
        r = client.post("/v2/components", json={"name": self.NAME, "source": source})
        assert r.status_code == 200
        assert r.json() == {"registered": self.NAME}
        comps = client.get("/v2/components").json()["components"]
        assert self.NAME in comps

    def test_broken_save_does_not_break_existing_component(self, client):
        good = "class Good:\n    def run(self, **inputs) -> dict:\n        return {'v': 1}\n"
        assert client.post("/v2/components", json={"name": self.NAME, "source": good}).status_code == 200
        # try to overwrite with broken source — must be rejected pre-write
        bad = "import nonexistent_module_xyz"
        r = client.post("/v2/components", json={"name": self.NAME, "source": bad})
        assert r.status_code == 422
        # original source is intact
        src = client.get(f"/v2/components/{self.NAME}/source").json()["source"]
        assert "class Good" in src


# ---------------------------------------------------------------------------
# API key gate (GENESYS_API_KEY)
# ---------------------------------------------------------------------------

class TestApiKeyGate:
    RT = "tmp_apikey_test"

    def test_mutations_open_when_no_key_configured(self, client, monkeypatch):
        monkeypatch.delenv("GENESYS_API_KEY", raising=False)
        r = client.put(f"/v2/pipelines/{self.RT}/config", json={"nodes": _VALID_NODES})
        assert r.status_code == 200
        assert client.delete(f"/v2/pipelines/{self.RT}/config").status_code == 200

    def test_config_put_requires_key(self, client, monkeypatch):
        monkeypatch.setenv("GENESYS_API_KEY", "sekrit")
        r = client.put(f"/v2/pipelines/{self.RT}/config", json={"nodes": _VALID_NODES})
        assert r.status_code == 401

    def test_config_put_accepts_valid_key(self, client, monkeypatch):
        monkeypatch.setenv("GENESYS_API_KEY", "sekrit")
        r = client.put(
            f"/v2/pipelines/{self.RT}/config",
            headers={"X-API-Key": "sekrit"},
            json={"nodes": _VALID_NODES},
        )
        assert r.status_code == 200
        assert client.delete(
            f"/v2/pipelines/{self.RT}/config", headers={"X-API-Key": "sekrit"}
        ).status_code == 200

    def test_component_post_requires_key(self, client, monkeypatch):
        monkeypatch.setenv("GENESYS_API_KEY", "sekrit")
        r = client.post("/v2/components", json={"name": "tmp_x", "source": "x = 1"})
        assert r.status_code == 401

    def test_wrong_key_rejected(self, client, monkeypatch):
        monkeypatch.setenv("GENESYS_API_KEY", "sekrit")
        r = client.put(
            f"/v2/pipelines/{self.RT}/config",
            headers={"X-API-Key": "wrong"},
            json={"nodes": _VALID_NODES},
        )
        assert r.status_code == 401

    def test_production_run_requires_key(self, client, monkeypatch):
        monkeypatch.setenv("GENESYS_API_KEY", "sekrit")
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Run-Type": "production"},
            json={"source": "sample"},
        )
        assert r.status_code == 401

    def test_production_run_with_key_succeeds(self, client, monkeypatch):
        monkeypatch.setenv("GENESYS_API_KEY", "sekrit")
        r = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Run-Type": "production", "X-API-Key": "sekrit"},
            json={"source": "sample"},
        )
        assert r.status_code == 200
        assert r.json()["run_type"] == "production"

    def test_non_production_runs_stay_open(self, client, monkeypatch):
        monkeypatch.setenv("GENESYS_API_KEY", "sekrit")
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Run persistence + history — GET /v2/runs
# ---------------------------------------------------------------------------

class TestRunPersistence:
    def test_sync_response_includes_run_id(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert r.status_code == 200
        assert r.json()["run_id"]

    def test_sync_run_is_retrievable(self, client):
        run_id = client.post(
            "/v2/pipelines/scoring_v1/run", json={"source": "sample"}
        ).json()["run_id"]
        record = client.get(f"/v2/runs/{run_id}").json()
        assert record["run_id"] == run_id
        assert record["status"] == "completed"
        assert record["succeeded"] is True
        assert record["runtime_type"] == "scoring_v1"

    def test_record_keeps_request_payload(self, client):
        run_id = client.post(
            "/v2/pipelines/scoring_v1/run", json={"source": "sample", "marker": "abc"}
        ).json()["run_id"]
        record = client.get(f"/v2/runs/{run_id}").json()
        assert record["request"] == {"source": "sample", "marker": "abc"}

    def test_record_keeps_resolved_node_inputs_and_outputs(self, client):
        run_id = client.post(
            "/v2/pipelines/scoring_v1/run", json={"source": "sample"}
        ).json()["run_id"]
        record = client.get(f"/v2/runs/{run_id}").json()
        completed = [n for n in record["nodes"] if n["status"] == "completed"]
        assert completed
        for node in completed:
            assert node["inputs"] is not None
            assert node["output"] is not None

    def test_record_keeps_node_timings(self, client):
        run_id = client.post(
            "/v2/pipelines/scoring_v1/run", json={"source": "sample"}
        ).json()["run_id"]
        record = client.get(f"/v2/runs/{run_id}").json()
        completed = [n for n in record["nodes"] if n["status"] == "completed"]
        assert completed
        for node in completed:
            assert isinstance(node["started_at"], float)
            assert isinstance(node["finished_at"], float)
            assert node["finished_at"] >= node["started_at"]

    def test_record_preserves_pipeline_node_order(self, client):
        # Regression: ORDER BY recorded_at degenerated to alphabetical order
        # (fetch, persist, train) because every node shares one recorded_at.
        run_id = client.post(
            "/v2/pipelines/scoring_v1/run", json={"source": "sample"}
        ).json()["run_id"]
        record = client.get(f"/v2/runs/{run_id}").json()
        assert [n["node_id"] for n in record["nodes"]] == ["fetch", "train", "persist"]

    def test_unknown_run_id_returns_404(self, client):
        assert client.get("/v2/runs/no-such-run").status_code == 404

    def test_list_runs_filters_by_run_type(self, client):
        label = "team_persist_check"
        run_id = client.post(
            "/v2/pipelines/scoring_v1/run",
            headers={"X-Run-Type": label},
            json={"source": "sample"},
        ).json()["run_id"]
        runs = client.get(f"/v2/runs?run_type={label}").json()["runs"]
        assert run_id in {r["run_id"] for r in runs}
        assert all(r["run_type"] == label for r in runs)

    def test_list_runs_filters_by_runtime_type(self, client):
        client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        runs = client.get("/v2/runs?runtime_type=scoring_v1").json()["runs"]
        assert runs
        assert all(r["runtime_type"] == "scoring_v1" for r in runs)

    def test_list_runs_respects_limit(self, client):
        for _ in range(3):
            client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert len(client.get("/v2/runs?limit=2").json()["runs"]) == 2

    def test_list_runs_includes_count(self, client):
        label = "team_count_check"
        for _ in range(3):
            client.post(
                "/v2/pipelines/scoring_v1/run",
                headers={"X-Run-Type": label},
                json={"source": "sample"},
            )
        body = client.get(f"/v2/runs?run_type={label}&limit=2").json()
        assert body["count"] >= 3
        assert len(body["runs"]) == 2

    def test_list_runs_supports_offset(self, client):
        label = "team_offset_check"
        created: list[str] = []
        for _ in range(3):
            run_id = client.post(
                "/v2/pipelines/scoring_v1/run",
                headers={"X-Run-Type": label},
                json={"source": "sample"},
            ).json()["run_id"]
            created.append(run_id)

        first = client.get(f"/v2/runs?run_type={label}&limit=1&offset=0").json()
        second = client.get(f"/v2/runs?run_type={label}&limit=1&offset=1").json()

        assert first["count"] >= 3
        assert second["count"] == first["count"]
        assert len(first["runs"]) == 1
        assert len(second["runs"]) == 1
        assert first["runs"][0]["run_id"] != second["runs"][0]["run_id"]
        assert first["runs"][0]["run_id"] in created
        assert second["runs"][0]["run_id"] in created


# ---------------------------------------------------------------------------
# Async runs — POST /v2/pipelines/{rt}/run?async=true
# ---------------------------------------------------------------------------

class TestAsyncRun:
    @staticmethod
    def _wait_for_finish(client, run_id, attempts=100):
        import time

        for _ in range(attempts):
            record = client.get(f"/v2/runs/{run_id}").json()
            if record["status"] != "running":
                return record
            time.sleep(0.05)
        raise AssertionError(f"run {run_id} never left 'running'")

    def test_returns_202_with_run_id(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run?async=true", json={"source": "sample"})
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "running"
        assert body["poll"] == f"/v2/runs/{body['run_id']}"

    def test_run_completes_in_background(self, client):
        run_id = client.post(
            "/v2/pipelines/scoring_v1/run?async=true", json={"source": "sample"}
        ).json()["run_id"]
        record = self._wait_for_finish(client, run_id)
        assert record["status"] == "completed"
        assert record["succeeded"] is True
        assert len(record["nodes"]) == 3

    def test_async_keeps_run_type(self, client):
        run_id = client.post(
            "/v2/pipelines/scoring_v1/run?async=true",
            headers={"X-Run-Type": "team_async_check"},
            json={"source": "sample"},
        ).json()["run_id"]
        record = self._wait_for_finish(client, run_id)
        assert record["run_type"] == "team_async_check"

    def test_unknown_runtime_rejected_immediately(self, client):
        r = client.post("/v2/pipelines/no_such_rt/run?async=true", json={})
        assert r.status_code == 404

    def test_contract_violation_rejected_immediately(self, client):
        # scoring_production declares a contract; an empty body must fail it
        # up-front instead of producing an errored run record.
        r = client.post("/v2/pipelines/scoring_production/run?async=true", json={})
        assert r.status_code == 422
        assert "violations" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Backfills — POST /v2/backfills
# ---------------------------------------------------------------------------

class TestBackfills:
    RT = "tmp_backfill_runtime"

    @pytest.fixture(autouse=True)
    def _setup_runtime(self, client):
        cfg = {
            "nodes": [
                {
                    "id": "fetch",
                    "component": "data_provider",
                    "inputs": {
                        "source": "${request.source}",
                        "start": "${request.start}",
                        "end": "${request.end}",
                    },
                }
            ]
        }
        assert client.put(f"/v2/pipelines/{self.RT}/config", json=cfg).status_code == 200
        yield
        client.delete(f"/v2/pipelines/{self.RT}/config")

    @staticmethod
    def _wait_for_backfill(backfill_id: str, timeout_s: float = 5.0) -> dict:
        start = time.time()
        while time.time() - start < timeout_s:
            record = state.run_store.get_backfill(backfill_id)
            if record and record["status"] != "running":
                return record
            time.sleep(0.05)
        raise AssertionError(f"backfill {backfill_id} did not finish in time")

    def test_post_backfill_returns_202(self, client):
        payload = {
            "runtime_type": self.RT,
            "base_payload": {"source": "sample"},
            "windows": [
                {"window_id": "w1", "start": "2024-01-01T00:00:00Z", "end": "2024-01-03T00:00:00Z"}
            ],
        }
        r = client.post("/v2/backfills", json=payload)
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "running"
        assert body["runtime_type"] == self.RT
        assert body["total_windows"] == 1
        assert body["failure_mode"] == "continue"

    def test_backfill_continue_runs_all_windows_and_records_lineage(self, client):
        run_type = "tmp_backfill_continue"
        payload = {
            "runtime_type": self.RT,
            "run_type": run_type,
            "failure_mode": "continue",
            "max_concurrent_runs": 2,
            "base_payload": {"source": "sample"},
            "windows": [
                {"window_id": "ok_1", "start": "2024-01-01T00:00:00Z", "end": "2024-01-03T00:00:00Z"},
                {"window_id": "bad", "start": "not-iso", "end": "2024-01-04T00:00:00Z"},
                {"window_id": "ok_2", "start": "2024-01-05T00:00:00Z", "end": "2024-01-06T00:00:00Z"},
            ],
        }

        backfill_id = client.post("/v2/backfills", json=payload).json()["backfill_id"]
        summary = self._wait_for_backfill(backfill_id)
        assert summary["status"] == "completed_with_errors"
        assert summary["total_windows"] == 3
        assert summary["succeeded_windows"] == 2
        assert summary["failed_windows"] == 1

        runs = client.get(f"/v2/runs?runtime_type={self.RT}&run_type={run_type}&limit=500").json()["runs"]
        backfill_runs = [r for r in runs if r["backfill_id"] == backfill_id]
        assert len(backfill_runs) == 3
        assert {r["backfill_window_id"] for r in backfill_runs} == {"ok_1", "bad", "ok_2"}
        assert all(r["window_start"] is not None and r["window_end"] is not None for r in backfill_runs)

    def test_backfill_stop_halts_after_first_failure(self, client):
        run_type = "tmp_backfill_stop"
        payload = {
            "runtime_type": self.RT,
            "run_type": run_type,
            "failure_mode": "stop",
            "max_concurrent_runs": 1,
            "base_payload": {"source": "sample"},
            "windows": [
                {"window_id": "bad_first", "start": "not-iso", "end": "2024-01-04T00:00:00Z"},
                {"window_id": "would_have_run", "start": "2024-01-05T00:00:00Z", "end": "2024-01-06T00:00:00Z"},
            ],
        }

        backfill_id = client.post("/v2/backfills", json=payload).json()["backfill_id"]
        summary = self._wait_for_backfill(backfill_id)
        assert summary["status"] == "failed"
        assert summary["succeeded_windows"] == 0
        assert summary["failed_windows"] == 1

        runs = client.get(f"/v2/runs?runtime_type={self.RT}&run_type={run_type}&limit=500").json()["runs"]
        backfill_runs = [r for r in runs if r["backfill_id"] == backfill_id]
        assert len(backfill_runs) == 1
        assert backfill_runs[0]["backfill_window_id"] == "bad_first"

    def test_get_backfill_summary_endpoint(self, client):
        payload = {
            "runtime_type": self.RT,
            "run_type": "tmp_backfill_summary",
            "base_payload": {"source": "sample"},
            "windows": [
                {"window_id": "w1", "start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"},
                {"window_id": "w2", "start": "2024-01-02T00:00:00Z", "end": "2024-01-03T00:00:00Z"},
            ],
        }
        backfill_id = client.post("/v2/backfills", json=payload).json()["backfill_id"]
        summary = self._wait_for_backfill(backfill_id)

        r = client.get(f"/v2/backfills/{backfill_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["backfill_id"] == backfill_id
        assert body["runtime_type"] == self.RT
        assert body["total_windows"] == 2
        assert body["status"] == summary["status"]

    def test_get_backfill_runs_endpoint_paginates(self, client):
        payload = {
            "runtime_type": self.RT,
            "run_type": "tmp_backfill_runs_api",
            "base_payload": {"source": "sample"},
            "windows": [
                {"window_id": "w1", "start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"},
                {"window_id": "w2", "start": "2024-01-02T00:00:00Z", "end": "2024-01-03T00:00:00Z"},
                {"window_id": "w3", "start": "2024-01-03T00:00:00Z", "end": "2024-01-04T00:00:00Z"},
            ],
        }
        backfill_id = client.post("/v2/backfills", json=payload).json()["backfill_id"]
        self._wait_for_backfill(backfill_id)

        first = client.get(f"/v2/backfills/{backfill_id}/runs?limit=1&offset=0")
        second = client.get(f"/v2/backfills/{backfill_id}/runs?limit=1&offset=1")
        assert first.status_code == 200
        assert second.status_code == 200
        first_body = first.json()
        second_body = second.json()
        assert first_body["count"] == 3
        assert second_body["count"] == 3
        assert len(first_body["runs"]) == 1
        assert len(second_body["runs"]) == 1
        assert first_body["runs"][0]["run_id"] != second_body["runs"][0]["run_id"]

    def test_get_backfill_endpoints_unknown_id_returns_404(self, client):
        assert client.get("/v2/backfills/no-such-backfill").status_code == 404
        assert client.get("/v2/backfills/no-such-backfill/runs").status_code == 404


# ---------------------------------------------------------------------------
# Node replay + output diff — POST /v2/runs/{run_id}/replay/{node_id}
# ---------------------------------------------------------------------------

class TestReplayEndpoint:
    RT = "tmp_replay_missing_component"
    COMPONENT = "tmp_replay_component"
    LOSSY_RT = "tmp_replay_lossy"
    LOSSY_SOURCE = "tmp_replay_nan_source"
    LOSSY_SINK = "tmp_replay_nan_sink"

    @pytest.fixture(autouse=True)
    def _cleanup(self, client):
        yield
        client.delete(f"/v2/pipelines/{self.RT}/config")
        client.delete(f"/v2/components/{self.COMPONENT}")
        client.delete(f"/v2/pipelines/{self.LOSSY_RT}/config")
        client.delete(f"/v2/components/{self.LOSSY_SOURCE}")
        client.delete(f"/v2/components/{self.LOSSY_SINK}")

    def test_replay_completed_node_returns_diff(self, client):
        parent_run_id = client.post(
            "/v2/pipelines/scoring_v1/run", json={"source": "sample"}
        ).json()["run_id"]

        r = client.post(f"/v2/runs/{parent_run_id}/replay/train")
        assert r.status_code == 200
        body = r.json()
        assert body["parent_run_id"] == parent_run_id
        assert body["node_id"] == "train"
        assert body["status"] == "completed"
        assert set(body["diff"].keys()) == {"added", "removed", "changed"}

    def test_replay_run_is_persisted_with_parent_link(self, client):
        parent_run_id = client.post(
            "/v2/pipelines/scoring_v1/run", json={"source": "sample"}
        ).json()["run_id"]
        replay = client.post(f"/v2/runs/{parent_run_id}/replay/train").json()

        record = client.get(f"/v2/runs/{replay['run_id']}").json()
        assert record["status"] == "completed"
        assert record["run_type"] == "replay"
        assert record["parent_run_id"] == parent_run_id
        assert record["request"] == {
            "parent_run_id": parent_run_id,
            "replay_node_id": "train",
        }
        assert [n["node_id"] for n in record["nodes"]] == ["train"]

    def test_replay_lossy_inputs_returns_422(self, client):
        # A node whose recorded inputs required lossy coercion (NaN -> null)
        # must refuse replay: the pinned inputs no longer equal what ran.
        nan_source = (
            "class TmpReplayNanSource:\n"
            "    def run(self, **inputs) -> dict:\n"
            "        return {'value': float('nan')}\n"
        )
        sink = (
            "class TmpReplayNanSink:\n"
            "    def run(self, **inputs) -> dict:\n"
            "        return {'echo': str(inputs.get('value'))}\n"
        )
        assert client.post(
            "/v2/components", json={"name": self.LOSSY_SOURCE, "source": nan_source}
        ).status_code == 200
        assert client.post(
            "/v2/components", json={"name": self.LOSSY_SINK, "source": sink}
        ).status_code == 200

        cfg = {
            "nodes": [
                {"id": "make_nan", "component": self.LOSSY_SOURCE},
                {
                    "id": "sink",
                    "component": self.LOSSY_SINK,
                    "depends_on": ["make_nan"],
                    "inputs": {"value": "${make_nan.output.value}"},
                },
            ]
        }
        assert client.put(f"/v2/pipelines/{self.LOSSY_RT}/config", json=cfg).status_code == 200

        run_id = client.post(f"/v2/pipelines/{self.LOSSY_RT}/run", json={}).json()["run_id"]
        record = client.get(f"/v2/runs/{run_id}").json()
        sink_node = next(n for n in record["nodes"] if n["node_id"] == "sink")
        assert sink_node["inputs_lossy"] is True

        r = client.post(f"/v2/runs/{run_id}/replay/sink")
        assert r.status_code == 422
        assert "lossy" in r.json()["detail"]

    def test_replay_unknown_run_id_returns_404(self, client):
        r = client.post("/v2/runs/no-such-run/replay/train")
        assert r.status_code == 404

    def test_replay_unknown_node_id_returns_404(self, client):
        parent_run_id = client.post(
            "/v2/pipelines/scoring_v1/run", json={"source": "sample"}
        ).json()["run_id"]
        r = client.post(f"/v2/runs/{parent_run_id}/replay/no_such_node")
        assert r.status_code == 404

    def test_replay_component_no_longer_exists_returns_422(self, client):
        source = (
            "class TmpReplayComponent:\n"
            "    def run(self, **inputs) -> dict:\n"
            "        value = float(inputs.get('value', 0.0))\n"
            "        return {'pred': value + 1.0}\n"
        )
        assert client.post(
            "/v2/components", json={"name": self.COMPONENT, "source": source}
        ).status_code == 200

        cfg = {
            "nodes": [
                {
                    "id": "predict",
                    "component": self.COMPONENT,
                    "inputs": {"value": 2.0},
                }
            ]
        }
        assert client.put(f"/v2/pipelines/{self.RT}/config", json=cfg).status_code == 200

        parent_run_id = client.post(f"/v2/pipelines/{self.RT}/run", json={}).json()["run_id"]

        assert client.delete(f"/v2/components/{self.COMPONENT}").status_code == 200
        r = client.post(f"/v2/runs/{parent_run_id}/replay/predict")
        assert r.status_code == 422
        assert "not registered" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Model lifecycle (C2) — save_model / load_model / linear_predict
# ---------------------------------------------------------------------------

class TestModelLifecycle:
    RT = "tmp_model_registry"

    @pytest.fixture(autouse=True)
    def _cleanup(self, client):
        yield
        client.delete(f"/v2/pipelines/{self.RT}/config")

    @staticmethod
    def _config():
        return {
            "nodes": [
                {
                    "id": "fetch",
                    "component": "data_provider",
                    "inputs": {
                        "source": "passthrough",
                        "data": "${request.train_data}",
                        "targets": "${request.targets}",
                    },
                },
                {
                    "id": "train",
                    "component": "linear_regression",
                    "depends_on": ["fetch"],
                    "inputs": {
                        "data": "${fetch.output.data}",
                        "targets": "${fetch.output.targets}",
                        "learning_rate": 0.01,
                        "epochs": 50,
                    },
                },
                {
                    "id": "save",
                    "component": "save_model",
                    "depends_on": ["train"],
                    "inputs": {
                        "name": "${request.model_name}",
                        "artifact": {
                            "weights": "${train.output.weights}",
                            "bias": "${train.output.bias}",
                        },
                        "created_by_run_id": "${run.run_id}",
                        "metrics": {"epochs": "${train.output.epochs}"},
                    },
                },
                {
                    "id": "load",
                    "component": "load_model",
                    "depends_on": ["save"],
                    "inputs": {
                        "name": "${save.output.name}",
                        "version": "${save.output.version}",
                    },
                },
                {
                    "id": "predict",
                    "component": "linear_predict",
                    "depends_on": ["load"],
                    "inputs": {
                        "model": "${load.output.model}",
                        "data": "${request.predict_data}",
                    },
                },
            ]
        }

    @staticmethod
    def _payload(model_name: str):
        return {
            "model_name": model_name,
            "train_data": [
                {"x1": 1.0, "x2": 2.0, "x3": 3.0, "x4": 4.0},
                {"x1": 2.0, "x2": 3.0, "x3": 4.0, "x4": 5.0},
                {"x1": 3.0, "x2": 4.0, "x3": 5.0, "x4": 6.0},
            ],
            "targets": [10.0, 14.0, 18.0],
            "predict_data": [
                {"x1": 4.0, "x2": 5.0, "x3": 6.0, "x4": 7.0},
                {"x1": 5.0, "x2": 6.0, "x3": 7.0, "x4": 8.0},
            ],
        }

    def test_save_load_predict_flow(self, client):
        assert client.put(f"/v2/pipelines/{self.RT}/config", json=self._config()).status_code == 200
        r = client.post(f"/v2/pipelines/{self.RT}/run", json=self._payload("tmp_model_lifecycle_a"))
        assert r.status_code == 200
        nodes = {n["node_id"]: n for n in r.json()["nodes"]}
        assert nodes["save"]["status"] == "completed"
        assert nodes["load"]["status"] == "completed"
        assert nodes["predict"]["status"] == "completed"
        assert nodes["save"]["output"]["version"] == 1
        assert nodes["load"]["output"]["version"] == 1
        assert len(nodes["predict"]["output"]["predictions"]) == 2

    def test_model_version_increments(self, client):
        assert client.put(f"/v2/pipelines/{self.RT}/config", json=self._config()).status_code == 200
        first = client.post(
            f"/v2/pipelines/{self.RT}/run",
            json=self._payload("tmp_model_lifecycle_b"),
        ).json()
        second = client.post(
            f"/v2/pipelines/{self.RT}/run",
            json=self._payload("tmp_model_lifecycle_b"),
        ).json()

        first_nodes = {n["node_id"]: n for n in first["nodes"]}
        second_nodes = {n["node_id"]: n for n in second["nodes"]}
        assert first_nodes["save"]["output"]["version"] == 1
        assert second_nodes["save"]["output"]["version"] == 2
