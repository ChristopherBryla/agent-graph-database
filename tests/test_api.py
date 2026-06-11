"""API tests for the v2 orchestrator endpoints.

Covers HTTP behaviour: status codes, response structure, header processing.
All tests run entirely in-process via FastAPI TestClient — no Docker needed.
"""

import pytest
from fastapi.testclient import TestClient

from genesys.app import app


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
# Node replay + output diff — POST /v2/runs/{run_id}/replay/{node_id}
# ---------------------------------------------------------------------------

class TestReplayEndpoint:
    RT = "tmp_replay_missing_component"
    COMPONENT = "tmp_replay_component"

    @pytest.fixture(autouse=True)
    def _cleanup(self, client):
        yield
        client.delete(f"/v2/pipelines/{self.RT}/config")
        client.delete(f"/v2/components/{self.COMPONENT}")

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
