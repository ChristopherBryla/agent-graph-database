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
