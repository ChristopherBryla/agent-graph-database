"""Functional tests — request payload flows through the full orchestrator pipeline.

These tests examine WHAT travels through the system at each stage: what data
the fetch node produces, what weights come out of the train node, how the
persist node picks up from train. They complement test_api.py (which checks
HTTP behaviour) by verifying the data flow and reference resolution inside
the pipeline itself.

Flow:
  POST /v2/pipelines/{runtime_type}/run  { "source": "sample" }
        │
        ▼
  Orchestrator resolves  ${request.source}  → "sample"
        │
        ▼
  fetch node  →  DataProviderAdapter  →  { data: [...], targets: [...] }
        │                                         stored in run_context["fetch"]
        ▼
  train node  resolves ${fetch.output.data} / ${fetch.output.targets}
              →  LinearRegressionAdapter  →  { weights: [...], bias: float }
        │                                         stored in run_context["train"]
        ▼
  persist node  resolves ${train.output.weights} / ${train.output.bias}
                →  InMemorySaveResultsAdapter  →  { label: ..., record_count: int }
"""

import pytest
from fastapi.testclient import TestClient

from genesys.app import app, state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _nodes(response_json: dict) -> dict[str, dict]:
    """Index response nodes by node_id for easy lookup."""
    return {n["node_id"]: n for n in response_json["nodes"]}


def _lros(response_json: dict) -> dict[str, dict]:
    """Index response LROs by node_id."""
    result: dict[str, dict] = {}
    for op in response_json["lros"]:
        result.setdefault(op["node_id"], op)
    return result


# ---------------------------------------------------------------------------
# 1. Node output shape — each adapter returns the right keys
# ---------------------------------------------------------------------------

class TestNodeOutputShape:
    def test_fetch_output_has_data(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        output = _nodes(r.json())["fetch"]["output"]
        assert "data" in output
        assert isinstance(output["data"], list)

    def test_fetch_output_has_targets(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        output = _nodes(r.json())["fetch"]["output"]
        assert "targets" in output
        assert isinstance(output["targets"], list)

    def test_fetch_output_row_count_matches_data(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        output = _nodes(r.json())["fetch"]["output"]
        assert output["row_count"] == len(output["data"])
        assert len(output["data"]) == len(output["targets"])

    def test_train_output_has_weights(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        output = _nodes(r.json())["train"]["output"]
        assert "weights" in output
        assert isinstance(output["weights"], list)

    def test_train_output_has_bias(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        output = _nodes(r.json())["train"]["output"]
        assert "bias" in output
        assert isinstance(output["bias"], float)

    def test_train_output_has_hyperparams(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        output = _nodes(r.json())["train"]["output"]
        assert output["epochs"] == 500
        assert output["learning_rate"] == 0.01

    def test_persist_output_has_label(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        output = _nodes(r.json())["persist"]["output"]
        assert output["label"] == "scoring_v1"

    def test_persist_output_record_count_increments(self, client):
        r1 = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        r2 = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        count1 = _nodes(r1.json())["persist"]["output"]["record_count"]
        count2 = _nodes(r2.json())["persist"]["output"]["record_count"]
        assert count2 == count1 + 1


# ---------------------------------------------------------------------------
# 2. Reference resolution — values travel correctly between nodes
# ---------------------------------------------------------------------------

class TestReferenceResolution:
    def test_request_source_reaches_fetch(self, client):
        """${request.source} is resolved before fetch executes — if it were
        broken the fetch node would fail, not complete."""
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert _nodes(r.json())["fetch"]["status"] == "completed"

    def test_fetch_data_reaches_train(self, client):
        """If ${fetch.output.data} resolution broke, train would fail."""
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert _nodes(r.json())["train"]["status"] == "completed"

    def test_train_weights_reach_persist(self, client):
        """If ${train.output.weights} resolution broke, persist would fail."""
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert _nodes(r.json())["persist"]["status"] == "completed"

    def test_weights_in_persist_match_train(self, client):
        """The weights stored by persist must equal the weights trained by train."""
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        nodes = _nodes(r.json())
        train_weights = nodes["train"]["output"]["weights"]
        # InMemorySaveResultsAdapter stores the record in state.save_results_adapter.records
        last_record = state.save_results_adapter.records[-1]
        assert last_record["weights"] == train_weights


# ---------------------------------------------------------------------------
# 3. LRO tracking — every node registers and completes an operation
# ---------------------------------------------------------------------------

class TestLROTracking:
    def test_each_node_has_an_lro(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        lro_node_ids = {op["node_id"] for op in r.json()["lros"]}
        assert lro_node_ids == {"fetch", "train", "persist"}

    def test_each_lro_has_duration(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        for op in r.json()["lros"]:
            assert op["duration_s"] is not None
            assert op["duration_s"] >= 0

    def test_lro_kinds_are_correct(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        kinds = {op["node_id"]: op["kind"] for op in r.json()["lros"]}
        assert kinds["fetch"] == "fetch"
        assert kinds["train"] == "train"
        assert kinds["persist"] == "persist"

    def test_lro_ref_contains_useful_info(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        lros = {op["node_id"]: op for op in r.json()["lros"]}
        assert lros["fetch"]["ref"] == "sample"        # the data source name
        assert "epochs=500" in lros["train"]["ref"]    # hyperparams from YAML
        assert lros["persist"]["ref"] == "scoring_v1"  # pipeline label

    def test_failed_run_lro_not_completed(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={})  # missing source
        lros = r.json()["lros"]
        # fetch failed so its LRO should be failed too
        fetch_lros = [op for op in lros if op["node_id"] == "fetch"]
        if fetch_lros:
            assert fetch_lros[0]["state"] == "failed"


# ---------------------------------------------------------------------------
# 4. Runtime type routing — same infra, different pipeline graph
# ---------------------------------------------------------------------------

class TestRuntimeRouting:
    def test_scoring_v1_has_three_nodes(self, client):
        r = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert len(r.json()["nodes"]) == 3

    def test_scoring_v2_has_four_nodes(self, client):
        r = client.post("/v2/pipelines/scoring_v2/run", json={"source": "sample"})
        assert len(r.json()["nodes"]) == 4

    def test_scoring_v2_node_ids(self, client):
        r = client.post("/v2/pipelines/scoring_v2/run", json={"source": "sample"})
        ids = {n["node_id"] for n in r.json()["nodes"]}
        assert ids == {"fetch", "train_a", "train_b", "persist"}

    def test_different_runtimes_both_succeed(self, client):
        r1 = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        r2 = client.post("/v2/pipelines/scoring_v2/run", json={"source": "sample"})
        assert r1.json()["succeeded"] is True
        assert r2.json()["succeeded"] is True

    def test_runs_are_independent(self, client):
        """Back-to-back runs must not affect each other's output."""
        r1 = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        r2 = client.post("/v2/pipelines/scoring_v1/run", json={"source": "sample"})
        assert r1.json()["succeeded"] is True
        assert r2.json()["succeeded"] is True
        # Both runs should produce the same number of nodes
        assert len(r1.json()["nodes"]) == len(r2.json()["nodes"])
