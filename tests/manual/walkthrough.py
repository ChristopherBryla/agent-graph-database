"""Manual functional walkthrough of the v2 orchestrator.

HOW TO USE THIS FILE
====================
This is NOT a pytest file. It is an interactive script you run yourself,
one scenario at a time, while reading the printed output. Each scenario
makes a real HTTP call against the running server and explains what the
ComponentManager is doing under the hood.

Step 1 — start the server in one terminal:

    .\\.venv\\Scripts\\python.exe -m uvicorn genesys.app:app --reload --app-dir src

Step 2 — in a second terminal, run a single scenario:

    .\\.venv\\Scripts\\python.exe tests/manual/walkthrough.py 1

Or run them all in order:

    .\\.venv\\Scripts\\python.exe tests/manual/walkthrough.py

Each scenario prints:
  • What it is testing and why
  • The URL + headers + body it is sending
  • The full server response (status, nodes, lros)
  • What to look for in the output
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx

BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(n: int, title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\nSCENARIO {n}: {title}\n{line}")


def show(method: str, url: str, headers: dict[str, str] | None, body: Any) -> None:
    print(f"\n→ {method} {url}")
    if headers:
        for k, v in headers.items():
            print(f"  {k}: {v}")
    if body is not None:
        print(f"  body: {json.dumps(body, indent=2)}")


def call(method: str, path: str, *, headers: dict[str, str] | None = None,
         json_body: Any = None) -> httpx.Response:
    url = BASE_URL + path
    show(method, url, headers, json_body)
    with httpx.Client(timeout=60) as c:
        resp = c.request(method, url, headers=headers, json=json_body)
    print(f"\n← {resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)
    return resp


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def s1_list_runtimes() -> None:
    banner(1, "Discover available runtime_types")
    print("""
The orchestrator loads one YAML per runtime_type from pipeline_configs/.
GET /v2/runtimes lists every config it found at startup.
You should see ['scoring_v1', 'scoring_v2'].
""")
    call("GET", "/v2/runtimes")


def s2_simple_run() -> None:
    banner(2, "Happy path — scoring_v1 with the sample data source")
    print("""
This sends a JSON body with a 'source' field. Inside scoring_v1.yaml the
fetch node references it via ${request.source}. Watch the response:
  • nodes[]   → fetch / train / persist all 'completed', in that order
  • lros[]    → 3 entries, all 'completed'
  • each node's 'output' is the dict its component returned, AND it's what
    downstream nodes read via ${node_id.output.field}
""")
    call("POST", "/v2/pipelines/scoring_v1/run", json_body={"source": "sample"})


def s3_parallel_run() -> None:
    banner(3, "Parallel level — scoring_v2 trains two models at once")
    print("""
scoring_v2 has train_a and train_b both depending only on fetch and NOT on
each other, so they sit in the same level and run in parallel threads.
Watch the server terminal: you'll see '[orchestrator] level 1: ...' contain
both nodes. The response 'lros' will show their start times overlap.
""")
    call("POST", "/v2/pipelines/scoring_v2/run", json_body={"source": "sample"})


def s4_debug_filter() -> None:
    banner(4, "X-Debug-Component-List — only run specific components")
    print("""
Send X-Debug-Component-List: data_provider. Only the fetch node will run;
train + persist will be 'skipped' with reason 'not in X-Debug-Component-List'.
This is your selective-execution feature.
""")
    call(
        "POST",
        "/v2/pipelines/scoring_v1/run",
        headers={"X-Debug-Component-List": "data_provider"},
        json_body={"source": "sample"},
    )


def s5_unknown_runtime() -> None:
    banner(5, "Unknown runtime_type returns 404")
    print("""
There is no pipeline_configs/scoring_v9.yaml so the orchestrator returns 404.
This proves runtime routing is config-driven — adding a YAML adds a runtime.
""")
    call("POST", "/v2/pipelines/scoring_v9/run", json_body={"source": "sample"})


def s6_fail_fast() -> None:
    banner(6, "Failure handling — bad reference + X-Fail-Fast")
    print("""
We send a body WITHOUT a 'source' field. The fetch node's input
${request.source} cannot resolve, so the node fails. With X-Fail-Fast: true
all subsequent nodes are 'skipped' with reason 'fail-fast: ...'.
With X-Trace-Failures: true the node's 'traceback' field is populated.

Watch:
  • succeeded: false
  • fetch.status: failed
  • train.status: skipped (fail-fast)
  • persist.status: skipped (fail-fast)
""")
    call(
        "POST",
        "/v2/pipelines/scoring_v1/run",
        headers={
            "X-Fail-Fast": "true",
            "X-Trace-Failures": "true",
        },
        json_body={},  # missing 'source' on purpose
    )


def s7_lro_wait() -> None:
    banner(7, "wait_for_lros — persist blocks until its LRO is complete")
    print("""
scoring_v1's persist node has 'wait_for_lros: true'. That means the
orchestrator blocks on the persist node's LROs after the node returns,
before marking the run finished. Look at the response 'lros' — the persist
operation's finished_at is populated and 'state' is 'completed' BEFORE the
overall run returns to you. If the LRO had failed (e.g. timeout) the persist
node's status would flip from 'completed' to 'failed'.
""")
    call("POST", "/v2/pipelines/scoring_v1/run", json_body={"source": "sample"})


SCENARIOS = [
    s1_list_runtimes,
    s2_simple_run,
    s3_parallel_run,
    s4_debug_filter,
    s5_unknown_runtime,
    s6_fail_fast,
    s7_lro_wait,
]


def main() -> None:
    if len(sys.argv) > 1:
        idx = int(sys.argv[1]) - 1
        if not (0 <= idx < len(SCENARIOS)):
            print(f"valid scenario numbers: 1..{len(SCENARIOS)}")
            sys.exit(1)
        SCENARIOS[idx]()
        return

    for fn in SCENARIOS:
        fn()


if __name__ == "__main__":
    main()
