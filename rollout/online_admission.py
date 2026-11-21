"""Admit a baseline request through the same broker slots as DyCon branches."""
from contextlib import contextmanager
from uuid import uuid4


@contextmanager
def online_request_admission(cfg, runtime):
    runtime = runtime or {}
    if (runtime.get("online_source") != "baseline"
            or cfg.get("parallel", {}).get("request_admission") != "round_robin_by_trajectory"):
        yield
        return
    from rollout.decision.coordinator import ServingParallelClient
    client = runtime.get("parallel_client") or ServingParallelClient(cfg)
    decision = "online-baseline-" + uuid4().hex
    client.call("reserve", {"decision_id": decision, "branches": [0]})
    try:
        client.call("branch-acquire", {"decision_id": decision, "branch_id": 0})
        cancellation = runtime.get("cancellation")
        if cancellation is not None:
            cancellation.check()
        yield
    finally:
        client.call("release", {"decision_id": decision})
