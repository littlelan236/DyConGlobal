"""Harbor termination, verifier and capture facts at the scheduler boundary."""
from types import SimpleNamespace

import pytest

from application.execution.evidence import RunEvidence
from core.io import hash_json, read_json, sha256, write_json
from rollout.runner import _retry_cursor
from runtime.backends.contracts import AttemptBinding
from runtime.backends.harbor import adapter as harbor
from runtime.backends.registry import adapter_for
from tests.test_harbor_contract import harbor_task
from tests.test_limits_retry import configure


@pytest.mark.parametrize(
    "reason,exception,scoring,capture,retry,terminal,status,disposition",
    [
        ("native_reasoning_loop", "RuntimeError", "invalid", "partial", True,
         "reasoning_loop", "reasoning_loop", "retry_pending"),
        ("repeated_reasoning_action_loop", "RuntimeError", "invalid", "complete", True,
         "reasoning_loop", "reasoning_loop", "retry_pending"),
        ("all_parallel_branches_reasoning_loop", "RuntimeError", "invalid", "disabled", True,
         "reasoning_loop", "reasoning_loop", "retry_pending"),
        ("generation_limit", "RuntimeError", "invalid", "partial", True,
         "LimitsExceeded", "LimitsExceeded", "retry_pending"),
        ("generation_limit", "RuntimeError", "invalid", "complete", False,
         "LimitsExceeded", "VerifierError", "failed"),
        ("native_reasoning_loop", "RuntimeError", "invalid", "partial", False,
         "reasoning_loop", "CaptureError", "failed"),
        (None, "AgentTimeoutError", "zero", "complete", True,
         "AgentTimeout", "AgentTimeout", "accepted"),
        (None, "AgentTimeoutError", "invalid", "disabled", True,
         "AgentTimeout", "VerifierError", "failed"),
        (None, "AgentTimeoutError", "unknown", "disabled", True,
         "AgentTimeout", "Review", "failed"),
        (None, "AgentTimeoutError", "zero", "partial", True,
         "AgentTimeout", "CaptureError", "failed"),
    ],
)
def test_harbor_termination_survives_scoring_capture_and_recovery(
    cfg, monkeypatch, reason, exception, scoring, capture, retry, terminal, status, disposition,
):
    from hidden.backends.vllm import online
    from rollout import runner

    cfg = configure(cfg, retry=retry)
    cfg["execution_context"] = {"profile_sha256": "profile"}
    cfg["capture"].update(enabled=capture != "disabled", failure_policy="block_rollout")
    task = harbor_task()
    run_id, aid = "terminal", "rep-000-attempt-000"
    binding = {
        "run_id": run_id, "attempt_id": aid, "sample_key": "sample", "repetition_id": "rep-000",
        "task_key": hash_json({k: task[k] for k in ("benchmark", "revision", "instance_id")}),
        "sample_seed": 42, "effective_seed": 42, "seed_round": 0,
        "seed_policy": "repetition_round_offset_v1",
    }
    sample = SimpleNamespace(sample_key=binding["sample_key"], as_dict=lambda: binding)
    out = RunEvidence(cfg["project_root"], run_id).path.parent / "rollout" / "canary" / aid

    def worker(*args, **kwargs):
        write_json(out / "cleanup.json", {"complete": True})
        path = out / "trials/trial/verifier/test-stdout.txt"
        path.parent.mkdir(parents=True)
        path.write_text(
            "================ 1 failed in 0.10s ================\n" if scoring == "zero" else "no score\n"
        )
        result = {
            "trial_name": "trial",
            "verifier_result": {"rewards": {"reward": 0} if scoring == "zero" else None},
            "exception_info": {"exception_type": exception},
            "reward_valid": scoring != "invalid",
        }
        return {"trial_id": "trial", "result": result}, SimpleNamespace(limit_reason=reason, records=[])

    class Capture:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, *args, **kwargs):
            pass

        def prepare(self, *args):
            return {"responses": [], "capture_failures": []}

        def finalize(self, *args, **kwargs):
            return {"capture_status": capture}

    monkeypatch.setattr(harbor, "run_worker_bridge", worker)
    monkeypatch.setattr(harbor, "drain_capture", lambda *args, **kwargs: None)
    monkeypatch.setattr(online, "CaptureClient", Capture)
    monkeypatch.setattr(runner, "_recover_capture_status",
                        lambda *args: "complete" if capture == "complete" else None)
    context = SimpleNamespace(cfg=cfg, run_id=run_id)
    runtime = {"capture_control_url": "unused", "capture_token_inventory": {
        "newline_token_ids": [], "close_token_ids": [], "tokenizer_fingerprint": "fixture", "capture_select": None,
    }}
    adapter = adapter_for(
        {"benchmark": {"backend": "terminalbench_2_1"}, "agent": {"backend": "openhands"}}, {"task": task}
    )
    result = adapter.execute_attempt(context, AttemptBinding(task, sample, binding, out, runtime))

    assert "error" not in result, result.get("error")
    assert (result.get("agent_exit_status"), result["exit_status"], result["disposition"]) == (
        terminal, status, disposition
    ), result
    assert result["resolved"] is (False if scoring == "zero" else None)
    assert result["capture_status"] == capture
    assert bool(result["capture_errors"]) == (capture == "partial")
    assert _retry_cursor(cfg, [result]) == ((1, "resample") if disposition == "retry_pending" else (0, "initial"))

    evidence = RunEvidence(cfg["project_root"], run_id)
    row = evidence.read()["rollouts"][result["rollout_id"]]
    assert row["termination_reason"] == terminal
    assert row["resolved"] is result["resolved"]
    assert evidence.verify()["verified"]
    artifact = out / "evaluation-report.json"
    report = read_json(artifact)
    score_sha = sha256(artifact)
    assert report["status"] == {"zero": "completed", "unknown": "unknown", "invalid": "infra_error"}[scoring]

    (out / "attempt.json").unlink()
    recovered = adapter.recover_attempt(cfg, run_id, task, sample, binding, out, runtime)
    assert recovered["agent_exit_status"] == terminal
    assert recovered["exit_status"] == status
    assert recovered["disposition"] == ("blocked" if status == "CaptureError" else disposition)
    assert recovered["resolved"] is result["resolved"]
    assert recovered["capture_status"] == ("pending_recovery" if capture == "partial" else capture)
    assert recovered["evaluation_outcome"]["raw_reward"] == report["raw_reward"]
    assert sha256(artifact) == score_sha
