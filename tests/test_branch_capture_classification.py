import numpy as np
import pytest

from core.config import snapshot
from core.io import hash_json, read_json, sha256, write_json
from hidden.artifacts import read_branch_index
from hidden.backends.vllm.online import capture_request_metadata
from hidden.branch_coverage import classify_close
from hidden.reclassify_online import classify_session, reclassify_online
from runtime.serving.capture import OnlineCaptureController
from runtime.serving.spool import publish_files


@pytest.mark.parametrize("output,finish,reason,status", [
    ([20, 30], "stop", "no_close_generated", "complete"),
    ([20, 30], "length", "truncated_before_close", "complete"),
    ([], "stop", "no_close_generated", "complete"),
    ([20, 248069], "stop", "no_followup_forward", "partial"),
    ([20, 248069, 30], "stop", "missing_close_hidden", "partial"),
])
def test_native_tokens_distinguish_no_event_from_lost_event(output, finish, reason, status):
    request = {"prompt_token_ids": [248068], "output_token_ids": output,
               "close_token_ids": [248069], "anchors": ["think_close"], "finish_reason": finish}
    state = classify_close(request, [])
    assert (state["reason"], state["status"]) == (reason, status)


def test_missing_receipt_cannot_be_excused_as_no_close():
    assert classify_close(None, [])["status"] == "partial"


def test_loop_abort_is_explicit_without_hiding_real_missing_close():
    receipt = {"prompt_token_ids": [248068], "output_token_ids": [1] * 2048,
               "anchors": ["think_close"], "finish_reason": "abort", "reasoning_loop": {"abort_completed": True}}
    state = classify_close(receipt, [])
    assert state["reason"] == "reasoning_loop_aborted" and state["not_required"]
    receipt["output_token_ids"] += [248069, 3]
    assert classify_close(receipt, [])["status"] == "partial"


def test_live_finalize_no_close_rejected_branch_is_complete(tmp_path):
    c = OnlineCaptureController(tmp_path, {"export": {"root": "online", "queue_size": 8},
        "run_id": "run", "open_token_ids": [10], "close_token_ids": [11], "finalize_seconds": 2})
    base = capture_request_metadata("s", "t", "run", "pending", 0, ["think_close"], close_token_ids=[11])
    c.start(base)
    try:
        candidates = []
        for branch, output in enumerate([[20, 11, 30], [20, 30]]):
            md = {**base, "decision_id": "d", "branch_id": branch, "branch_count": 2, "api_attempt_id": str(branch)}
            c.bind(f"r{branch}", {**md, "prompt_token_ids": [10]})
            c.record_output({"request_id": f"r{branch}", "prompt_token_ids": [10], "output_token_ids": output})
            candidates.append({"branch_id": branch, "api_attempt_id": str(branch), "response": {}, "score": None})
        c.record("r0", {"layer_0": np.ones(2)}, anchor="think_close", position=2, absolute_position=2, token_id=11)
        c.record_decision({"session_id": "s", "decision_id": "d", "branch_count": 2, "step_id": 0,
                           "selected_branch_id": 0, "candidates": candidates})
        path = tmp_path / "trajectory.json"
        write_json(path, {"messages": [{"role": "assistant", "extra": {"response": {
            "dycon_native_request_id": "r0", "prompt_token_ids": [10], "output_token_ids": [20, 11, 30]}}}]})
        result = c.finalize("s", rollout_id="rid", trajectory_path=str(path),
                            trajectory_sha256=sha256(path), model_fingerprint="model")
        assert result["capture_status"] == "complete"
        coverage = read_json(c.export_root / "sessions/s/parallel/branch-coverage.json")
        assert coverage["candidate_count"] == 2
        assert coverage["expected"] == coverage["observed"] == 1
        assert coverage["not_required"][0]["branch_id"] == 1
        assert coverage["missing"] == []
    finally:
        c.close()


def legacy_export(project, output):
    export = project / "spool/run"
    sid = "cap-run-task-attempt-000"
    base = export / "sessions" / sid
    trajectory = project / "trajectory.json"
    write_json(trajectory, {"messages": []})
    write_json(base / "finalization.json", {"rollout_id": "rid", "trajectory_path": str(trajectory),
                                           "trajectory_sha256": sha256(trajectory)})
    write_json(base / "coverage.json", {"expected": 1, "observed": 1, "missing": []})
    write_json(base / "receipt.json", {"metrics": {"fallbacks": 0}, "export": {"failed": []}})
    write_json(export / "requests/rejected.json", {"session_id": sid, "request_id": "rejected",
        "prompt_token_ids": [248068], "output_token_ids": output, "close_token_ids": [248069],
        "anchors": ["think_close"], "finish_reason": "length"})
    write_json(base / "parallel/decisions/d.json", {"candidates": [{"branch_id": 1,
        "ledger_ref": "requests/rejected.json", "ledger_sha256": sha256(export / "requests/rejected.json")}]})
    rows = [{"decision_id": "d", "branch_id": 0, "native_request_id": "winner", "close_events": [{"absolute_position": 2}]},
            {"decision_id": "d", "branch_id": 1, "native_request_id": "rejected", "close_events": []}]
    write_json(base / "parallel/branch-index.json", rows)
    write_json(base / "parallel/branch-coverage.json", {"expected": 2, "observed": 1,
               "missing": [{"decision_id": "d", "branch_id": 1, "reason": "missing_close_hidden"}]})
    files = {str(p.relative_to(export)): sha256(p) for p in base.rglob("*.json")}
    write_json(base / "parallel/branch-receipt.json", {"files": files})
    return export, sid


def test_revision_preserves_original_receipts_and_is_idempotent(tmp_path):
    export, sid = legacy_export(tmp_path, [20, 30])
    files = {str(p): sha256(p) for p in export.rglob("*.json")}
    revised = classify_session(export, sid)
    assert revised["coverage"]["missing"] == []
    assert revised == classify_session(export, sid)
    assert all(sha256(path) == digest for path, digest in files.items())
    assert read_branch_index(export, sid)["coverage"]["expected"] == 1
    revision_path = export / "sessions" / sid / "parallel/branch-classification-v2.json"
    value = read_json(revision_path)
    value["coverage"]["expected"] = 999
    write_json(revision_path, value)
    with pytest.raises(ValueError, match="checksum"):
        read_branch_index(export, sid)


def test_real_lost_hidden_stays_partial_on_reclassification(tmp_path):
    export, sid = legacy_export(tmp_path, [20, 248069, 30])
    revised = classify_session(export, sid)
    assert revised["coverage"]["missing"][0]["reason"] == "missing_close_hidden"


@pytest.mark.parametrize("other_failure", [False, True])
def test_cli_repair_updates_manifests_results_and_reconciles_stale_progress(tmp_path, monkeypatch, other_failure):
    export, sid = legacy_export(tmp_path, [20, 30])
    monkeypatch.setattr("application.execution.evidence.RunEvidence.read", lambda self: {"rollouts": {"rid": {
        "trajectory_path": "trajectory.json", "trajectory_sha256": sha256(tmp_path / "trajectory.json")}}})
    cfg = {"project_root": str(tmp_path), "paths": {"nas_root": str(tmp_path / "nas")},
           "images": {"reserve_bytes": 0},
           "capture": {"export": {"root": "online", "spool_root": "spool", "upload_max_elapsed_seconds": 10}}}
    cfg["config_hash"] = hash_json(cfg)
    snapshot(cfg, "run", {"task_keys": ["key"], "tasks": {"key": {"instance_id": "task"}}})
    write_json(tmp_path / "runs/run/process.json", {"pid": 999999999, "supervisor_pid": 999999999})
    nas = tmp_path / "nas/online/run"
    publish_files(export, nas, {str(p.relative_to(export)): sha256(p) for p in export.rglob("*.json")}, 10)
    for owner in ["hidden", "enrichment"]:
        write_json(tmp_path / owner / "manifest.json", {"online_captures": {sid: {
            "run_id": "run", "status": "partial", "published": False,
            "eligible_for_training": False, "dataset_role": "rollout"}}})
    result = {"rollout_id": "rid", "attempt_id": "attempt-000", "exit_status": "LimitsExceeded",
              "capture_status": "partial", "capture_errors": ["RuntimeError('Capture is partial')"]}
    if other_failure:
        result["capture_errors"].append("unrelated capture failure")
    work = tmp_path / "runs/run/tasks/task"
    for path in [work / "result.json", work / "attempts/attempt-000/attempt.json"]:
        write_json(path, result)
    write_json(tmp_path / "runs/run/progress.json", {"key": result})
    if other_failure:
        with pytest.raises(ValueError, match="Additional capture failures"):
            reclassify_online(cfg, "run", [sid])
        assert read_json(tmp_path / "hidden/manifest.json")["online_captures"][sid]["status"] == "partial"
        assert not (nas / "sessions" / sid / "capture-status-v2.json").exists()
        return
    reclassify_online(cfg, "run", [sid])
    for owner in ["hidden", "enrichment"]:
        entry = read_json(tmp_path / owner / "manifest.json")["online_captures"][sid]
        assert entry["status"] == "complete" and entry["published"] and entry["eligible_for_training"]
    assert read_json(work / "result.json")["exit_status"] == "LimitsExceeded"
    assert read_json(work / "result.json")["capture_errors"] == []
    write_json(tmp_path / "runs/run/progress.json", {"key": result})  # Old running code overwrites progress.
    reclassify_online(cfg, "run", [sid], wait_for_run=True)
    assert read_json(tmp_path / "runs/run/progress.json")["key"]["capture_status"] == "complete"
    # Recovery consumes the same revision instead of reviving legacy partial.
    from hidden import recover_online as recovery

    class PublishedSpool:
        def __init__(self, *args):
            self.root = export

        def recover(self):
            return []

        def publish_session(self, *args):
            return True

        def release_session_metadata(self, *args):
            pass

        def close(self, *args):
            return True

    monkeypatch.setattr(recovery, "CaptureSpool", PublishedSpool)
    assert recovery.recover_online(cfg, "run")["sessions"][sid] == "complete"
    assert read_json(tmp_path / "hidden/manifest.json")["online_captures"][sid]["eligible_for_training"]
