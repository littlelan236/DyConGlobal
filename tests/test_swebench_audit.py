import json
from pathlib import Path

import pytest

from benchmark_evaluation.swebench_audit import audit_run, collect
from benchmark_evaluation.swebench_receipts import get_test_command, instrument
from core.io import hash_json, sha256, write_json


def fixture(tmp_path, monkeypatch):
    from swebench.harness import grading

    task = {"instance_id": "org__repo-1", "repo": "org/repo", "version": "1",
            "FAIL_TO_PASS": ["tests/test_x.py::test_x"], "PASS_TO_PASS": [],
            "base_commit": "abc", "test_patch": "frozen", "patch": "reference patch\n"}
    rid, eid = "r-fixture", "task-fixture"
    output = tmp_path / "runs/run-fixture/harness"
    logs = output / "harness" / eid / rid / task["instance_id"]
    env = output / "environment" / eid / task["instance_id"]
    logs.mkdir(parents=True)
    env.mkdir(parents=True)
    raw_script = "#!/bin/bash\n: '>>>>> Start Test Output'\npytest -q\n: '>>>>> End Test Output'\ntrue\n"
    (logs / "eval.sh").write_text(instrument(raw_script)[0])
    (logs / "test_output.txt").write_text(
        "+ : '>>>>> Start Test Output'\nPASSED tests/test_x.py::test_x\n+ : '>>>>> End Test Output'\n")
    report = {"resolved": True, "patch_successfully_applied": True,
              "tests_status": {"FAIL_TO_PASS": {"success": task["FAIL_TO_PASS"], "failure": []},
                                "PASS_TO_PASS": {"success": [], "failure": []}}}
    write_json(logs / "report.json", {task["instance_id"]: report})
    outcome = {"status": "completed", "resolved": True, "patch_sha256": hash_json(task["patch"]),
               "attempts": [{"evaluation_task_id": eid}]}
    selection = {"rollout_ids": [rid], "rollouts": {rid: {"task_key": "key"}},
                 "tasks": {"key": {"instance_id": task["instance_id"], "instance": task}}}
    write_json(output / "selection.json", selection)
    write_json(output / "selection-receipt.json", {"selection_sha256": hash_json(selection)})
    write_json(output / "summary.json", {"gold": True, "outcomes": {rid: outcome}})
    write_json(output / "resolved-config.json", {"selection": selection, "benchmark": {"backend": "swebench"}})
    monkeypatch.setattr(grading, "get_logs_eval", lambda *args: ({task["FAIL_TO_PASS"][0]: "PASSED"}, True))
    return output, logs, env, task, outcome, rid, eid, raw_script


def test_collect_uses_only_bound_attempt_and_rejects_bad_report(tmp_path, monkeypatch):
    output, logs, env, task, outcome, rid, eid, _ = fixture(tmp_path, monkeypatch)
    document = collect(output, eid, rid, task, outcome, gold=True)
    assert document["report"]["resolved"] is True
    assert not document["collection_errors"]
    assert document["gold_provenance"]["verified"] is True
    (logs / "report.json").write_text("bad json")
    document = collect(output, eid, rid, task, outcome, gold=True)
    assert document["collection_errors"] == ["Malformed official report"]


def test_offline_cli_service_preserves_every_historical_file(tmp_path, monkeypatch):
    output, *_ = fixture(tmp_path, monkeypatch)
    originals = {p: sha256(p) for p in output.rglob("*") if p.is_file()}
    target = tmp_path / "audits/first"
    result = audit_run(tmp_path, "run-fixture", target)
    assert result["complete"] is True
    assert result["classification_counts"] == {"ValidOutcome": 1}
    assert {p: sha256(p) for p in originals} == originals
    assert {p for p in output.rglob("*") if p.is_file()} == set(originals)
    with pytest.raises(FileExistsError):
        audit_run(tmp_path, "run-fixture", target)


def test_audit_refuses_output_inside_runs(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside historical runs"):
        audit_run(tmp_path, "run-fixture", tmp_path / "runs/other/audit")


def test_changed_frozen_selection_is_rejected(tmp_path, monkeypatch):
    output, *_ = fixture(tmp_path, monkeypatch)
    selection = json.loads((output / "selection.json").read_text())
    selection["added"] = "changed"
    write_json(output / "selection.json", selection)
    with pytest.raises(ValueError):
        audit_run(tmp_path, "run-fixture", tmp_path / "audit")


def test_missing_log_is_incomplete_review_not_fabricated_pass(tmp_path, monkeypatch):
    output, logs, *_ = fixture(tmp_path, monkeypatch)
    (logs / "test_output.txt").unlink()
    result = audit_run(tmp_path, "run-fixture", tmp_path / "audit")
    assert result["complete"] is False and result["classification_counts"] == {"Review": 1}


def test_receipt_is_bound_to_attempt_and_original_command(tmp_path, monkeypatch):
    output, logs, env, task, outcome, rid, eid, script = fixture(tmp_path, monkeypatch)
    details = get_test_command(script)
    receipt = {"instance_id": task["instance_id"], "evaluation_task_id": eid,
               "returncode": 0, "test_command_sha256": details["command_sha256"]}
    write_json(env / "eval-execution.json", receipt)
    assert not collect(output, eid, rid, task, outcome)["collection_errors"]
    receipt["test_command_sha256"] = "wrong"
    write_json(env / "eval-execution.json", receipt)
    assert "Execution/test script command mismatch" in collect(output, eid, rid, task, outcome)["collection_errors"]
    receipt["evaluation_task_id"] = "other"
    write_json(env / "eval-execution.json", receipt)
    assert "Execution receipt identity mismatch" in collect(output, eid, rid, task, outcome)["collection_errors"]


def test_symlink_evidence_cannot_escape_harness_root(tmp_path, monkeypatch):
    output, logs, env, task, outcome, rid, eid, _ = fixture(tmp_path, monkeypatch)
    external = tmp_path / "external.txt"
    external.write_text("not attempt evidence")
    (logs / "test_output.txt").unlink()
    (logs / "test_output.txt").symlink_to(external)
    assert any("escapes harness root" in e for e in collect(output, eid, rid, task, outcome)["collection_errors"])


@pytest.mark.parametrize("returncode,timed_out", [(0, False), (124, True), (137, False), (139, False)])
def test_live_adapter_writes_execution_without_rebuilding_metadata(tmp_path, monkeypatch, returncode, timed_out):
    from types import SimpleNamespace

    from benchmark_evaluation.harness import install_harness_adapter
    from swebench.harness import run_evaluation as official
    from swebench.harness.test_spec.test_spec import TestSpec

    raw = "#!/bin/bash\n: '>>>>> Start Test Output'\npytest -q\n: '>>>>> End Test Output'\ntrue\n"
    original = property(lambda self: raw)
    monkeypatch.setattr(TestSpec, "eval_script", original)
    probes = []
    def forbidden_metadata(*args, **kwargs):
        raise AssertionError("Execution receipt must not rebuild the test spec or fetch metadata")
    monkeypatch.setattr(official, "make_test_spec", forbidden_metadata)
    factory = SimpleNamespace(
        policy_for=lambda row: SimpleNamespace(wrap_eval=lambda script: instrument(script)[0]),
        execute=lambda *args: {"output": "captured output", "returncode": returncode},
        probe=lambda *args: probes.append(args), close=lambda *args: None)
    iid = "org__repo-1"
    container = SimpleNamespace(instance={"instance_id": iid},
                                context=SimpleNamespace(attempt_id="task-bound", output_dir=tmp_path / "environment"))
    cfg = {"environment": {"requirements_cache": str(tmp_path / "requirements")},
           "project_root": str(tmp_path), "timeouts": {"metadata_request_seconds": 1}}
    before_exec = official.exec_run_with_timeout
    adapter = install_harness_adapter(cfg, factory, [{"instance_id": iid}], tmp_path)
    try:
        script = TestSpec.eval_script.fget(SimpleNamespace(instance_id=iid))
        assert "__DYCON_TEST_RESULT__" in script
        output, observed_timeout, _ = adapter.exec_run_with_timeout(container, "/bin/bash /eval.sh", 30)
        assert output == "captured output" and observed_timeout == timed_out
        receipt = json.loads((container.context.output_dir / "eval-execution.json").read_text())
        assert receipt["returncode"] == returncode and receipt["evaluation_task_id"] == "task-bound"
        assert receipt["test_command_sha256"] == get_test_command(raw)["command_sha256"]
        assert bool(probes) is not timed_out
    finally:
        adapter.dycon_restore()
    assert TestSpec.eval_script is original and official.exec_run_with_timeout is before_exec
