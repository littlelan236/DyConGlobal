"""Negative evidence contracts and real pinned official evaluator fixtures."""
import copy
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from application.execution.webarena_harness import _verify_selection
from benchmark_evaluation.webarena import REVISION, RUNTIME_REF, classify_result, evaluate
from core.io import hash_json, read_json, sha256, write_json


def _subject(directory, *, answer=None, har=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "agent_response.json", answer or {
        "task_type": "RETRIEVE", "status": "SUCCESS", "retrieved_data": [], "error_details": None})
    write_json(directory / "network.har", har or {"log": {"entries": []}})
    (directory / "actions.jsonl").write_text("", encoding="utf-8")
    subject = {"schema_version": 1, "kind": "web_trace", "complete": True, "har_flushed": True,
               "answer_path": "agent_response.json", "har_path": "network.har", "actions_path": "actions.jsonl",
               "binding": {}, "runtime_identity": {"environments": {
                   "shopping_admin": {"urls": ["http://shopping-admin/admin"]}}},
               "outputs": [{"path": path.name, "sha256": sha256(path), "size": path.stat().st_size}
                           for path in sorted(directory.iterdir()) if path.is_file()]}
    write_json(directory / "subject.json", subject)
    return directory / "subject.json"


def _row(root, subject):
    trajectory = subject.parent / "trajectory.json"
    write_json(trajectory, {"messages": []})
    return {"evaluation_subject": {"kind": "web_trace"}, "submission_status": "submitted", "task_key": "task",
            "trajectory_path": str(trajectory.relative_to(root)), "trajectory_sha256": sha256(trajectory),
            "subject_path": str(subject.relative_to(root)), "subject_sha256": sha256(subject)}


def test_official_failure_and_errors_are_distinct():
    assert classify_result({"status": "success", "score": 1.0})["resolved"] is True
    assert classify_result({"status": "failure", "score": 0.0}) == {
        "status": "completed", "resolved": False, "score": 0.0}
    assert classify_result({"status": "error", "score": 0.0})["resolved"] is None
    assert classify_result({"status": "error", "score": 0.0})["status"] == "infra_error"
    with pytest.raises(ValueError, match="Unknown official"):
        classify_result({"status": "mystery", "score": 0.0})


def test_harness_refuses_changed_unsubmitted_and_unflushed_subjects(tmp_path):
    subject = _subject(tmp_path / "trace")
    row = _row(tmp_path, subject)
    selection = {"rollout_ids": ["r"], "rollouts": {"r": row}, "tasks": {"task": {
        "benchmark": "webarena_verified"}}}
    _verify_selection(tmp_path, selection, {"selection_sha256": hash_json(selection)})
    changed = copy.deepcopy(selection)
    changed["rollout_ids"].append("forged")
    with pytest.raises(ValueError, match="selection changed"):
        _verify_selection(tmp_path, changed, {"selection_sha256": hash_json(selection)})
    changed = copy.deepcopy(selection)
    changed["rollouts"]["r"]["submission_status"] = "not_submitted"
    with pytest.raises(ValueError, match="actual agent submission"):
        _verify_selection(tmp_path, changed, {"selection_sha256": hash_json(changed)})
    manifest = read_json(subject)
    manifest.update(complete=False, har_flushed=False)
    write_json(subject, manifest)
    row["subject_sha256"] = sha256(subject)
    with pytest.raises(ValueError, match="incomplete"):
        _verify_selection(tmp_path, selection, {"selection_sha256": hash_json(selection)})


def test_harness_detects_changed_har_bytes(tmp_path):
    subject = _subject(tmp_path / "trace")
    row = _row(tmp_path, subject)
    selection = {"rollout_ids": ["r"], "rollouts": {"r": row}, "tasks": {"task": {
        "benchmark": "webarena_verified"}}}
    write_json(subject.parent / "network.har", {"log": {"entries": []}, "forged": True})
    with pytest.raises(ValueError, match="Changed subject output"):
        _verify_selection(tmp_path, selection, {"selection_sha256": hash_json(selection)})


def test_cross_run_re_evaluation_retains_original_browser_binding(cfg, monkeypatch):
    from application.execution.evidence import RunEvidence
    from application.execution.webarena_harness import evaluate_run
    from benchmark_evaluation import webarena as official_adapter
    from catalog.webarena import DATASET_SHA256
    from rollout.artifacts import import_rollout

    root = Path(cfg["project_root"])
    task = {"benchmark": "webarena_verified", "revision": REVISION, "instance_id": "wav-108",
            "partition": "test", "exclusion_reason": None, "instance": {
                "source_task_id": 108, "task_revision": 2, "intent_template_id": 270, "hard": False,
                "dataset_revision": REVISION, "dataset_sha256": DATASET_SHA256,
                "source_task_sha256": "a" * 64, "intent": "Read report",
                "sites": ["shopping_admin"], "start_urls": ["__SHOPPING_ADMIN__"]}}
    subject = _subject(root / "runs/origin/rollout/source")
    manifest = read_json(subject)
    manifest["binding"] = {"run_id": "origin", "attempt_id": "a0", "task_key": hash_json({
        key: task[key] for key in ("benchmark", "revision", "instance_id")})}
    write_json(subject, manifest)
    raw = subject.parent / "raw.json"
    write_json(raw, {"instance_id": "wav-108", "info": {"exit_status": "Submitted"}, "messages": []})
    rid = import_rollout(root, {"source": str(raw), "subject": str(subject), "subject_sha256": sha256(subject),
                               "task": task, "run_id": "origin", "attempt_id": "a0",
                               "model_fingerprint": "fixture", "agent": "minisweagent"})
    monkeypatch.setattr(official_adapter, "runtime_reference", lambda cfg: {"revision": REVISION})
    monkeypatch.setattr(official_adapter, "evaluate", lambda *args: {"status": "completed", "resolved": True})
    source_run = "origin"
    for target_run in ("grade1", "grade2"):
        directory = root / "runs" / target_run
        directory.mkdir(parents=True)
        selected = RunEvidence(root, source_run).select([rid])
        selected.update(schema_version=1, benchmark="webarena_verified", run_id=target_run)
        result = evaluate_run({**cfg, "selection": selected}, directory, "e0")
        assert result["outcomes"][rid]["resolved"] is True
        evidence = RunEvidence(root, target_run)
        row = evidence.read()["rollouts"][rid]
        assert row["provenance"]["source_run_id"] == "origin"
        assert row["attempt_id"] == "a0"
        assert evidence.verify()["verified"] is True
        source_run = target_run


@pytest.fixture
def pinned_official():
    """Uses the separately published evaluator, never installs during pytest."""
    root = Path(__file__).resolve().parents[1]
    runtime_path = root / RUNTIME_REF
    if not runtime_path.is_file():
        pytest.skip("Run dycon webarena evaluator-prepare for the pinned official fixture tests")
    runtime = read_json(runtime_path)
    source = Path(os.environ.get("DYCON_WEBARENA_EVALUATOR_SOURCE", runtime["source"]))
    source_dataset = source / "assets/dataset/webarena-verified.json"
    if not source_dataset.is_file() or not (source / "examples/agent_logs/demo/108/network.har").is_file():
        pytest.skip("Pinned official source fixture checkout is unavailable")
    directory = root / "temp/webarena-offline-evaluator-tests"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        # The production evaluator accepts only run-owned paths. Keep the
        # separately published source checkout external, but stage the frozen
        # dataset bytes inside this test run before constructing its config.
        dataset = Path(scratch) / "webarena-verified.json"
        shutil.copyfile(source_dataset, dataset)
        cfg = {"project_root": str(root), "webarena": {"evaluator_python": runtime["python"]},
               "timeouts": {"evaluation_seconds": 120}, "execution_context": {
                   "evaluator": {"path": dataset.relative_to(root).as_posix(), "sha256": sha256(dataset),
                                 "revision": REVISION},
                   "evaluator_runtime": {"path": RUNTIME_REF, "sha256": sha256(runtime_path)}}}
        yield cfg, source, read_json(dataset), Path(scratch)


def _official_subject(source, definitions, scratch, task_id, *, wrong_task_type=False, corrupt_har=False):
    fixture = source / f"examples/agent_logs/demo/{task_id}"
    definition = next(task for task in definitions if task["task_id"] == task_id)
    directory = scratch / str(task_id)
    directory.mkdir()
    for name in ("agent_response.json", "network.har"):
        shutil.copyfile(fixture / name, directory / name)
    if wrong_task_type:
        answer = read_json(directory / "agent_response.json")
        answer["task_type"] = "MUTATE"
        write_json(directory / "agent_response.json", answer)
    har = read_json(directory / "network.har")
    first = urlsplit(har["log"]["entries"][0]["request"]["url"])
    environments = {"shopping_admin": {"urls": [f"{first.scheme}://{first.netloc}/admin"]}}
    if corrupt_har:
        (directory / "network.har").write_text("malformed HAR", encoding="utf-8")
    (directory / "actions.jsonl").write_text("", encoding="utf-8")
    manifest = {"schema_version": 1, "kind": "web_trace", "complete": True, "har_flushed": True,
                "answer_path": "agent_response.json", "har_path": "network.har", "actions_path": "actions.jsonl",
                "runtime_identity": {"environments": environments},
                "outputs": [{"path": path.name, "sha256": sha256(path), "size": path.stat().st_size}
                            for path in sorted(directory.iterdir())]}
    write_json(directory / "subject.json", manifest)
    task = {"benchmark": "webarena_verified", "instance": {
        "source_task_id": task_id, "task_revision": definition["revision"], "sites": definition["sites"],
        "source_task_sha256": hash_json(definition)}}
    return task, directory / "subject.json"


@pytest.mark.parametrize(("task_id", "expected"), [(108, True), (107, False)])
def test_real_official_fixture_success_and_failure(pinned_official, task_id, expected):
    cfg, source, definitions, scratch = pinned_official
    task, subject = _official_subject(source, definitions, scratch, task_id)
    result = evaluate(cfg, task, subject, scratch / "result")
    assert result["status"] == "completed", result
    assert result["resolved"] is expected, result
    assert result["evaluator_checksum"]
    detailed = Path(cfg["project_root"]) / result["official_result_path"]
    assert detailed.stat().st_mode & 0o077 == 0


def test_real_official_wrong_task_type_is_negative(pinned_official):
    cfg, source, definitions, scratch = pinned_official
    task, subject = _official_subject(source, definitions, scratch, 108, wrong_task_type=True)
    result = evaluate(cfg, task, subject, scratch / "result")
    assert (result["status"], result["resolved"]) == ("completed", False), result


def test_real_official_har_parser_error_stays_unknown(pinned_official):
    cfg, source, definitions, scratch = pinned_official
    task, subject = _official_subject(source, definitions, scratch, 108, corrupt_har=True)
    result = evaluate(cfg, task, subject, scratch / "result")
    assert result["status"] == "infra_error", result
    assert result["resolved"] is None



@pytest.mark.parametrize("case", ["mixed_default", "explicit_unsubmitted", "only_unsubmitted"])
def test_prepare_mixed_run_only_defaults_to_submitted(tmp_path, monkeypatch, case):
    from application.execution import webarena_harness as harness
    subject = _subject(tmp_path / "trace")
    good = _row(tmp_path, subject)
    bad = {**copy.deepcopy(good), "submission_status": "not_submitted"}
    rows = {"bad": bad} if case == "only_unsubmitted" else {"ok": good, "bad": bad}
    run = tmp_path / "runs/grade"
    monkeypatch.setattr(harness, "load_plan", lambda *a: {
        "run": run, "tasks": {"tasks": [{"task_key": "task"}]}})

    def select(ids):
        selected = {rid: rows[rid] for rid in ids} if ids else rows
        return {"rollouts": selected, "tasks": {"task": {"benchmark": "webarena_verified"}},
                "evidence_revision": 1, "evidence_sha256": "a" * 64}

    monkeypatch.setattr(harness, "RunEvidence", lambda *a: SimpleNamespace(select=select))
    if case == "explicit_unsubmitted":
        with pytest.raises(ValueError, match="actual agent submission"):
            harness.prepare(tmp_path, "grade", ("bad",))
    elif case == "only_unsubmitted":
        with pytest.raises(ValueError, match="no eligible submitted"):
            harness.prepare(tmp_path, "grade")
    else:
        assert harness.prepare(tmp_path, "grade")["rollout_count"] == 1
        assert read_json(run / "harness/selection.json")["rollout_ids"] == ["ok"]


def test_evaluator_result_uses_owned_external_run_alias(tmp_path, monkeypatch):
    from benchmark_evaluation import webarena as adapter
    from core.io import owned_path

    root, storage = tmp_path / "project", tmp_path / "external-storage"
    run = storage / "runs/external"
    run.mkdir(parents=True)
    (root / "runs").mkdir(parents=True)
    (root / "runs/external").symlink_to(run, target_is_directory=True)
    definition = {"task_id": 108, "revision": 2}
    write_json(root / "dataset.json", [definition])
    write_json(root / "runtime.json", {"schema_version": 1, "revision": REVISION,
        "python": str(root / "fake-python")})
    cfg = {"project_root": str(root), "run_storage_root": str(storage),
           "webarena": {"evaluator_python": "fake-python"}, "execution_context": {
               "evaluator": {"path": "dataset.json", "sha256": sha256(root / "dataset.json"),
                             "revision": REVISION},
               "evaluator_runtime": {"path": "runtime.json", "sha256": sha256(root / "runtime.json")}}}
    selection = {"tasks": []}
    write_json(run / "resolved-config.json", {**cfg, "config_hash": hash_json(cfg),
        "selection": selection, "selection_sha256": hash_json(selection)})
    subject = _subject(root / "trace")
    task = {"benchmark": "webarena_verified", "instance": {"source_task_id": 108,
        "task_revision": 2, "source_task_sha256": hash_json(definition), "sites": ["shopping_admin"]}}

    def worker(command, *args):
        write_json(command[-1], {"status": "failure", "score": 0.0,
                                "webarena_verified_evaluator_checksum": "test-fixture"})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adapter, "_invoke_worker", worker)
    result = adapter.evaluate(cfg, task, subject, run / "harness/evaluations/e0/subjects/r0")
    assert result["status"] == "completed" and result["resolved"] is False, result
    assert result["official_result_path"] == "runs/external/harness/evaluations/e0/subjects/r0/official-result.json"
    path = owned_path(root, result["official_result_path"])
    assert sha256(path) == result["official_result_sha256"]
