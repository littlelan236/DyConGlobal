import subprocess
import sys
import threading

import pytest

from core.io import read_json, write_json
from runtime.scripts import launch_selection as launcher


def prepare(tmp_path, monkeypatch):
    selection = {"task_keys": ["key"], "tasks": {"key": {"instance_id": "test-task"}}}
    monkeypatch.setattr(launcher, "load_snapshot", lambda *args: {"selection": selection})
    monkeypatch.setattr(launcher, "resolve", lambda *args: {"config_hash": "initial"})
    return selection


def predecessor(tmp_path, code=0, progress=None):
    write_json(tmp_path / "state/supervisors/old/exit.json", {"exit_code": code})
    write_json(tmp_path / "runs/old/status.json", {"stage": "complete"})
    write_json(tmp_path / "runs/old/progress.json", {"key": {}} if progress is None else progress)


def test_queue_launches_only_after_dependency(tmp_path, monkeypatch):
    selection = prepare(tmp_path, monkeypatch)
    predecessor(tmp_path)
    launched = []

    class Child:
        pid = 123

        def wait(self):
            return 0

    def popen(command, **kwargs):
        status = read_json(tmp_path / "state/rollout-queue/new/status.json")
        assert status["predecessor"]["tasks"] == 1
        assert status["stage"] == "launching"
        launched.append(command)
        return Child()

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    assert launcher.run_queued(tmp_path, "new", selection, "old") == 0
    assert launched == [[sys.executable, "-m", "runtime.scripts.supervise_rollout", "new", "--task-id", "test-task"]]
    assert read_json(tmp_path / "state/rollout-queue/new/status.json")["stage"] == "finished"
    with pytest.raises(FileExistsError):
        launcher.run_queued(tmp_path, "new", selection, "old")


@pytest.mark.parametrize("code,progress", [(1, {"key": {}}), (0, {})])
def test_incomplete_dependency_prevents_launch(tmp_path, monkeypatch, code, progress):
    selection = prepare(tmp_path, monkeypatch)
    predecessor(tmp_path, code, progress)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("launched early"))
    with pytest.raises(RuntimeError):
        launcher.run_queued(tmp_path, "new", selection, "old")
    assert read_json(tmp_path / "state/rollout-queue/new/status.json")["stage"] == "failed"


def test_config_change_prevents_launch(tmp_path, monkeypatch):
    selection = prepare(tmp_path, monkeypatch)
    predecessor(tmp_path)
    configs = iter([{"config_hash": "initial"}, {"config_hash": "changed"}])
    monkeypatch.setattr(launcher, "resolve", lambda *args: next(configs))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("launched changed config"))
    with pytest.raises(RuntimeError, match="config.yaml changed"):
        launcher.run_queued(tmp_path, "new", selection, "old")


def test_waits_for_live_supervisor_exit(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    # The actual pidfd event must hold the queue until the supervisor writes its receipt.
    code = "import sys;sys.stdin.read()"
    child = subprocess.Popen([sys.executable, "-c", code, "runtime.scripts.supervise_rollout", "old"], stdin=subprocess.PIPE)
    write_json(tmp_path / "runs/old/process.json", {"supervisor_pid": child.pid})
    result = []
    errors = []

    def wait():
        try:
            result.append(launcher.wait_predecessor(tmp_path, "old"))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=wait)
    thread.start()
    try:
        thread.join(timeout=0.1)
        assert thread.is_alive()
        predecessor(tmp_path)
        child.communicate(timeout=5)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors
        assert result[0]["tasks"] == 1
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        thread.join(timeout=5)
