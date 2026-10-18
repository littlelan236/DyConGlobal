import os
from contextlib import contextmanager
from threading import Event, Thread

import pytest

from core.config import snapshot
from core.io import hash_json, read_json, write_json
from hidden.evaluation.first_rollout import first_completed, wait_first
from hidden.evaluation.replay_online_artifacts import replay_config


def completion(run, iid, finished, status="partial"):
    path = run / "tasks" / iid / "attempts/attempt-000/attempt.json"
    write_json(
        path,
        {"instance_id": iid, "rollout_id": "r-" + iid, "finished_at": finished, "capture_status": status},
    )
    return path


def test_first_completion_does_not_skip_partial_capture(tmp_path):
    completion(tmp_path, "later", "2026-10-18T14:40:00+08:00", "complete")
    completion(tmp_path, "first", "2026-10-18T14:38:00+08:00", "partial")
    result = first_completed(tmp_path, {"first", "later"})
    assert result["instance_id"] == "first"
    assert result["capture_status"] == "partial"


def test_trigger_waits_on_progress_event_without_polling(tmp_path, monkeypatch):
    import hidden.evaluation.first_rollout as module

    run = tmp_path / "runs/r"
    write_json(run / "process.json", {"pid": os.getpid()})
    ready = Event()
    original = module.directory_changes

    @contextmanager
    def notify_ready(path):
        with original(path) as changed:
            ready.set()
            yield changed

    monkeypatch.setattr(module, "directory_changes", notify_ready)

    def finish():
        assert ready.wait(3)
        completion(run, "first", "2026-10-18T14:38:00+08:00")
        write_json(run / "progress.json", {"first": "finished"})

    writer = Thread(target=finish)
    writer.start()
    result = wait_first(tmp_path, "r", {"first"}, 5)
    writer.join(3)
    assert result["instance_id"] == "first"
    assert not writer.is_alive()


def test_shared_gpu_is_explicit_and_requires_free_memory(tmp_path, monkeypatch):
    from hidden.backends.vllm import engine

    def query(command, **kwargs):
        if "--query-gpu=index,uuid" in command:
            return "4, GPU-test\n"
        if "--query-gpu=index,memory.free" in command:
            return "4, 18000\n"
        return "GPU-test, 123\n"

    monkeypatch.setattr(engine.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="occupied"):
        with engine.gpu_reservation(tmp_path, [4]):
            pass
    with engine.gpu_reservation(tmp_path, [4], allow_shared=True, min_free_bytes=14 * 1024**3):
        pass
    with pytest.raises(RuntimeError, match="Insufficient"):
        with engine.gpu_reservation(tmp_path, [4], allow_shared=True, min_free_bytes=20 * 1024**3):
            pass


def test_replay_uses_source_identity_and_only_new_resource_settings(cfg):
    cfg["capture"]["layers"] = [0, 1]
    cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    original_hash = cfg["config_hash"]
    snapshot(cfg, "source", {"tasks": {}})
    options = [
        "model.fingerprint=changed-current-model",
        "generation.temperature=0.1",
        "hidden.vllm.max_num_seqs=1",
        "hidden.vllm.tensor_parallel_size=" + str(cfg["serving"]["tensor_parallel_size"]),
    ]
    result = replay_config(cfg["config_source"], options, "source")
    assert result["model"]["fingerprint"] == "fixture-model"
    assert result["generation"]["temperature"] == cfg["generation"]["temperature"]
    assert result["hidden"]["layers"] == [0, 1]
    assert result["hidden"]["vllm"]["max_num_seqs"] == 1
    assert result["source_run_config_hash"] == original_hash
    assert (
        read_json(cfg["project_root"] + "/runs/source/resolved-config.json")["config_hash"] == original_hash
    )
