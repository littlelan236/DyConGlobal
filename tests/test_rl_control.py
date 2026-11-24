from pathlib import Path
from core.io import write_json, read_json
from RL.control import cancel_if_rl


def test_non_rl_run_falls_through(tmp_path):
    assert cancel_if_rl({"project_root": str(tmp_path)}, "rollout", "reason") is None


def test_rl_cancel_targets_only_bound_supervisor_and_confirms_exit(tmp_path, monkeypatch):
    import RL.observability as obs
    directory = tmp_path/"RL/checkpoints/train"
    write_json(directory/"attempt.json", {"attempt_id": "train"})
    record = {"pid": 123456789, "starttime": "test", "command": ["dycon", "rl-train", "--training-config", "train.yaml"]}
    monkeypatch.setattr(obs, "training_process", lambda *a: record)
    monkeypatch.setattr(obs, "process", lambda pid: record if pid == record["pid"] else None)
    live = [True]
    monkeypatch.setattr(obs, "matches", lambda item: live[0])
    calls = []
    def kill(pid, signal):
        calls.append(pid)
        live[0] = False
    monkeypatch.setattr("RL.control.os.kill", kill)
    result = cancel_if_rl({"project_root": str(tmp_path)}, "train", "switch to eight GPUs")
    assert calls == [record["pid"]]
    assert result["cleanup_confirmed"]
    assert read_json(directory/"cancel-receipt.json")["cleanup_confirmed"]
