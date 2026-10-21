"""The production replay interface makes one ordered pass, including missing points."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.io import hash_json, read_json, sha256, write_json
from hidden.evaluation import replay_online_artifacts as replay


def row(sid, step, length, rollout="r"):
    ids = list(range(length))
    return {
        "sample_id": sid,
        "rollout_id": rollout,
        "session_id": "s",
        "step_id": step,
        "absolute_position": length - 1,
        "prefix_token_ids": ids,
        "prefix_sha256": hash_json(ids),
    }


def test_one_pass_orders_missing_points_with_compared_events_and_never_resets(tmp_path):
    rows = [row("later", 2, 5), row("first", 0, 2), row("missing", 1, 4)]
    metadata = {r["sample_id"]: {k: v for k, v in r.items() if k != "prefix_token_ids"} for r in rows}
    metadata["missing"].update(compared_to_online=False, online_status="missing")
    calls = []

    class Adapter:
        batch_size = 2
        metrics = []
        llm = object()  # No reset method exists: a reset would fail this test.

        def capture_batch(self, batch, layers, chunk_size):
            calls.extend(r["sample_id"] for r in batch)
            self.metrics.append({"requests": len(batch)})
            return {r["sample_id"]: np.ones((len(layers), 3)) for r in batch}

    windows = replay.replay_to_disk(Adapter(), rows, metadata, tmp_path, [0, 1], 1024, [0, 2])
    assert calls == ["first", "missing", "later"]
    assert sum(w["requests"] for w in windows) == 3
    assert {p.name for p in tmp_path.iterdir()} == {"trajectory-apc", "offline-only"}
    assert [r["sample_id"] for r in read_json(tmp_path / "trajectory-apc/index.json")] == ["first", "later"]
    missing = read_json(tmp_path / "offline-only/index.json")
    assert len(missing) == 1 and missing[0]["compared_to_online"] is False
    with np.load(tmp_path / "trajectory-apc/first.npz") as data:
        assert set(data.files) == {"layer_0", "layer_2"}


def test_duplicate_or_missing_metadata_rejected_before_capture(tmp_path):
    adapter = SimpleNamespace(batch_size=2, metrics=[])
    with pytest.raises(ValueError, match="Duplicate"):
        replay.replay_to_disk(adapter, [row("x", 0, 2)] * 2, {}, tmp_path, [0], 1, [0])
    with pytest.raises(ValueError, match="metadata"):
        replay.replay_to_disk(adapter, [row("x", 0, 2)], {}, tmp_path, [0], 1, [0])
    assert not list(tmp_path.iterdir())


def ledger_source(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "output"
    out.mkdir()
    ledger = {
        "request_id": "req",
        "session_id": "session",
        "rollout_id": "rollout",
        "step_id": 2,
        "prompt_token_ids": [1],
        "output_token_ids": [2, 3, 4],
    }
    write_json(source / "requests/req.json", ledger)
    event = {
        "request_id": "req",
        "sample_id": "x",
        "token_id": 3,
        "absolute_position": 2,
        "request_sha256": hash_json([1]),
        "anchor": "think_close",
    }
    events = tmp_path / "events.json"
    write_json(events, [event])
    monkeypatch.setattr(replay, "read_request", lambda *args: ledger)
    monkeypatch.setattr(replay, "artifact_bytes", lambda *args: b"reference")
    return source, out, events, event, ledger


def test_reference_preserves_trajectory_and_step_identity(tmp_path, monkeypatch):
    source, out, events, _, _ = ledger_source(tmp_path, monkeypatch)
    _, rows, _ = replay.prepare_reference(source, out, events)
    assert rows[0]["rollout_id"] == "rollout"
    assert rows[0]["session_id"] == "session" and rows[0]["step_id"] == 2
    assert rows[0]["prefix_token_ids"] == [1, 2, 3]


@pytest.mark.parametrize("position", [-1, 3, 100])
def test_reference_rejects_unprocessed_or_invalid_positions(tmp_path, monkeypatch, position):
    source, out, events, event, _ = ledger_source(tmp_path, monkeypatch)
    event["absolute_position"] = position
    write_json(events, [event])
    with pytest.raises(ValueError, match="processed input-token"):
        replay.prepare_reference(source, out, events)


def test_missing_points_rebind_to_exact_ledger_and_reject_changed_tokens(tmp_path, monkeypatch):
    source, _, _, _, _ = ledger_source(tmp_path, monkeypatch)
    path = tmp_path / "missing.json"
    item = {"sample_id": "missing", "prefix_token_ids": [1, 2, 3], "ledger_ref": "requests/req.json"}
    write_json(path, [item])
    result = replay.prepare_extras(source, path)[0]
    assert result["step_id"] == 2 and result["session_id"] == "session"
    assert result["compared_to_online"] is False
    item["prefix_token_ids"] = [1, 99]
    write_json(path, [item])
    with pytest.raises(ValueError, match="exact request ledger"):
        replay.prepare_extras(source, path)


def test_missing_points_use_finalized_rollout_identity(tmp_path, monkeypatch):
    source, _, _, _, ledger = ledger_source(tmp_path, monkeypatch)
    ledger.pop("rollout_id")
    trajectory = tmp_path / "canonical.json"
    write_json(trajectory, {})
    finalization = {
        "accepted_native_request_ids": ["req"],
        "trajectory_path": str(trajectory),
        "trajectory_sha256": sha256(trajectory),
        "rollout_id": "finalized-rollout",
    }
    write_json(source / "sessions/session/finalization.json", finalization)
    path = tmp_path / "missing.json"
    write_json(path, [{"sample_id": "missing", "prefix_token_ids": [1, 2, 3]}])
    result = replay.prepare_extras(source, path)[0]
    assert result["rollout_id"] == "finalized-rollout"
    finalization["accepted_native_request_ids"] = []
    write_json(source / "sessions/session/finalization.json", finalization)
    with pytest.raises(ValueError, match="unaccepted"):
        replay.prepare_extras(source, path)


def test_native_event_index_reads_only_bound_request(tmp_path, monkeypatch):
    source, out, events, event, ledger = ledger_source(tmp_path, monkeypatch)
    key = hash_json("req")
    (source / "requests/req.json").rename(source / "requests" / (key + ".json"))
    (source / "requests/unrelated.json").write_text("invalid unrelated request")
    event["request_key"] = key
    write_json(events, [event])
    monkeypatch.setattr(replay, "read_request", lambda base, proof: read_json(base / proof["ledger_ref"]))
    _, rows, _ = replay.prepare_reference(source, out, events)
    assert len(rows) == 1 and rows[0]["sample_id"] == "x"
