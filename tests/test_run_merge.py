from application.execution.merge import _legacy_samples, choose_sample, sample_state


def _value(status, disposition=None):
    return {"exit_status": status, "disposition": disposition or ("accepted" if status in {"Submitted", "AgentCompleted"} else "failed")}


def test_merge_precedence_rules():
    cfg = {"execution": {"limits_retry": {"enabled": False}}}
    assert sample_state(cfg, _value("Pending", "pending")) == "pending"
    assert sample_state(cfg, _value("Submitted")) == "accepted"
    assert sample_state(cfg, _value("ValueError", "failed")) == "failed"
    assert choose_sample(_value("Pending", "pending"), _value("Submitted"), cfg, cfg) == "source"
    assert choose_sample(_value("ValueError", "failed"), _value("AgentCompleted"), cfg, cfg) == "source"
    assert choose_sample(_value("Submitted"), _value("Submitted"), cfg, cfg) == "destination"
    assert choose_sample(_value("Pending", "pending"), _value("ValueError", "failed"), cfg, cfg) == "source"


def test_legacy_progress_rebuilds_samples_from_evidence():
    loaded = {
        "tasks": {"tasks": [{"task_key": "task-1", "instance_id": "repo__issue-1"}]},
        "work_items": {"items": [{
            "task_key": "task-1", "instance_id": "repo__issue-1",
            "repetition_id": "rep-000", "repetition_index": 0,
            "sample_key": "sample-1", "sample_seed": 42,
            "seed_policy": "legacy_unrecorded",
        }]},
    }
    evidence = {"rollouts": {"rollout-1": {
        "task_key": "task-1", "attempt_id": "attempt-000",
        "termination_reason": "Submitted",
    }}}
    samples = _legacy_samples(
        "legacy-run", loaded, {"task-1": "legacy-digest"}, evidence,
        {"execution": {"limits_retry": {"enabled": False}}},
    )
    assert list(samples) == ["sample-1"]
    assert samples["sample-1"]["rollout_id"] == "rollout-1"
    assert samples["sample-1"]["instance_id"] == "repo__issue-1"
    assert samples["sample-1"]["disposition"] == "accepted"
