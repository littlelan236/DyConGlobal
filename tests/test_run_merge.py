from application.execution.merge import choose_sample, sample_state


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
