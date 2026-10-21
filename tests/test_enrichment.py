from application.execution.evidence import RunEvidence
from pathlib import Path

import pytest
from conftest import ToyTokenizer, import_fixture

from core.io import read_json
from enrichment.trajectory import enrich, reasoning_count, reconstruct


def test_success_labels_and_causal_boundaries(cfg):
    rid = import_fixture(cfg, resolved=True)
    out = enrich({**cfg, "source_run_id": "source"}, "dataset", ToyTokenizer())
    rows = read_json(out / f"{rid}.samples.json")
    assert len(rows) == 4
    opened, closed = rows[:2]
    tok = ToyTokenizer()
    assert "reason-1" not in tok.decode(opened["prefix_token_ids"])
    assert tok.decode(opened["prefix_token_ids"]).endswith("<think>")
    text = tok.decode(closed["prefix_token_ids"])
    assert text.endswith("reason-1</think>")
    assert "FUTURE_ACTION_SECRET" not in text and "OBSERVATION_SECRET" not in text
    assert opened["labels"] == {"remaining_steps": 2, "remaining_reasoning_tokens": 5}
    assert closed["labels"] == {"remaining_steps": 2, "remaining_reasoning_tokens": 2}
    assert enrich({**cfg, "source_run_id": "source"}, "dataset", tok) == out  # verified idempotence


@pytest.mark.parametrize("resolved", [False, None])
def test_failed_or_ungraded_has_no_success_labels(cfg, resolved):
    rid = import_fixture(cfg, resolved=resolved)
    out = enrich({**cfg, "source_run_id": "source"}, "dataset", ToyTokenizer())
    for row in read_json(out / f"{rid}.samples.json"):
        assert row["labels"]["remaining_steps"] is None
        assert row["labels"]["remaining_reasoning_tokens"] is None
        assert row["terminal_remaining_steps"] > 0


def test_missing_usage_does_not_become_zero():
    assert reasoning_count({"response": None, "reasoning": ""}, ToyTokenizer())[0] is None


def test_prefix_count_mismatch_is_rejected(cfg):
    from core.io import write_json
    from enrichment.reader import TrajectoryReader

    p = Path(cfg["project_root"]) / "bad.json"
    write_json(
        p,
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "reasoning_content": "x",
                    "extra": {"response": {"usage": {"prompt_tokens": 999}}},
                },
            ]
        },
    )
    r = TrajectoryReader(p)
    with pytest.raises(ValueError, match="count mismatch"):
        reconstruct(next(r.steps()), r, ToyTokenizer(), cfg)
