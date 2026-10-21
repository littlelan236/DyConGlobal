from application.execution.evidence import RunEvidence
from pathlib import Path

from conftest import task

from rollout.artifacts import import_rollout
from core.io import read_json, write_json
from enrichment.token_trace import trace_rollout


class TraceTokenizer:
    name_or_path = "trace-fixture"
    vocab_size = 300
    _text = {10: "<think>", 11: "</think>", 1: "system", 2: "\n", 20: "a", 21: "b", 22: "c", 271: "\n\n"}

    def encode(self, text, add_special_tokens=False, **kwargs):
        if text == "<think>":
            return [10]
        if text == "</think>":
            return [11]
        if text == "a\n\nb":
            return [20, 271, 21]
        if text == "c":
            return [22]
        return [20 for _ in text]

    def decode(self, ids, **kwargs):
        return "".join(self._text.get(i, "x") for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return [1, 10, 2]


def test_global_remaining_reasoning_label(cfg):
    root = Path(cfg["project_root"])
    raw = root / "trace.traj.json"
    write_json(
        raw,
        {
            "instance_id": "trace-task",
            "info": {"exit_status": "Submitted", "submission": "patch"},
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "reasoning_content": "a\n\nb", "content": "x", "extra": {"response": {"choices": [{"finish_reason": "tool_calls"}]}}},
                {"role": "tool", "content": "obs"},
                {"role": "assistant", "reasoning_content": "c", "content": "done", "extra": {"response": {"choices": [{"finish_reason": "stop"}]}}},
            ],
        },
    )
    rid = import_rollout(root, {"source": str(raw), "task": task("trace-task"), "run_id": "trace", "attempt_id": "a0", "purposes": ["training"], "model_fingerprint": "fixture-model"})
    out = trace_rollout({**cfg, "source_run_id": "trace"}, rid, TraceTokenizer())
    meta = read_json(out / "token_trace.json")
    assert meta["reasoning_blocks"][0]["content_token_count"] == 3
    assert meta["reasoning_blocks"][1]["content_token_count"] == 1
    newline = next(row for row in meta["newline_events"] if row["token_id"] == 271)
    assert newline["newline_count"] == 2
    import numpy as np

    arrays = np.load(out / "token_trace.npz")
    index = newline["trace_index"]
    assert arrays["block_remaining_reasoning_after"][index] == 1
    assert arrays["remaining_reasoning_after"][index] == 2
