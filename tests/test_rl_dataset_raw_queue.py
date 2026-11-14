"""Out-of-order, refill, bounded-input and single-writer recovery contracts."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from core.io import hash_json, sha256
from RL.dataset.artifacts import publication
from RL.dataset.config import ReplayConfig
from RL.dataset.loader import iter_groups
from RL.dataset.materialize import materialize
from RL.dataset.replay.artifacts import ReplayWriter
from RL.dataset.replay.inputs import Prefetch
from RL.dataset.replay.output import AsyncWriter
from RL.dataset.replay.validation import raw_contract
from RL.dataset.replay.worker import execute_continuous, sampling_options
from test_rl_dataset import build, token_source


def config(**changes):
    value = {"schema_version": 2, "replay_id": "queue-test", "dataset": {"id": "test", "root": "/tmp"},
             "runtime": {"gpus": [0], "max_num_seqs": 2}, "storage": {"output_root": "/tmp"},
             "scheduling": {"max_inflight_requests": 2}, **changes}
    return ReplayConfig.model_validate(value)


def test_continuous_refills_by_id_before_long_request_finishes():
    rows = [{"branch_sample_id": f"b{i}", "group_id": "g", "prompt_count": 2,
             "completion_count": 1, "prompt_token_ids": [1, 2], "completion_token_ids": [i + 3]} for i in range(4)]
    class Engine:
        def __init__(self): self.active = {}; self.events = []; self.steps = 0
        def get_num_unfinished_requests(self): return len(self.active)
        def has_unfinished_requests(self): return bool(self.active)
        def add_request(self, rid, prompt, params):
            self.events.append(("add", params.trace_decode_token_ids[0], self.steps))
            self.active[rid] = params.trace_decode_token_ids
        def step(self):
            self.steps += 1
            rid = next(reversed(self.active)) if self.steps < 4 else next(iter(self.active))
            tokens = self.active.pop(rid); self.events.append(("finish", tokens[0], self.steps))
            return [NS(request_id=rid, finished=True, outputs=[NS(token_ids=tokens, logprobs=[{tokens[0]: NS(logprob=-.2)}])])]
    class Groups:
        END = object(); peak = 123
        def __init__(self): self.items = deque([{"members": rows}, self.END]); self.released = 0
        def get(self, **kwargs): return self.items.popleft() if self.items else None
        def release(self, group): self.released += 1
    class Writer:
        finished = set(); peak = 0; seconds = 0
        def __init__(self): self.seen = []
        def check(self): pass
        def put(self, row, output, values, **kwargs): self.seen.append((row["branch_sample_id"], output["token_ids"]))
        def close(self): pass
    engine = Engine(); groups = Groups(); writer = Writer(); runtime = {}
    execute_continuous(NS(llm_engine=engine), lambda **k: NS(**k, _validate_trace_replay=lambda *a: None), groups, writer,
                       lambda row: {"policy_sha256": "p"}, "m", config(), runtime, 10)
    assert dict(writer.seen) == {r["branch_sample_id"]: r["completion_token_ids"] for r in rows}
    assert engine.events.index(("add", 5, 1)) < engine.events.index(("finish", 3, 4))
    assert groups.released == 1 and runtime["queue_metrics"]["completed"] == 4


def test_cache_prefetch_never_rereads_native_ledgers(tmp_path, monkeypatch):
    dataset = build(token_source(tmp_path)); cache = Path(materialize(dataset, tmp_path / "cache")["path"])
    def forbidden(*args, **kwargs): raise AssertionError("Cache should not read native ledgers")
    monkeypatch.setattr("RL.dataset.replay.inputs.native_tokens", forbidden)
    source = Prefetch(dataset, cache, config(), vocab_size=100)
    try:
        group = None
        while group is None: group = source.get()
        assert len(group["members"]) == 6
        assert group["members"][0]["prompt_token_ids"] is group["members"][1]["prompt_token_ids"]
        source.release(group)
        assert source.peak <= source.limit
    finally: source.close()


def test_async_writer_failure_propagates_and_no_publication(tmp_path):
    dataset = build(token_source(tmp_path)); row = next(iter_groups(dataset))["members"][0]
    identity = {"replay_id": "writer-fail", "dataset_receipt_sha256": sha256(dataset / "receipt.json"),
                "raw_contract_sha256": hash_json(raw_contract()), "selection_sha256": "test"}
    with pytest.raises(RuntimeError, match="writer failed"):
        with publication(tmp_path / "out", "rl-replay", "writer-fail", signature=identity) as (stage, _):
            writer = AsyncWriter(ReplayWriter(stage, dataset, identity, 1024), 1)
            writer.put(row, {"token_ids": row["completion_token_ids"]}, [float("nan")] * row["completion_count"],
                       policy_sha256="p", model_sha256="m")
            writer.close()
    assert not (tmp_path / "out/RL/replays/writer-fail/receipt.json").exists()


def test_raw_sampling_keeps_internal_eos_and_ignores_history():
    row = {"completion_token_ids": [2, 7, 2], "completion_count": 3, "seed": 42}
    options = sampling_options(row)
    assert options["ignore_eos"] and options["stop_token_ids"] == []
    assert options["trace_decode_token_ids"] == [2, 7, 2] and options["max_tokens"] == 3
    assert options["temperature"] == options["top_p"] == 1 and options["top_k"] == -1
