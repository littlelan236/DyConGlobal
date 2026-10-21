"""Contract, resume and compatibility tests without importing vLLM/GPU code."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ToyTokenizer, import_fixture

from core.config import load_snapshot, resolve, snapshot
from core.io import hash_json, read_json, sha256, write_json
from enrichment.trajectory import enrich
from hidden.backends.vllm.capture import native_layers, validate_result
from hidden.backends.vllm.engine import engine_args, gpu_reservation
from hidden.extract import extract_hidden


def test_legacy_config_and_snapshot_identity_unchanged(cfg):
    assert cfg["hidden"]["adapter"] == "vllm_eager"
    assert "vllm" in cfg["hidden"]
    selection = {"rollouts": {}}
    snapshot(cfg, "old", selection)
    assert load_snapshot(cfg["project_root"], "old") == read_json(
        cfg["project_root"] + "/runs/old/resolved-config.json"
    )
    new = resolve(cfg["config_source"], ["hidden.adapter=vllm_eager", "hidden.vllm.gpus=[4]"])
    assert new["hidden"]["vllm"]["enable_prefix_caching"]
    assert new["hidden"]["vllm"]["enable_chunked_prefill"]
    assert engine_args(new)["enforce_eager"] is True
    assert engine_args(new)["async_scheduling"] is False
    new["hidden"]["vllm"]["moe_backend"] = "triton"
    assert engine_args(new)["kernel_config"] == {"moe_backend": "triton"}
    assert cfg["serving"] == new["serving"]
    assert cfg["steering"] == new["steering"]


@pytest.mark.parametrize("layers", [[], [1, 1], [-1], [32]])
def test_reject_invalid_sites(layers):
    with pytest.raises(ValueError):
        native_layers(layers, 32)


def test_hf_site_mapping_skips_last_prenorm():
    assert native_layers([0, 15, 31], 32) == [0, 15, 32]
    assert native_layers([63, 1], 64) == [64, 1]


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def float(self):
        return self

    def numpy(self):
        return self.value


class Result:
    def __init__(self):
        self.outputs = [
            SimpleNamespace(request_id="a", prompt_token_ids=[7, 8], finished=True),
            SimpleNamespace(request_id="b", prompt_token_ids=[9], finished=True),
        ]
        self.layers = {0: None, 2: None}
        self.positions = [[1], [0]]
        self.tokens = [[8], [9]]

    def sample_positions(self, i):
        return self.positions[i]

    def sample_token_ids(self, i):
        return self.tokens[i]

    def sample_rows(self, i, layer):
        return Tensor([[i + layer, i + layer + 1]])


def test_labelled_result_retains_layer_and_input_order():
    rows = [{"sample_id": "s-a", "prefix_token_ids": [7, 8]}, {"sample_id": "s-b", "prefix_token_ids": [9]}]
    values = validate_result(Result(), rows, [1, 0], [2, 0], 2)
    np.testing.assert_array_equal(values["s-b"], [[3, 4], [1, 2]])


@pytest.mark.parametrize("failure", ["missing", "position", "token", "duplicate", "prompt", "unfinished"])
def test_reject_missing_or_wrong_target_labels(failure):
    result = Result()
    if failure == "missing":
        result.layers.pop(2)
    elif failure == "position":
        result.positions[0] = []
    elif failure == "token":
        result.tokens[1] = [99]
    elif failure == "duplicate":
        result.outputs[1].request_id = "a"
    elif failure == "prompt":
        result.outputs[0].prompt_token_ids = [7]
    else:
        result.outputs[0].finished = False
    with pytest.raises(ValueError):
        validate_result(
            result,
            [{"sample_id": "a", "prefix_token_ids": [7, 8]}, {"sample_id": "b", "prefix_token_ids": [9]}],
            [0, 1],
            [0, 2],
            2,
        )


class BatchAdapter:
    fingerprint = "fixture-model"
    hidden_size = 4
    batch_size = 3
    metadata = {"name": "vllm_eager", "runtime_source_sha256": "fixture"}

    def __init__(self, fail_at=None, mismatch=False):
        self.calls, self.metrics = [], []
        self.fail_at, self.mismatch = fail_at, mismatch

    def capture_batch(self, rows, layers, chunk_size):
        self.calls.append([r["sample_id"] for r in rows])
        if len(self.calls) == self.fail_at:
            raise RuntimeError("interrupted")
        if self.mismatch:
            return {}
        return {r["sample_id"]: np.ones((len(layers), 4), dtype=np.float32) for r in rows}

    def sites(self):
        return [{"kind": "final_norm", "block_index": None}]


def test_native_batches_resume_and_keep_manifests(cfg):
    import_fixture(cfg, resolved=True)
    enrich({**cfg, "source_run_id": "source"}, "dataset", ToyTokenizer())
    cfg["hidden"]["layers"] = [0]
    root = cfg["project_root"]
    write_json(root + "/hidden/manifest.json", {"schema_version": 1, "legacy": "preserve"})
    adapter = BatchAdapter(fail_at=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        extract_hidden({**cfg, "source_run_id": "source"}, "dataset", "capture", adapter)
    assert [len(x) for x in adapter.calls] == [3, 1]
    resumed = BatchAdapter()
    out = extract_hidden({**cfg, "source_run_id": "source"}, "dataset", "capture", resumed)
    assert [len(x) for x in resumed.calls] == [1]
    receipt = read_json(out / "receipt.json")
    assert receipt["backend"]["name"] == "vllm_eager"
    assert receipt["replay"] == "native_trajectory_apc"
    manifest = read_json(root + "/hidden/manifest.json")
    assert manifest["legacy"] == "preserve"
    binding = next(iter(manifest["native_captures"]["capture"]["rollouts"].values()))
    assert binding["task_id"] == "t-1" and binding["run_id"] == "source"
    assert {a["text"] for a in binding["anchors"]} == {"<think>", "</think>"}
    assert extract_hidden({**cfg, "source_run_id": "source"}, "dataset", "capture", resumed) == out
    assert len(resumed.calls) == 1
    with pytest.raises(ValueError, match="different inputs"):
        changed = BatchAdapter()
        changed.metadata = {"name": "changed-runtime"}
        extract_hidden({**cfg, "source_run_id": "source"}, "dataset", "capture", changed)


def test_batch_failure_never_commits_receipt(cfg):
    import_fixture(cfg)
    enrich({**cfg, "source_run_id": "source"}, "dataset", ToyTokenizer())
    cfg["hidden"]["layers"] = [0]
    with pytest.raises(ValueError, match="mismatched sample"):
        extract_hidden({**cfg, "source_run_id": "source"}, "dataset", "bad", BatchAdapter(mismatch=True))
    from pathlib import Path

    assert not (Path(cfg["project_root"]) / "hidden/outputs/bad/receipt.json").exists()


def test_prefix_is_verified_before_capture(cfg):
    rid = import_fixture(cfg)
    dataset = enrich({**cfg, "source_run_id": "source"}, "dataset", ToyTokenizer())
    cfg["hidden"]["layers"] = [0]
    path = dataset / f"{rid}.samples.json"
    rows = read_json(path)
    rows[0]["prefix_sha256"] = hash_json([99])
    write_json(path, rows)
    receipt = read_json(dataset / "receipt.json")
    receipt["files"][path.name] = sha256(path)
    write_json(dataset / "receipt.json", receipt)
    adapter = BatchAdapter()
    with pytest.raises(ValueError, match="Prefix checksum"):
        extract_hidden({**cfg, "source_run_id": "source"}, "dataset", "bad", adapter)
    assert not adapter.calls


def test_gpu_admission_rejects_occupied_or_missing_device(tmp_path, monkeypatch):
    import hidden.backends.vllm.engine as engine

    def occupied(command, **kwargs):
        if "--query-gpu=index,uuid" in command:
            return "4, GPU-test\n"
        return "GPU-test, 123\n"

    monkeypatch.setattr(engine.subprocess, "check_output", occupied)
    with pytest.raises(RuntimeError, match="occupied"):
        with gpu_reservation(tmp_path, [4]):
            pytest.fail("Must not acquire an occupied GPU")
    with pytest.raises(ValueError, match="does not exist"):
        with gpu_reservation(tmp_path, [5]):
            pytest.fail("Must not acquire a missing GPU")
    with pytest.raises(ValueError, match="unique"):
        with gpu_reservation(tmp_path, [4, 4]):
            pytest.fail("Must not acquire a duplicate GPU")


def test_replay_requires_apc(cfg):
    cfg["hidden"]["vllm"]["gpus"] = [0]
    cfg["hidden"]["vllm"]["enable_prefix_caching"] = False
    with pytest.raises(ValueError, match="enable_prefix_caching=true"):
        engine_args(cfg)


def replay_row(sid, ids, rollout="a", session="s", step=0):
    return {
        "sample_id": sid,
        "prefix_token_ids": ids,
        "prefix_sha256": hash_json(ids),
        "rollout_id": rollout,
        "session_id": session,
        "step_id": step,
    }


def cached_adapter(cfg, monkeypatch):
    import hidden.adapters.vllm as module

    completed, calls = [], []

    def capture(llm, rows, layers, layer_count, hidden_size, options):
        calls.append([r["sample_id"] for r in rows])
        assert len({module.replay_group(r) for r in rows}) == len(rows)
        samples = []
        for row in rows:
            ids = row["prefix_token_ids"]
            matches = [
                next((i for i, (a, b) in enumerate(zip(old, ids)) if a != b), min(len(old), len(ids)))
                for old in completed
            ]
            cached = min(max(matches, default=0), len(ids) - 1)
            samples.append({"sample_id": row["sample_id"], "cached_tokens": cached})
        completed.extend(r["prefix_token_ids"] for r in rows)
        return (
            {r["sample_id"]: np.ones((len(layers), hidden_size)) for r in rows},
            {
                "requests": len(rows),
                "prompt_tokens": sum(len(r["prefix_token_ids"]) for r in rows),
                "cached_tokens": sum(s["cached_tokens"] for s in samples),
                "samples": samples,
            },
        )

    monkeypatch.setattr(module, "capture_window", capture)
    cfg["hidden"]["vllm"].update(max_inflight_requests=4, max_num_seqs=2, max_model_len=100)
    return module.VLLMAdapter(None, cfg, 2, 4, {}), calls


def test_same_trajectory_waits_while_independent_trajectories_batch(cfg, monkeypatch):
    adapter, calls = cached_adapter(cfg, monkeypatch)
    rows = [
        replay_row("a0", [1, 2]),
        replay_row("a1", [1, 2, 3]),
        replay_row("b0", [9, 8], "b"),
        replay_row("b1", [9, 8, 7], "b"),
    ]
    values = adapter.capture_batch(rows, [0], 1024)
    assert calls == [["a0", "b0"], ["a1", "b1"]]
    assert list(values) == [r["sample_id"] for r in rows]
    assert [w["cached_tokens"] for w in adapter.metrics] == [0, 4]
    assert adapter.metrics[1]["samples"][0]["previous_sample_id"] == "a0"
    assert adapter.metrics[1]["samples"][0]["append_only"]


def test_cache_and_predecessor_continue_across_submission_windows(cfg, monkeypatch):
    adapter, calls = cached_adapter(cfg, monkeypatch)
    adapter.capture_batch([replay_row("a0", [1, 2])], [0], 1024)
    adapter.capture_batch([replay_row("a1", [1, 2, 3, 4])], [0], 1024)
    sample = adapter.metrics[-1]["samples"][0]
    assert calls == [["a0"], ["a1"]]
    assert sample["cached_tokens"] == sample["common_prefix_tokens"] == 2
    assert sample["uncached_prompt_tokens"] == 2
    assert sample["previous_sample_id"] == "a0"
    adapter.capture_batch([replay_row("branch", [9, 8])], [0], 1024)
    sample = adapter.metrics[-1]["samples"][0]
    assert sample["common_prefix_tokens"] == sample["cached_tokens"] == 0
    assert not sample["append_only"]


def test_session_boundaries_and_unscoped_rows_are_safe(cfg, monkeypatch):
    adapter, calls = cached_adapter(cfg, monkeypatch)
    adapter.capture_batch([replay_row("a0", [1, 2]), replay_row("a1", [1, 2, 3], session="other")], [0], 1024)
    assert calls == [["a0", "a1"]]
    assert all(s["previous_sample_id"] is None for s in adapter.metrics[-1]["samples"])
    plain = [{"sample_id": str(i), "prefix_token_ids": [i], "prefix_sha256": hash_json([i])} for i in [4, 5]]
    adapter.capture_batch(plain, [0], 1024)
    assert calls[-2:] == [["4"], ["5"]]


def test_duplicate_sample_rejected_before_any_native_request(cfg, monkeypatch):
    adapter, calls = cached_adapter(cfg, monkeypatch)
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.capture_batch([replay_row("same", [1]), replay_row("same", [2])], [0], 1024)
    assert not calls


def test_failed_wave_does_not_submit_later_anchors(cfg, monkeypatch):
    import hidden.adapters.vllm as module

    adapter, calls = cached_adapter(cfg, monkeypatch)
    capture = module.capture_window

    def fail(llm, rows, *args):
        if rows[0]["sample_id"] == "fail":
            raise RuntimeError("capture interrupted")
        return capture(llm, rows, *args)

    monkeypatch.setattr(module, "capture_window", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        adapter.capture_batch(
            [replay_row("first", [1]), replay_row("fail", [1, 2]), replay_row("never", [1, 2, 3])], [0], 1024
        )
    assert calls == [["first"]]
    assert len(adapter.metrics) == 1


def test_receipt_metrics_exclude_previous_queue_jobs(cfg):
    import_fixture(cfg)
    enrich({**cfg, "source_run_id": "source"}, "dataset", ToyTokenizer())
    cfg["hidden"]["layers"] = [0]
    adapter = BatchAdapter()
    adapter.metrics.append({"sample_id": "another-queue-job"})
    out = extract_hidden({**cfg, "source_run_id": "source"}, "dataset", "isolated-metrics", adapter)
    assert read_json(out / "receipt.json")["execution_metrics"] == []
