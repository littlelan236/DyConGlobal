"""CPU checks against the actual pinned capture SDK; no GPU engine required."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from hidden.backends.vllm.capture import capture_window  # noqa: E402


class Engine:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    def collective_rpc(self, method, args=(), kwargs=None):
        from vllm.model_hooks.capture.serialization import serialize_capture_layer

        self.calls.append((method, args, kwargs))
        if self.failure == method:
            raise RuntimeError(method)
        if method == "capture_status":
            return [
                {
                    "tokens_dropped": int(self.failure == "overflow"),
                    "cache_elided_reqs": int(self.failure == "elided"),
                    "graph_replays": int(self.failure == "graph"),
                    "graph_ready": False,
                    "eager_capture_forwards": 2,
                }
            ]
        if method == "fetch_captured":
            # Interleaved/reversed worker rows, including native uniqueness suffix.
            meta = torch.tensor([[0, 0, 9], [1, 1, 8]], dtype=torch.int32)
            table = {0: "b-12345678", 1: "a-87654321"}
            return [
                {
                    lid: serialize_capture_layer(
                        torch.tensor([[20.0, 21.0], [10.0, 11.0]]), meta, table, str(lid)
                    )
                    for lid in [0, 2]
                }
            ]
        return [True]

    def generate(self, prompts, sampling_params, use_tqdm):
        assert sampling_params.max_tokens == 1
        assert sampling_params.ignore_eos and not use_tqdm
        return [
            SimpleNamespace(request_id="a", prompt_token_ids=[7, 8], finished=True, num_cached_tokens=1),
            SimpleNamespace(request_id="b", prompt_token_ids=[9], finished=True, num_cached_tokens=0),
        ]


def run(engine):
    rows = [{"sample_id": "a", "prefix_token_ids": [7, 8]}, {"sample_id": "b", "prefix_token_ids": [9]}]
    return capture_window(engine, rows, [0, 1], 2, 2, {"capture_budget_bytes": 100000, "fetch_bytes": 10000})


def test_native_rpc_reuses_request_matching_and_labels():
    engine = Engine()
    values, metrics = run(engine)
    assert values["a"].tolist() == [[10, 11], [10, 11]]
    assert values["b"].tolist() == [[20, 21], [20, 21]]
    assert metrics["cached_tokens"] == 1
    start = next(x for x in engine.calls if x[0] == "start_capture")
    assert start[1] == ("tact_states",)
    assert start[2]["select"]["prompt_positions"] == [-1]
    assert engine.calls[-1][0] == "stop_capture"


@pytest.mark.parametrize("failure", ["start_capture", "fetch_captured", "overflow", "elided", "graph"])
def test_native_rpc_failure_cleans_stream(failure):
    engine = Engine(failure)
    with pytest.raises(RuntimeError):
        run(engine)
    assert engine.calls[-1][0] == "stop_capture"


def test_native_cache_policy_keeps_final_token_recomputable():
    from vllm.capture import SelectSpec
    from vllm.model_hooks.capture.policy import CaptureRequestPolicy

    policy = CaptureRequestPolicy()
    policy.record_rpc(
        "start_capture", ("tact_states",), {"select": SelectSpec(prompt_positions=[-1]).to_wire()}
    )
    assert not policy.skip_prefix_read(list(range(4097)), None)


def test_shared_capture_boundary_limits_native_lookup_before_hybrid_restore():
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    from runtime.serving.cache_boundary import install

    dispatch = CudaGraphManager.dispatch
    install()
    assert CudaGraphManager.dispatch is dispatch, "Cache bounds must not reclassify prompt chunks as decode"
    seen = []
    def lookup(block_hashes, maximum):
        seen.append(maximum)
        return (["native-blocks"], maximum // 16 * 16, 0)
    coordinator = SimpleNamespace(find_longest_cache_hit=lookup)
    manager = SimpleNamespace(
        coordinator=coordinator, prefix_cache_lookup_enabled=lambda request: True,
        create_kv_cache_blocks=lambda blocks: blocks, enable_kv_cache_events=False,
    )
    request = SimpleNamespace(num_tokens=129, block_hashes=["native-hash"],
                              sampling_params=SimpleNamespace(extra_args={"dycon_capture_cache_boundary": 95}))
    blocks, tokens, _ = KVCacheManager.get_computed_blocks(manager, request)
    assert seen == [95] and tokens == 80 and blocks == ["native-blocks"]
    assert coordinator.find_longest_cache_hit is lookup
    request.sampling_params.extra_args = {}
    KVCacheManager.get_computed_blocks(manager, request)
    assert seen[-1] == 128


@pytest.mark.parametrize("with_lora", [False, True])
def test_activation_graph_preserves_native_lora_cases_and_setup(monkeypatch, with_lora):
    from contextlib import contextmanager
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils, lora_utils, model_hook_utils

    events = []
    cases = [0, 2] if with_lora else [0]
    lora = object() if with_lora else None
    hook = object() if with_lora else None

    @contextmanager
    def setup(config):
        assert config is lora
        events.append("setup")
        try:
            yield
        finally:
            events.append("restore")

    @contextmanager
    def record(model):
        events.append("record")
        yield

    class Manager:
        def __init__(self, *args, **kwargs):
            assert kwargs["lora_capture_cases"] == cases
        def capture(self, *args, **kwargs):
            assert events == ["setup", "record"]
            assert kwargs["has_lora"] is with_lora
            assert kwargs["lora_capture_hook"] is hook
            assert kwargs["use_aux_hidden_state_outputs"] is False
            assert kwargs["skip_compiled"] is True
            events.append("capture")

    monkeypatch.setattr(cudagraph_utils, "ModelCudaGraphManager", Manager)
    monkeypatch.setattr(lora_utils, "create_lora_capture_hook", lambda config, runner: hook)
    monkeypatch.setattr(model_hook_utils.torch.accelerator, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(model_hook_utils, "current_platform", SimpleNamespace(graph_pool_handle=lambda: "separate-pool"))
    runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL, lora_capture_cases=cases),
        vllm_config=object(), device="cuda", decode_query_len=1, lora_config=lora,
        use_aux_hidden_state_outputs=False, maybe_setup_dummy_loras=setup,
        model=object(), model_state=object(), input_buffers=object(),
        intermediate_tensors=None, block_tables=object(), attn_groups=[], kv_cache_config=object(),
    )
    state = SimpleNamespace(record_outputs=record)
    graph = model_hook_utils._initialize_capture_graph(runner, state)
    assert graph.pool == "separate-pool"
    assert events == ["setup", "record", "capture", "restore"]
    assert state.allocation_bytes == 0
