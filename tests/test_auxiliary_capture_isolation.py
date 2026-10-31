from types import SimpleNamespace

import pytest


def test_auxiliary_selection_is_local_and_bound_requests_keep_capture():
    pytest.importorskip("starlette")
    from runtime.serving.vllm_plugin import DyConCapturePlugin, REQUEST_CONTEXT, STREAM

    def process():
        return SimpleNamespace(prompt_token_ids=[1, 2], sampling_params=SimpleNamespace(),
                               capture_select=None)

    async def add(*args, **kwargs):
        pass

    async def generate(*args, **kwargs):
        yield None

    plugin = DyConCapturePlugin()
    plugin.engine = SimpleNamespace(input_processor=SimpleNamespace(process_inputs=process),
                                    _add_request=add, generate=generate)
    plugin._attach_admission()
    token = REQUEST_CONTEXT.set(None)
    try:
        auxiliary = plugin.engine.input_processor.process_inputs()
        assert auxiliary.capture_select == {STREAM: {
            "prompt": "all", "generation": "all",
            "exclude_prompt_window": [0, None],
            "exclude_generation_window": [0, None],
        }}
        bound = REQUEST_CONTEXT.set({"session_id": "primary", "anchors": ["think_close"]})
        try:
            assert plugin.engine.input_processor.process_inputs().capture_select is None
        finally:
            REQUEST_CONTEXT.reset(bound)
        assert plugin.engine.input_processor.process_inputs().capture_select == auxiliary.capture_select
        assert plugin.request_metadata == {}
    finally:
        REQUEST_CONTEXT.reset(token)


@pytest.mark.parametrize("computed,prompt,output", [(0, 4, 0), (4, 4, 1), (100, 4, 97)])
def test_native_exclusion_selects_no_rows(computed, prompt, output):
    torch = pytest.importorskip("torch")
    capture = pytest.importorskip("vllm.capture")
    from vllm.model_hooks.selection.batch import BatchView
    from vllm.model_hooks.selection.runtime import collect_positions_apply_spec

    wire = capture.SelectSpec(prompt="all", generation="all",
        exclude_prompt_window=(0, None), exclude_generation_window=(0, None)).to_wire()
    batch = BatchView(query_start_loc=torch.tensor([0, 4]),
        num_computed=torch.tensor([computed]), num_prompt=torch.tensor([prompt]),
        num_output=torch.tensor([output]))
    assert collect_positions_apply_spec(torch.tensor([198, 248069, 1, 2]), batch, wire) is None
