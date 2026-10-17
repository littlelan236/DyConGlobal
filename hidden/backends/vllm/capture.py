"""Thin lifecycle adapter around existing labelled capture RPCs and SDK paging.

The SDK's public capture helper excludes tact_states, so only its lifecycle
is adapted here. Row selection, residual composition, cache admission, TP
assembly, paging and request matching remain in the pinned runtime/SDK.
"""

import numpy as np

STREAM = "tact_states"


def native_layers(layers, layer_count):
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Capture needs unique nonempty sites")
    if any(type(i) is not int or not 0 <= i < layer_count for i in layers):
        raise ValueError("Capture site is outside the text decoder")
    # HF excludes the last block's pre-norm output; native tact_states has
    # all N decoder outputs followed by final norm at native index N.
    return [layer_count if i == layer_count - 1 else i for i in layers]


def validate_result(result, rows, layers, native, hidden_size):
    """Require exactly the requested input token at every site, never decode."""
    if len(result.outputs) != len(rows) or set(result.layers) != set(native):
        raise ValueError("Capture returned incomplete samples or sites")
    values = {}
    request_ids = set()
    for i, row in enumerate(rows):
        output = result.outputs[i]  # LLM.generate contract: input order.
        ids = row["prefix_token_ids"]
        if list(output.prompt_token_ids or []) != ids:
            raise ValueError("vLLM output prompt differs from the submitted prefix")
        if output.request_id in request_ids or not output.finished:
            raise ValueError("Duplicate or unfinished capture request")
        request_ids.add(output.request_id)
        if result.sample_positions(i) != [len(ids) - 1] or result.sample_token_ids(i) != [ids[-1]]:
            raise ValueError("Target input token was skipped, duplicated or mislabelled")
        sample = np.stack([result.sample_rows(i, site).float().numpy()[0] for site in native])
        if sample.shape != (len(layers), hidden_size) or not np.isfinite(sample).all():
            raise ValueError("Invalid hidden shape or nonfinite values")
        if row["sample_id"] in values:
            raise ValueError("Duplicate sample_id in capture window")
        values[row["sample_id"]] = sample
    return values


def capture_window(llm, rows, layers, layer_count, hidden_size, options):
    from easysteer.capture.fetch import fetch_pages
    from easysteer.capture.result import CaptureResult
    from vllm import SamplingParams
    from vllm.capture import SelectSpec, validate_capture_topology

    native = native_layers(layers, layer_count)
    tp_size = None

    def rpc(method, *args, **kwargs):
        results = llm.collective_rpc(method, args=args, kwargs=kwargs)
        if tp_size is not None and len(results) != tp_size:
            raise RuntimeError("Incomplete capture worker group")
        if method in {"start_capture", "stop_capture"} and not all(x is True for x in results):
            raise RuntimeError(f"Capture RPC was not acknowledged: {method}")
        return results

    tp_size = validate_capture_topology(rpc("capture_status", STREAM))
    try:
        rpc(
            "start_capture",
            STREAM,
            layers=native,
            dtype="float32",
            select=SelectSpec(prompt_positions=[-1]).to_wire(),
            budget_bytes=options["capture_budget_bytes"],
            device_budget_bytes=options["capture_budget_bytes"],
        )
        outputs = llm.generate(
            [{"prompt_token_ids": row["prefix_token_ids"]} for row in rows],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        statuses = rpc("capture_status", STREAM)
        validate_capture_topology(statuses)
        if any(s["tokens_dropped"] or s["cache_elided_reqs"] for s in statuses):
            raise RuntimeError("Capture budget overflow or cache-elided target token")
        if any(s["graph_replays"] or s["graph_ready"] for s in statuses):
            raise RuntimeError("Eager extraction unexpectedly used capture graphs")
        tensors, meta, layouts = fetch_pages(rpc, STREAM, statuses, tp_size, options["fetch_bytes"], None)
        result = CaptureResult(tensors, meta, outputs, layouts=layouts, component=STREAM)
        values = validate_result(result, rows, layers, native, hidden_size)
        metrics = {
            "requests": len(rows),
            "prompt_tokens": sum(len(r["prefix_token_ids"]) for r in rows),
            "cached_tokens": sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outputs),
            "eager_capture_forwards": max(s["eager_capture_forwards"] for s in statuses),
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "request_id": output.request_id,
                    "anchor_position": len(row["prefix_token_ids"]) - 1,
                    "anchor_token_id": row["prefix_token_ids"][-1],
                    "cached_tokens": getattr(output, "num_cached_tokens", 0) or 0,
                }
                for row, output in zip(rows, outputs, strict=True)
            ],
        }
    except BaseException as error:
        try:
            rpc("stop_capture", STREAM)
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    else:
        rpc("stop_capture", STREAM)
    return values, metrics


def online_capture_spec(
    layers,
    *,
    anchor="all",
    cache_boundary=None,
    close_token_count=1,
    newline_token_ids=None,
    close_token_ids=None,
):
    """Build the native EasySteer capture request for a rollout forward.

    The opening marker is deliberately selected at prompt position ``-2``
    because Qwen chat templates append ``<think>\n``.  The close marker is
    selected only by the subsequent input forward and must be supplied by the
    caller after it is observed.
    """
    from vllm.capture import SelectSpec

    if anchor == "all":
        from hidden.backends.vllm.online import build_capture_select

        select = build_capture_select(newline_token_ids or [], close_token_ids or [])["tact_states"]
    elif anchor == "think_open":
        select = SelectSpec(prompt_positions=[-2]).to_wire()
    elif anchor == "reasoning_newline":
        select = SelectSpec(generation_tokens=list(newline_token_ids or [])).to_wire()
    elif anchor == "think_close":
        if close_token_ids:
            select = SelectSpec(generation_tokens=list(close_token_ids)).to_wire()
        else:
            if type(close_token_count) is not int or close_token_count <= 0:
                raise ValueError("close_token_count must be a positive integer")
            select = SelectSpec(prompt_positions=[-close_token_count]).to_wire()
    else:
        raise ValueError(f"Unknown online capture anchor: {anchor}")
    result = {"layers": list(layers), "select": select, "graph_mode": "FULL"}
    if cache_boundary is not None:
        if type(cache_boundary) is not int or cache_boundary < 0:
            raise ValueError("cache_boundary must be a nonnegative integer")
        result["capture_cache_boundary"] = cache_boundary
    return result


def start_online_capture(llm, stream, layers, *, anchor="think_open", cache_boundary=None, close_token_count=1, **options):
    """Ask every vLLM worker to use EasySteer's FULL capture graph.

    Scheduler, KV/GDN state, TP assembly and page fetching remain in the
    pinned runtime; this function only forwards the control request.
    """
    spec = online_capture_spec(
        layers, anchor=anchor, cache_boundary=cache_boundary, close_token_count=close_token_count
    )
    kwargs = {**options, **spec}
    results = llm.collective_rpc("start_capture", args=(stream,), kwargs=kwargs)
    if not results or not all(value is True for value in results):
        raise RuntimeError("online capture worker admission failed")
    return spec
