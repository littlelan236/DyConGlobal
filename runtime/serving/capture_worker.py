"""Observe native capture execution using host batch geometry, without GPU copies."""


def execution_at(intervals, request_id, position):
    matches = [row for row in intervals.get(request_id, []) if row["start"] <= position < row["end"]]
    if not matches:
        raise ValueError(f"Missing capture execution evidence: {request_id}:{position}")
    kinds = {(row["graph_execution"], row["batch_kind"]) for row in matches}
    if len(kinds) != 1:
        raise ValueError("Conflicting capture execution evidence")
    mode, kind = kinds.pop()
    return {"graph_execution": mode, "batch_kind": kind}


class CaptureWorkerExtension:
    def configure_capture_execution(self):
        from vllm.forward_context import get_forward_context

        session = self.model_runner.capture_session
        if hasattr(self, "_dycon_execution"):
            return True
        self._dycon_execution = {}
        original_graph = session.collect_graph_outputs
        original_flush = session._flush_hook

        def record(mode):
            geometry = get_forward_context().batch_geometry
            if geometry is None:
                raise RuntimeError("Native capture batch has no geometry")
            generated = geometry.num_output.tolist()
            kind = "mixed" if any(generated) and not all(generated) else "decode" if all(generated) else "prefill"
            computed = geometry.num_computed.tolist()
            boundaries = geometry.query_start_loc_cpu
            for i, request_id in enumerate(geometry.req_ids):
                start = int(computed[i])
                end = start + int(boundaries[i + 1] - boundaries[i])
                rows = self._dycon_execution.setdefault(request_id, [])
                if rows and rows[-1]["end"] == start and rows[-1]["graph_execution"] == mode and rows[-1]["batch_kind"] == kind:
                    rows[-1]["end"] = end
                else:
                    rows.append({"start": start, "end": end, "graph_execution": mode, "batch_kind": kind})

        def graph(num_tokens):
            record("full")
            return original_graph(num_tokens)

        def flush(*args):
            state = session.graph_state
            if not (state is not None and state.recording) and any(
                store is not None and store._pending for store in session._streams.values()
            ):
                mode = get_forward_context().cudagraph_runtime_mode.name.lower()
                record("eager" if mode == "none" else mode)
            return original_flush(*args)

        session.collect_graph_outputs = graph
        session._flush_hook = flush
        return True

    def fetch_capture_execution(self):
        result = self._dycon_execution
        self._dycon_execution = {}
        return result
