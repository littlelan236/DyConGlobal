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

        import os
        if os.environ.get("DYCON_GENERATION_ENTROPY") == "1":
            self.configure_generation_entropy()
        session = self.model_runner.capture_session
        if hasattr(self, "_dycon_execution"):
            return True
        self._dycon_execution = {}
        original_graph = session.collect_graph_outputs
        original_flush = session._flush_hook
        original_prepare = session.prepare_batch
        self._dycon_selector = {}
        self._dycon_selector_errors = 0
        import json
        import os
        self._dycon_close_ids = frozenset(json.loads(
            os.environ.get("DYCON_CAPTURE_CLOSE_IDS", "[248069]")
        ))

        def prepare(geometry):
            result = original_prepare(geometry)
            try:
                self._observe_close_candidates(geometry, session)
            except Exception:
                # Diagnostics must not affect scheduling, capture, or generation.
                self._dycon_selector_errors += 1
                for request_id in geometry.req_ids:
                    report = self._dycon_selector.setdefault(request_id, {"positions": [], "count": 0})
                    report["observer_errors"] = report.get("observer_errors", 0) + 1
            return result

        def record(mode):
            geometry = get_forward_context().batch_geometry
            if geometry is None:
                raise RuntimeError("Native capture batch has no geometry")
            generated = geometry.num_output.tolist()
            kind = (
                "mixed"
                if any(generated) and not all(generated)
                else "decode"
                if all(generated)
                else "prefill"
            )
            computed = geometry.num_computed.tolist()
            boundaries = geometry.query_start_loc_cpu
            for i, request_id in enumerate(geometry.req_ids):
                start = int(computed[i])
                end = start + int(boundaries[i + 1] - boundaries[i])
                rows = self._dycon_execution.setdefault(request_id, [])
                if (
                    rows
                    and rows[-1]["end"] == start
                    and rows[-1]["graph_execution"] == mode
                    and rows[-1]["batch_kind"] == kind
                ):
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
        session.prepare_batch = prepare
        return True

    def _observe_close_candidates(self, geometry, session):
        """Use native predicates on close candidates already available on the host.

        Only this forward's generation inputs are examined. Prompt prefixes and
        complete output histories are never rescanned; no GPU reads are added.
        """
        store = session._streams.get("tact_states")
        if store is None:
            return
        close_ids = self._dycon_close_ids
        batch = self.model_runner.input_batch
        boundaries = geometry.query_start_loc_cpu
        for i, request_id in enumerate(geometry.req_ids):
            report = self._dycon_selector.get(request_id)
            if report is None:
                report = self._dycon_selector[request_id] = {"positions": [], "count": 0}
            prompt_length = int(geometry.num_prompt[i])
            computed = int(geometry.num_computed[i])
            end = computed + int(boundaries[i + 1] - boundaries[i])
            start = max(computed, prompt_length)
            if start >= end:
                continue
            row = batch.req_id_to_index.get(request_id)
            if row is None:
                continue
            # This host array is maintained by synchronous vLLM scheduling.
            tokens = batch.token_ids_cpu[row, start:end]
            candidates = ([0] if int(tokens[0]) in close_ids else []) if len(tokens) == 1 else [
                offset for offset, token in enumerate(tokens) if int(token) in close_ids
            ]
            if not len(candidates):
                continue
            import torch
            from vllm.model_hooks.selection.batch import BatchView
            from vllm.model_hooks.selection.runtime import collect_positions_apply_spec

            spec = session._request_selects.get(request_id, {}).get(
                "tact_states", store.config.select
            )
            for offset in candidates:
                position = start + int(offset)
                selected = collect_positions_apply_spec(
                    torch.tensor([int(tokens[offset])], dtype=torch.long),
                    BatchView(
                        query_start_loc=torch.tensor([0, 1]),
                        num_computed=torch.tensor([position]),
                        num_prompt=torch.tensor([prompt_length]),
                        num_output=torch.tensor([max(1, int(geometry.num_output[i]))]),
                    ),
                    spec,
                ) if spec is not None else True
                if selected is not True and selected is None:
                    continue
                if selected is not True and selected.numel() == 0:
                    continue
                report["count"] += 1
                if len(report["positions"]) < 256:
                    report["positions"].append(position)

    def fetch_capture_execution(self):
        result = self._dycon_execution
        self._dycon_execution = {}
        result["__capture_diagnostics__"] = {
            "requests": self._dycon_selector,
            "observer_errors": self._dycon_selector_errors,
        }
        self._dycon_selector = {}
        self._dycon_selector_errors = 0
        return result

    def generation_entropy(self, request_id, token_ids):
        observer = getattr(self, '_dycon_entropy', None)
        if observer is None:
            raise ValueError('Generation entropy observer is unavailable')
        return observer.take(request_id, token_ids)

    def configure_generation_entropy(self):
        if not hasattr(self, '_dycon_entropy'):
            from runtime.serving.generation_entropy import SamplerEntropy
            self._dycon_entropy = SamplerEntropy(self.model_runner.sampler)
        return True
