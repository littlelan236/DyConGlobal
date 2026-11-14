"""Project worker extension: aggregate actual dispatch without changing kernels."""
from collections import Counter


class ReplayObservation:
    def replay_observe_start(self):
        runner = self.model_runner
        manager = runner.cudagraph_manager
        self._replay_dispatch = Counter()
        self._replay_shapes = Counter()
        original = manager.run_fullgraph
        def observed(desc, *args, **kwargs):
            result = original(desc, *args, **kwargs)
            self._replay_dispatch["FULL"] += 1
            self._replay_shapes[str(desc.num_tokens)] += 1
            return result
        manager.run_fullgraph = observed
        return self.replay_observe_snapshot()

    def replay_observe_snapshot(self):
        import torch
        runner = self.model_runner; manager = runner.cudagraph_manager
        mode = manager.cudagraph_mode
        return {"effective_mode": mode.name, "capture_shapes": [str(k) for k in manager.graphs],
                "actual_full_dispatch": dict(self._replay_dispatch),
                "actual_full_shape_counts": dict(self._replay_shapes),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "coverage": "model forward full-graph replay; sampler, IO and coordinator outside"}
