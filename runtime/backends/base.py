"""Run-scoped facade over existing factories and their lease ownership."""
from __future__ import annotations

from pathlib import Path

from runtime.backends.contracts import BackendRunContext, CleanupFacts, PreflightFacts


class BackendAdapter:
    def __init__(self, binding):
        self.binding = binding

    def open_run(self, cfg, run_id, factory=None):
        return BackendRunContext(self.binding, cfg, run_id,
                                 factory if factory is not None else self.make_factory(cfg, run_id), self)

    def make_factory(self, cfg, run_id):
        raise NotImplementedError

    def preflight(self, context, tasks):
        context.factory.lock_images([t["instance"] for t in tasks.values()])
        return PreflightFacts(True)

    def startup(self, context, cfg, task, runtime):
        from runtime.serving.startup import check_tool_chain
        check_tool_chain(cfg, context.run_id, task, context.factory, runtime)

    def close_run(self, context, cancellation=None):
        if context.closed:
            return CleanupFacts("already_closed", self.binding.adapter)
        factory = context.factory
        if cancellation is not None and cancellation.event.is_set() and hasattr(factory, "cleanup_cancelled"):
            factory.cleanup_cancelled(context.run_id)
        else:
            factory.cleanup_owned(context.run_id)
        context.closed = True
        return CleanupFacts("completed", self.binding.adapter)

    @staticmethod
    def run_path(context):
        return Path(context.cfg["project_root"]) / "runs" / context.run_id
