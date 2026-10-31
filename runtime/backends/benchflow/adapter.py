"""Preserve the existing Rollout/OwnedPlanes/live verifier lifecycle."""
from runtime.backends.base import BackendAdapter
from runtime.backends.contracts import PreflightFacts


class BenchFlowAdapter(BackendAdapter):
    def make_factory(self, cfg, run_id):
        from runtime.agents.skillsbench import SkillsBenchFactory
        return SkillsBenchFactory(cfg, run_id)

    def preflight(self, context, tasks):
        from core.io import write_json
        from runtime.agents.skillsbench import preflight
        super().preflight(context, tasks)
        checked = preflight(context.cfg["project_root"], context.run_id,
                            workers=min(8, context.cfg["execution"]["rollout_workers"]))
        write_json(self.run_path(context) / "validation/startup-preflight.json", checked)
        return PreflightFacts(not checked["blockers"], tuple(checked["blockers"]), checked)

    def startup(self, context, cfg, task, runtime):
        # The existing preflight already initializes the actual ACP/tool driver.
        pass

    def execute_attempt(self, context, attempt):
        from runtime.agents.skillsbench import execute_attempt
        return execute_attempt(context.cfg, context.run_id, attempt.task, attempt.runtime,
                               attempt.sample, attempt.context, attempt.output, context.factory)
