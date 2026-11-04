"""Preserve the existing Rollout/OwnedPlanes/live verifier lifecycle."""
from runtime.backends.base import BackendAdapter
from runtime.backends.contracts import PreflightFacts


class BenchFlowAdapter(BackendAdapter):
    def make_factory(self, cfg, run_id):
        from runtime.agents.skillsbench import SkillsBenchFactory
        return SkillsBenchFactory(cfg, run_id)

    def preflight(self, context, tasks):
        from core.io import read_json, write_json
        from runtime.agents.skillsbench import preflight
        from core.preflight import long_checks_enabled, timed_check
        admission_checks = {}
        deep = long_checks_enabled(context.cfg)
        with timed_check(admission_checks, "image_admission_oci_sha", enabled=deep):
            if deep:
                super().preflight(context, tasks)
            else:
                context.factory.lock_images([t["instance"] for t in tasks.values()], deep=False)
        checked = preflight(context.cfg["project_root"], context.run_id,
                            workers=min(8, context.cfg["execution"]["rollout_workers"]))
        checked["image_admission_checks"] = admission_checks
        model_receipt = self.run_path(context) / "validation/startup-model-identity.json"
        if model_receipt.exists():
            checked["runner_model_identity"] = read_json(model_receipt)
        write_json(self.run_path(context) / "validation/startup-preflight.json", checked)
        return PreflightFacts(not checked["blockers"], tuple(checked["blockers"]), checked)

    def startup(self, context, cfg, task, runtime):
        # The existing preflight already initializes the actual ACP/tool driver.
        pass

    def execute_attempt(self, context, attempt):
        from runtime.agents.skillsbench import execute_attempt
        return execute_attempt(context.cfg, context.run_id, attempt.task, attempt.runtime,
                               attempt.sample, attempt.context, attempt.output, context.factory)
