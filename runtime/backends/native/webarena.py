"""Browser, HAR finalization and offline evaluator binding for Verified."""
from runtime.backends.contracts import PreflightFacts
from runtime.backends.native.swe import NativeSWEAdapter


class NativeWebArenaAdapter(NativeSWEAdapter):
    def make_factory(self, cfg, run_id):
        from runtime.containers.webarena_adapter import WebArenaFactory
        return WebArenaFactory(cfg, run_id)

    def preflight(self, context, tasks):
        from core.io import write_json
        from runtime.containers.webarena_adapter import preflight
        super().preflight(context, tasks)
        checked = preflight(context.cfg["project_root"], context.run_id,
                            factory=context.factory, tasks=list(tasks.values()))
        write_json(self.run_path(context) / "validation/webarena-preflight.json", checked)
        return PreflightFacts(bool(checked.get("passed")), tuple(map(str, checked.get("blockers", []))), checked)

    def environment_class(self):
        from runtime.containers.webarena_adapter import WebArenaEnvironment
        return WebArenaEnvironment

    def agent_class(self):
        from runtime.agents.webarena import WebArenaAgent
        return WebArenaAgent

    def agent_config(self, cfg):
        profile = cfg["execution_context"]["profile"]
        return {**cfg["agent"], **{k: profile[k] for k in (
            "system_template", "instance_template", "observation_template", "format_error_template")}}

    def prompt(self, task):
        return task["instance"]["intent"]

    def finalize_subject(self, task, output, context, env, submission, status):
        from runtime.agents.webarena import finalize_subject
        return finalize_subject(task, output, context, env, submission, status)

    def subject_spec(self, task, output, context, state):
        from core.io import sha256
        subject = output / "web-subject/subject.json"
        if not subject.exists():
            subject, _ = self.finalize_subject(task, output, context, None, "", state.get("agent_exit_status", "Unknown"))
        return {"subject": str(subject), "subject_sha256": sha256(subject)}

    def verify_subject(self, root, row):
        from application.execution.subject import verify_subject
        verify_subject(root, row)

    def subject_failed(self, root, row, state):
        from core.io import read_json
        return (state.get("web_subject_error") or state.get("web_subject_errors") or
                (state["agent_exit_status"] == "Submitted" and not read_json(root / row["subject_path"])["complete"]))
