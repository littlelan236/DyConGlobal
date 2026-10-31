"""Explicit SWE environment, mini driver and patch replay binding."""
from rollout.runner import _execute_mini_attempt
from runtime.backends.base import BackendAdapter


class NativeSWEAdapter(BackendAdapter):
    def make_factory(self, cfg, run_id):
        from runtime.containers.factory import ContainerFactory
        return ContainerFactory(cfg)

    def execute_attempt(self, context, attempt):
        return _execute_mini_attempt(context.cfg, context.run_id, attempt.task, context.factory,
                                     attempt.runtime, attempt.sample, attempt.context, attempt.output, self)

    def environment_class(self):
        from runtime.containers.rollout_adapter import RolloutEnvironment
        return RolloutEnvironment

    def agent_class(self):
        from rollout.agent import DurableAgent
        return DurableAgent

    def agent_config(self, cfg):
        return cfg["agent"]

    def prompt(self, task):
        return task["instance"]["problem_statement"]

    def finalize_subject(self, task, output, context, env, submission, status):
        return None, []

    def subject_spec(self, task, output, context, state):
        return {}

    def verify_subject(self, root, row):
        pass

    def subject_failed(self, root, row, state):
        return False
