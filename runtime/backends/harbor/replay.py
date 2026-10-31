"""Exact subject selection around the unchanged official Pro replay agent."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from harbor.agents.base import BaseAgent

from core.io import sha256


def exact_replay_class(evaluator):
    path = Path(evaluator["path"])
    if sha256(path) != evaluator["sha256"]:
        raise ValueError("Official Pro evaluator changed")
    spec = importlib.util.spec_from_file_location("dycon_official_pro_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class ExactReplayAgent(module.PatchReplayAgent):
        def __init__(self, *args, exact_subject, **kwargs):
            super().__init__(*args, source_job="unused-exact-subject", **kwargs)
            self.exact_subject = exact_subject

        def _find_patch(self, task_name):
            patch = Path(self.exact_subject["path"])
            if not patch.is_file() or patch.is_symlink() or sha256(patch) != self.exact_subject["sha256"]:
                raise ValueError("Missing/corrupt exact Pro patch subject")
            if task_name != self.exact_subject["instance_id"]:
                raise ValueError("Pro replay selected another task")
            return patch

    return ExactReplayAgent


class ExactReplayAgent(BaseAgent):
    """Factory-compatible late binding; upstream owns apply/fallback semantics."""
    def __init__(self, *args, evaluator, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner = exact_replay_class(evaluator)(*args, **kwargs)

    @staticmethod
    def name():
        return "dycon-exact-official-pro-replay"

    def version(self):
        return "1"

    async def setup(self, environment):
        await self.inner.setup(environment)

    async def run(self, instruction, environment, context):
        await self.inner.run(instruction, environment, context)
