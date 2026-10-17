"""Reuse mini-swe-agent behavior, adding atomic durable saves only."""

from pathlib import Path

from minisweagent.agents.default import DefaultAgent

from core.io import write_json


class DurableAgent(DefaultAgent):
    def save(self, path, *extra_dicts):
        data = self.serialize(*extra_dicts)
        if path:
            write_json(Path(path), data)
        return data
