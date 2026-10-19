"""Reuse mini-swe-agent behavior, adding atomic durable saves only."""

import time
from pathlib import Path

from minisweagent.agents.default import DefaultAgent

from core.io import write_json


class DurableAgent(DefaultAgent):
    def execute_actions(self, message):
        hook = getattr(self.model, "record_action_phase", None)
        if hook is None or not message.get("extra", {}).get("dycon_parallel"):
            return super().execute_actions(message)
        hook(message, "started")
        start = time.monotonic()
        try:
            observations = super().execute_actions(message)
        except Exception as exc:
            exits = getattr(exc, "messages", [])
            submitted = any(m.get("extra", {}).get("exit_status") == "Submitted" for m in exits)
            hook(message, "completed" if submitted else "failed",
                 {"agent_flow": type(exc).__name__, "submitted": submitted,
                  "elapsed_ms": (time.monotonic() - start) * 1000})
            raise
        from core.io import hash_json

        hook(message, "completed", {"observations_sha256": hash_json(observations),
                                    "observations": len(observations),
                                    "elapsed_ms": (time.monotonic() - start) * 1000})
        return observations

    def save(self, path, *extra_dicts):
        data = self.serialize(*extra_dicts)
        if path:
            write_json(Path(path), data)
        return data
