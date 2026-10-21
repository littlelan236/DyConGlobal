"""Native agent flow with bounded loop detection and atomic durable saves."""

import time
from collections import deque
from pathlib import Path

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded
from pydantic import Field

from core.io import hash_json, write_json


class DurableAgentConfig(AgentConfig):
    loop_repeat_limit: int | None = Field(default=None, ge=2, strict=True)
    loop_max_period: int = Field(default=7, ge=1, le=7, strict=True)


def _step_fingerprint(message):
    extra = message.get("extra", {})
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        choices = (extra.get("response") or {}).get("choices") or []
        reasoning = (choices[0].get("message") or {}).get("reasoning_content") if choices else None
    return hash_json({
        "reasoning": reasoning if reasoning is not None else "",
        "content": message.get("content") if message.get("content") is not None else "",
        # The native parser accepts bash only. Keep execution parameters and
        # ordering, excluding transport IDs which change on every request.
        "actions": [{"tool": "bash", "arguments": {k: v for k, v in action.items() if k != "tool_call_id"}}
                    for action in extra.get("actions", [])],
    })


class DurableAgent(DefaultAgent):
    def __init__(self, model, env, *, config_class=DurableAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self._loop_history = deque(maxlen=(self.config.loop_repeat_limit or 0) * self.config.loop_max_period)
        self._loop_step = 0
        self._loop_hit = None

    def run(self, task="", **kwargs):
        self._loop_history.clear()
        self._loop_step = 0
        self._loop_hit = None
        return super().run(task, **kwargs)

    def query(self):
        try:
            message = super().query()
        except FormatError:
            self._loop_history.clear()
            raise
        step = self._loop_step
        self._loop_step += 1
        repeats = self.config.loop_repeat_limit
        if repeats is None:
            return message
        self._loop_history.append(_step_fingerprint(message))
        history = list(self._loop_history)
        for period in range(1, self.config.loop_max_period + 1):
            size = repeats * period
            if len(history) < size or history[-size:] != history[-period:] * repeats:
                continue
            self._loop_hit = {
                "period": period, "repeats": repeats,
                "start_step": step - size + 1, "end_step": step,
                "action_executed": False,
                "match_policy": "exact_reasoning_content_and_actions_v1",
            }
            message.setdefault("extra", {}).update(action_executed=False, loop_detection=self._loop_hit)
            # Native formatting pads empty outputs with explicit not-executed
            # observations. Preserve tool-call pairing without running tools.
            self.add_messages(*self.model.format_observation_messages(message, [], self.get_template_vars()))
            raise LimitsExceeded({
                "role": "exit", "content": "Repeated reasoning-action loop detected",
                "extra": {"exit_status": "LimitsExceeded", "submission": "",
                          "limit_reason": "repeated_reasoning_action_loop", "loop_detection": self._loop_hit},
            })
        return message

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
        hook(message, "completed", {"observations_sha256": hash_json(observations),
                                    "observations": len(observations),
                                    "elapsed_ms": (time.monotonic() - start) * 1000})
        return observations

    def serialize(self, *extra_dicts):
        evidence = {"info": {"loop_detection": self._loop_hit}} if self._loop_hit is not None else {}
        return super().serialize(evidence, *extra_dicts)

    def save(self, path, *extra_dicts):
        data = self.serialize(*extra_dicts)
        if path:
            write_json(Path(path), data)
        return data
