"""Native agent flow with bounded loop detection and atomic durable saves."""

import time
from collections import deque
from pathlib import Path

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded, InterruptAgentFlow
from pydantic import Field

from core.io import hash_json, write_json
from rollout.step_loop import repeated_suffix


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
        reset = getattr(self.model, "reset_loop_escape", None)
        if reset is not None:
            reset()
        self._loop_history.clear()
        self._loop_step = 0
        self._loop_hit = None
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)))
        return self.continue_run()

    def continue_run(self):
        """Native step/exception semantics, shared with fresh tasks; never reset H_t."""
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as error:
                self.n_consecutive_format_errors += 1
                self.add_messages(*error.messages)
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages({"role":"exit", "content":"RepeatedFormatError",
                        "extra":{"exit_status":"RepeatedFormatError", "submission":""}})
            except InterruptAgentFlow as error:
                self.add_messages(*error.messages)
            except Exception as error:
                self.handle_uncaught_exception(error)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                return self.messages[-1].get("extra", {})

    def query(self):
        recorder = getattr(self, "key_step_recorder", None)
        if recorder is not None:
            recorder.before_query(self)
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
        hit = repeated_suffix(self._loop_history, repeats, self.config.loop_max_period, step,
                              match_policy='exact_reasoning_content_and_actions_v1')
        if hit is not None:
            self._loop_hit = hit
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
        check = getattr(self.model, "check_cancelled", None)
        if check is not None:
            check()
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
        data = super().serialize(evidence, *extra_dicts)
        reference = getattr(self, '_continuation_ref', None)
        if reference is not None:
            # The full H_t stays in memory for inference. Persist only the
            # original source pointer and this child's actual suffix.
            data['messages'] = data['messages'][reference['prefix_message_count']:]
            data['continuation'] = reference
        return data

    def save(self, path, *extra_dicts):
        data = self.serialize(*extra_dicts)
        if path:
            write_json(Path(path), data)
        return data
