"""Web task evidence at the shared mini-agent execution boundary.

The agent, model requests, capture and retry loop remain in rollout.runner.
This module only publishes the browser's saved evaluation subject.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.io import hash_json, read_json, sha256, write_json
from rollout.agent import DurableAgent


def finalize_subject(task, out, context, env, submission, exit_status):
    """Flush real browser evidence, retaining an explicit incomplete subject on failure."""
    target = Path(out) / "web-subject"
    target.mkdir(parents=True, exist_ok=True)
    details, errors = {}, []
    if env is not None:
        try:
            details = env.finish()
        except Exception as exc:
            errors.append(repr(exc))
            details = getattr(env, "finish_evidence", {}) or {}
    outputs = []
    paths = {}
    for key, default in (("har_path", "network.har"), ("actions_path", "actions.jsonl")):
        source = details.get(key)
        if source is None:
            continue
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            errors.append("Missing regular browser output: " + key)
            continue
        destination = target / default
        if source.resolve() != destination.resolve():
            if destination.exists() and sha256(destination) != sha256(source):
                raise ValueError("Conflicting browser evidence: " + default)
            if not destination.exists():
                shutil.copyfile(source, destination)
        paths[key] = default
    if exit_status == "Submitted":
        try:
            from runtime.containers.webarena_adapter import validate_answer

            answer = json.loads(submission) if isinstance(submission, str) else submission
            answer = validate_answer(answer)
            if hasattr(answer, "model_dump"):
                answer = answer.model_dump(mode="json")
            answer_path = target / "agent_response.json"
            if answer_path.exists() and read_json(answer_path) != answer:
                raise ValueError("Conflicting immutable agent answer")
            write_json(answer_path, answer)
            paths["answer_path"] = answer_path.name
        except Exception as exc:
            errors.append("Invalid submitted response: " + repr(exc))
    for relative in paths.values():
        file = target / relative
        outputs.append({"path": relative, "sha256": sha256(file), "size": file.stat().st_size})
    complete = bool(exit_status == "Submitted" and not errors and details.get("har_flushed")
                    and all(key in paths for key in ("answer_path", "har_path", "actions_path")))
    manifest = {
        "schema_version": 1, "kind": "web_trace", "binding": context,
        "task_sha256": hash_json(task), "runtime_identity": details.get("runtime_identity", {}),
        "outputs": outputs, "complete": complete,
        "har_flushed": bool(details.get("har_flushed")), "re_evaluation": complete,
        "exit_status": exit_status, "errors": errors,
        **{key: paths.get(key) for key in ("answer_path", "har_path", "actions_path")},
    }
    path = target / "subject.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Web subject is already bound to other bytes")
    write_json(path, manifest)
    return path, errors


class WebArenaAgent(DurableAgent):
    """Keep one model response aligned to at most one real browser action."""

    def execute_actions(self, message):
        from minisweagent.exceptions import FormatError, InterruptAgentFlow, Submitted

        actions = message.get("extra", {}).get("actions", [])
        if len(actions) != 1:
            output = [{"returncode": 2, "output": (
                "WebArena protocol accepts exactly one browser action per response. "
                "Return one bash(command=\"browser ...\") call and do not combine actions."
            )}]
            message.setdefault("extra", {})["action_executed"] = False
            raise FormatError(*self.model.format_observation_messages(message, output, self.get_template_vars()))
        command = actions[0].get("command", "")
        if not isinstance(command, str) or not command.strip().startswith("browser "):
            output = [{"returncode": 2, "output": "Only the browser command protocol is available for this task."}]
            message.setdefault("extra", {})["action_executed"] = False
            raise FormatError(*self.model.format_observation_messages(message, output, self.get_template_vars()))
        try:
            return super().execute_actions(message)
        except InterruptAgentFlow as exc:
            # Environment exceptions bypass DefaultAgent's observation formatter.
            # Preserve the actual final tool-call/result pair, including stop.
            output = {"returncode": 0 if isinstance(exc, Submitted) else 2,
                      "output": "Browser task submitted" if isinstance(exc, Submitted) else
                                "Browser action interrupted: " + json.dumps(exc.messages, ensure_ascii=False)}
            observations = self.model.format_observation_messages(message, [output], self.get_template_vars())
            terminal = [item for item in exc.messages if item.get('role') == 'exit']
            raise type(exc)(*observations, *terminal) from exc
        except Exception as exc:
            # A timed-out mutation may have reached the site. Keep its uncertainty
            # in the trace and let the shared attempt policy end this environment.
            output = {"returncode": 2, "output": "Browser action outcome is unknown: " + type(exc).__name__}
            self.add_messages(*self.model.format_observation_messages(message, [output], self.get_template_vars()))
            raise
