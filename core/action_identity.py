"""Canonical exact identities for assistant tool actions."""

from __future__ import annotations

import json

from core.io import hash_json


class ActionIdentityUnavailable(ValueError):
    """The message does not contain a canonicalizable tool action."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate tool argument key")
        result[key] = value
    return result


def canonical_action(message):
    """Return ``(hash, canonical_json)`` while preserving exact command strings."""
    if not isinstance(message, dict):
        raise ActionIdentityUnavailable("invalid_message")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ActionIdentityUnavailable("missing_tool_calls")
    actions = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ActionIdentityUnavailable("missing_tool_name")
        fn = call["function"]
        if not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ActionIdentityUnavailable("missing_tool_name")
        arguments = fn.get("arguments")
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments, object_pairs_hook=_object)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            hash_json(arguments)
        except (ValueError, TypeError) as exc:
            raise ActionIdentityUnavailable("invalid_tool_arguments") from exc
        actions.append({"name": fn["name"], "arguments": arguments})
    return hash_json(actions), json.dumps(actions, sort_keys=True, ensure_ascii=False, allow_nan=False)


def action_identity(candidate):
    choices = (candidate.get("response") or {}).get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    return canonical_action(message)
