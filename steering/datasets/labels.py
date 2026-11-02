"""Actual historical action agreement, independent of retention and scores."""

from __future__ import annotations

import json

from core.action_identity import ActionIdentityUnavailable, canonical_action

LABEL_RULE = "actual-consensus-action-v1"


def branch_action(branch):
    response = branch.get("response")
    if response is None:
        if branch.get("action") is None:
            return None, None, "missing_full_response"
        try:
            calls = [{"function": call} for call in json.loads(branch["action"])]
            digest, action = canonical_action({"tool_calls": calls})
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Invalid saved compact action") from exc
        if action != branch["action"]:
            raise ValueError("Saved action is not canonical")
        if branch.get("action_hash") is not None and branch["action_hash"] != digest:
            raise ValueError("Saved action hash differs from compact action")
        return None, None, "missing_full_response"
    choices = response.get("choices") or []
    if len(choices) != 1:
        return None, None, "missing_or_ambiguous_response_choice"
    try:
        digest, action = canonical_action(choices[0].get("message"))
    except ActionIdentityUnavailable as exc:
        if branch.get("action") is not None:
            raise ValueError("Saved action exists but raw response has no comparable action") from exc
        return None, None, str(exc)
    if branch.get("action") is not None and branch["action"] != action:
        raise ValueError("Saved action differs from full response")
    if branch.get("action_hash") is not None and branch["action_hash"] != digest:
        raise ValueError("Saved action hash differs from full response")
    if choices[0].get("finish_reason") not in {"stop", "tool_calls", "function_call"}:
        return digest, action, "incomplete_response"
    if str(branch.get("status", "")).lower() in {"failed", "cancelled", "canceled", "partial"}:
        return digest, action, "failed_cancelled_or_partial_response"
    return digest, action, None


def label_branch(*, action_hash, winner_hash, unavailable_reason=None):
    reason = unavailable_reason or (
        "missing_action_identity"
        if action_hash is None
        else "missing_winner_action"
        if winner_hash is None
        else None
    )
    if reason:
        return {"label": "excluded", "y": None, "label_reason": reason}
    y = int(action_hash == winner_hash)
    return {"label": "positive" if y else "negative", "y": y, "label_reason": LABEL_RULE}
