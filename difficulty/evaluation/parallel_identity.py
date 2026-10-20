"""Exact tool-action identities and explicit reasoning identity evidence."""

import json

from core.io import hash_json, owned_path, read_json
from enrichment.online_trace import reasoning_blocks
from hidden.artifacts import read_request


class IdentityUnavailable(ValueError):
    """A legacy candidate lacks required identity evidence (not a bad checksum)."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate tool argument key")
        result[key] = value
    return result


def action_identity(candidate):
    """Ignore tool IDs/JSON key ordering; preserve call order and command bytes."""
    choices = (candidate.get("response") or {}).get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise IdentityUnavailable("missing_tool_calls")
    actions = []
    for call in calls:
        fn = call.get("function") or {}
        if not isinstance(fn.get("name"), str) or not fn["name"]:
            raise IdentityUnavailable("missing_tool_name")
        arguments = fn.get("arguments")
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments, object_pairs_hook=_object)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            hash_json(arguments)  # Reject NaN/Infinity without rewriting strings.
        except (ValueError, TypeError) as exc:
            raise IdentityUnavailable("invalid_tool_arguments") from exc
        actions.append({"name": fn["name"], "arguments": arguments})
    return hash_json(actions), json.dumps(actions, sort_keys=True, ensure_ascii=False, allow_nan=False)


class CandidateIdentityReader:
    def __init__(self, root, mode):
        if mode not in {"text", "tokens"}:
            raise ValueError("Reasoning identity must be text or tokens")
        self.root, self.mode, self.inventories, self.sources = root, mode, {}, {}

    def read(self, candidate, decision):
        action_hash, action = action_identity(candidate)
        prefix = decision.get("parent_prefix_sha256")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("Decision lacks parent-prefix identity")
        if self.mode == "text":
            msg = candidate["response"]["choices"][0]["message"]
            reasoning = msg.get("reasoning_content")
            if reasoning is None:
                reasoning = (msg.get("provider_specific_fields") or {}).get("reasoning")
            if not isinstance(reasoning, str):
                raise IdentityUnavailable("missing_reasoning_text")
            reasoning_hash = hash_json(reasoning)
        else:
            if not candidate.get("ledger_ref") or not candidate.get("ledger_sha256"):
                raise IdentityUnavailable("missing_native_ledger")
            request = read_request(self.root, candidate)
            for key, expected in [("request_id", candidate["native_request_id"]),
                                  ("decision_id", decision["decision_id"]),
                                  ("branch_id", candidate["branch_id"]),
                                  ("parent_prefix_sha256", prefix)]:
                if request.get(key) != expected:
                    raise ValueError(f"Native reasoning identity mismatch: {key}")
            prompt, output = request["prompt_token_ids"], request["output_token_ids"]
            if hash_json(prompt) != request["request_sha256"]:
                raise ValueError("Native prompt token checksum mismatch")
            inventory_ref = request.get("inventory_ref")
            if not inventory_ref:
                raise IdentityUnavailable("missing_reasoning_marker_inventory")
            inventory_key = (inventory_ref, request["inventory_sha256"])
            if inventory_key not in self.inventories:
                self.inventories[inventory_key] = read_json(owned_path(self.root, inventory_ref))
            inventory = self.inventories[inventory_key]
            blocks = reasoning_blocks(prompt, output, inventory["open_token_ids"], inventory["close_token_ids"])
            position = candidate.get("absolute_position")
            matches = [b for b in blocks if b["complete"] and position == len(prompt) + b["close"]]
            if len(matches) != 1:
                raise ValueError("Scored close does not identify exactly one complete reasoning block")
            block = matches[0]
            reasoning_hash = hash_json(output[block["start"]:block["end"]])
            self.sources[candidate["ledger_ref"]] = {"sha256": candidate["ledger_sha256"], "hash_kind": "file"}
            self.sources[inventory_ref] = {"sha256": request["inventory_sha256"], "hash_kind": "canonical_json"}
        return {"action_hash": action_hash, "action": action, "reasoning_hash": reasoning_hash,
                "reasoning_evidence": self.mode, "parent_prefix_sha256": prefix}
