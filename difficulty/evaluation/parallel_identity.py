"""Exact tool-action identities and explicit reasoning identity evidence."""

from core.action_identity import ActionIdentityUnavailable, action_identity as _action_identity
from core.io import hash_json, owned_path, read_json
from enrichment.online_trace import reasoning_blocks
from hidden.artifacts import read_request


class IdentityUnavailable(ActionIdentityUnavailable):
    """A legacy candidate lacks required identity evidence (not a bad checksum)."""


def action_identity(candidate):
    try:
        return _action_identity(candidate)
    except ActionIdentityUnavailable as exc:
        raise IdentityUnavailable(str(exc)) from exc


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
